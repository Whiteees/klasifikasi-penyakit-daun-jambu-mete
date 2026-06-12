import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.preprocessing import image as keras_image


IMG_SIZE = (224, 224)
MODEL_PATH = Path(__file__).resolve().parent / "cashew_model.h5"
CLASS_NAMES = ["anthracnose", "healthy", "leaf miner", "red rust"]
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MIN_LEAF_COLOR_RATIO = 0.06
MIN_COLOR_SATURATION = 0.04
VERY_LOW_CASHEW_CONFIDENCE = 0.40
VERY_LOW_CASHEW_MARGIN = 0.08
VERY_HIGH_CASHEW_ENTROPY = 0.90
CLASS_DETAILS = {
    "anthracnose": {
        "status": "Penyakit Terdeteksi",
        "description": (
            "Penyakit jamur yang menyebabkan bercak gelap pada daun dan dapat "
            "menghambat pertumbuhan tanaman."
        ),
    },
    "healthy": {
        "status": "Daun Sehat",
        "description": (
            "Daun terlihat berada pada kondisi sehat berdasarkan pola visual "
            "yang dikenali oleh model."
        ),
    },
    "leaf miner": {
        "status": "Penyakit Terdeteksi",
        "description": (
            "Serangan hama yang biasanya meninggalkan jalur atau bercak tipis "
            "pada permukaan daun."
        ),
    },
    "red rust": {
        "status": "Penyakit Terdeteksi",
        "description": (
            "Penyakit yang ditandai bercak kemerahan atau oranye pada daun dan "
            "dapat menurunkan kualitas tanaman."
        ),
    },
}
UNKNOWN_DETAIL = {
    "class_name": "bukan daun jambu mete",
    "status": "Tidak Dikenali",
    "description": (
        "Gambar tidak cukup cocok dengan pola daun jambu mete yang dikenali "
        "model. Gunakan foto daun jambu mete yang jelas agar diagnosis "
        "penyakit tidak keliru."
    ),
}

app = Flask(__name__)
model = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_cashew_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"File model tidak ditemukan: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


def prepare_image(uploaded_image: Image.Image) -> np.ndarray:
    img = uploaded_image.convert("RGB").resize(IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0


def image_to_data_uri(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def numpy_to_data_uri(arr: np.ndarray) -> str:
    """Convert a numpy array (grayscale or BGR) to a PNG data URI."""
    _, buffer = cv2.imencode(".png", arr)
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def normalized_entropy(probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities.astype(np.float32), 1e-8, 1.0)
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / float(np.log(len(probabilities)))


def analyze_leaf_visuals(uploaded_image: Image.Image) -> dict:
    img = uploaded_image.convert("RGB").resize(IMG_SIZE)
    rgb = np.array(img, dtype=np.float32, copy=True) / 255.0
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    max_channel = np.max(rgb, axis=2)
    min_channel = np.min(rgb, axis=2)
    saturation = np.zeros_like(max_channel)
    np.divide(max_channel - min_channel, max_channel, out=saturation, where=max_channel > 0)

    green_leaf = (
        (green >= red * 0.82)
        & (green >= blue * 0.82)
        & (saturation >= 0.12)
        & (max_channel >= 0.12)
    )
    dry_or_rust_leaf = (
        (red >= 0.20)
        & (green >= 0.14)
        & (blue <= 0.55)
        & (red >= blue * 1.10)
        & (green >= blue * 0.85)
        & (saturation >= 0.10)
    )
    leaf_color_mask = green_leaf | dry_or_rust_leaf

    return {
        "leaf_color_ratio": float(np.mean(leaf_color_mask)),
        "mean_saturation": float(np.mean(saturation)),
    }


def should_reject_as_non_cashew(probabilities: np.ndarray, leaf_signals: dict) -> bool:
    sorted_probabilities = np.sort(probabilities)
    confidence = float(sorted_probabilities[-1])
    margin = float(sorted_probabilities[-1] - sorted_probabilities[-2])
    entropy = normalized_entropy(probabilities)

    has_leaf_colors = (
        leaf_signals["leaf_color_ratio"] >= MIN_LEAF_COLOR_RATIO
        and leaf_signals["mean_saturation"] >= MIN_COLOR_SATURATION
    )
    model_is_extremely_uncertain = (
        confidence < VERY_LOW_CASHEW_CONFIDENCE
        and margin < VERY_LOW_CASHEW_MARGIN
        and entropy > VERY_HIGH_CASHEW_ENTROPY
    )

    return (not has_leaf_colors) or (
        leaf_signals["leaf_color_ratio"] < 0.20 and model_is_extremely_uncertain
    )

def process_pcd_and_glcm(uploaded_image: Image.Image) -> dict:
    """
    Menerima PIL Image, lalu:
    1. Membaca gambar dan resize ke IMG_SIZE.
    2. Mengubah ke Grayscale menggunakan OpenCV.
    3. Menghitung Canny Edge Detection.
    4. Menghitung 4 fitur GLCM dasar (Contrast, Homogeneity, Energy, Correlation).
    5. Mengembalikan data URI gambar grayscale & canny + dictionary fitur GLCM.
    """
    img_rgb = uploaded_image.convert("RGB").resize(IMG_SIZE)
    img_np = np.array(img_rgb)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Canny Edge Detection
    canny = cv2.Canny(gray, threshold1=100, threshold2=200)

    # GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )
    contrast = round(float(graycoprops(glcm, "contrast")[0, 0]), 4)
    homogeneity = round(float(graycoprops(glcm, "homogeneity")[0, 0]), 4)
    energy = round(float(graycoprops(glcm, "energy")[0, 0]), 4)
    correlation = round(float(graycoprops(glcm, "correlation")[0, 0]), 4)

    return {
        "path_gray": numpy_to_data_uri(gray),
        "path_canny": numpy_to_data_uri(canny),
        "features": {
            "Contrast": contrast,
            "Homogeneity": homogeneity,
            "Energy": energy,
            "Correlation": correlation,
        },
    }

def get_solusi_pertanian(class_name: str) -> list[str]:
    """
    Mengembalikan rekomendasi tindakan penanganan berdasarkan kelas penyakit.
    """
    solusi_db = {
        "anthracnose": [
            "Pangkas dan musnahkan bagian daun yang terinfeksi.",
            "Gunakan fungisida berbahan tembaga atau mankozeb sesuai anjuran.",
            "Jaga jarak tanam dan sirkulasi udara agar kelembapan berkurang.",
        ],
        "healthy": [
            "Lanjutkan penyiraman dan pemupukan berimbang secara rutin.",
            "Pantau daun secara berkala untuk mendeteksi gejala sejak awal.",
            "Lakukan pemangkasan sanitasi agar tajuk tetap sehat.",
        ],
        "leaf miner": [
            "Kumpulkan dan musnahkan daun yang menunjukkan jalur serangan.",
            "Gunakan insektisida sistemik seperti abamektin atau imidakloprid sesuai anjuran.",
            "Pasang perangkap kuning untuk memantau populasi hama.",
        ],
        "red rust": [
            "Semprot fungisida heksaconazole atau propiconazole sesuai anjuran.",
            "Kurangi kelembapan dengan memangkas cabang yang terlalu rapat.",
            "Tambahkan pupuk kalium untuk membantu ketahanan tanaman.",
        ],
    }
    return solusi_db.get(
        class_name.lower(),
        [
            "Gunakan foto daun yang lebih jelas untuk diagnosis ulang.",
            "Konsultasikan sampel daun dengan ahli pertanian setempat.",
        ],
    )

def build_processing_steps(uploaded_image: Image.Image):
    input_image = uploaded_image.convert("RGB")
    resized_image = input_image.resize(IMG_SIZE)
    gray_image = input_image.convert("L").resize(IMG_SIZE)

    steps = [
        {
            "title": "Citra Input",
            "description": "Gambar asli yang diupload pengguna.",
            "image": input_image,
        },
        {
            "title": "Resize 224x224",
            "description": "Ukuran gambar disamakan dengan input model.",
            "image": resized_image,
        },
        {
            "title": "Grayscale",
            "description": "Warna diubah menjadi intensitas abu-abu.",
            "image": gray_image,
        },
    ]

    return [
        {
            "title": step["title"],
            "description": step["description"],
            "image": image_to_data_uri(step["image"]),
        }
        for step in steps
    ]


def predict(uploaded_image: Image.Image):
    loaded_model = load_cashew_model()
    prepared_image = prepare_image(uploaded_image)
    probabilities = loaded_model.predict(prepared_image, verbose=0)[0]
    leaf_signals = analyze_leaf_visuals(uploaded_image)
    best_index = int(np.argmax(probabilities))
    best_class = CLASS_NAMES[best_index]
    confidence = float(probabilities[best_index])
    class_detail = CLASS_DETAILS[best_class]
    is_unknown = should_reject_as_non_cashew(probabilities, leaf_signals)

    if confidence > 0.8:
        confidence_level = "Tinggi"
    elif confidence >= 0.6:
        confidence_level = "Sedang"
    else:
        confidence_level = "Rendah"

    if is_unknown:
        return {
            "class_name": UNKNOWN_DETAIL["class_name"],
            "matched_class": best_class,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "status": UNKNOWN_DETAIL["status"],
            "description": UNKNOWN_DETAIL["description"],
            "is_healthy": False,
            "is_unknown": True,
            "display_type": "unknown",
            "probabilities": [
                {"name": class_name, "score": float(score)}
                for class_name, score in zip(CLASS_NAMES, probabilities)
            ],
        }

    return {
        "class_name": best_class,
        "matched_class": best_class,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "status": class_detail["status"],
        "description": class_detail["description"],
        "is_healthy": best_class == "healthy",
        "is_unknown": False,
        "display_type": "healthy" if best_class == "healthy" else "disease",
        "probabilities": [
            {"name": class_name, "score": float(score)}
            for class_name, score in zip(CLASS_NAMES, probabilities)
        ],
    }


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/analysis", methods=["GET", "POST"])
def analysis():
    result = None
    error = None
    processing_steps = None
    glcm_data = None
    solusi_text = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if not uploaded_file or uploaded_file.filename == "":
            error = "Pilih gambar daun terlebih dahulu."
        elif not allowed_file(uploaded_file.filename):
            error = "Format gambar harus jpg, jpeg, png, bmp, atau webp."
        else:
            try:
                image_bytes = uploaded_file.read()
                uploaded_image = Image.open(BytesIO(image_bytes))
                result = predict(uploaded_image)
                processing_steps = build_processing_steps(uploaded_image)

                # Pengolahan citra + GLCM 
                glcm_data = process_pcd_and_glcm(uploaded_image)

                # Solusi Pertanian berdasarkan kelas yang terdeteksi
                detected_class = result.get("matched_class", "")
                solusi_text = get_solusi_pertanian(detected_class)
            except Exception as exc:
                error = f"Gagal memproses gambar: {exc}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        processing_steps=processing_steps,
        glcm_data=glcm_data,
        solusi_text=solusi_text,
        model_name=MODEL_PATH.name,
    )


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
