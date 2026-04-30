import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "cashew_model.h5"
PLOT_PATH = "training_history.png"

CLASS_FOLDERS = ["healthy", "anthracnose", "leaf miner", "red rust"]


def find_dataset_dirs(dataset_dir: Path) -> tuple[Path, Path | None]:
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    if not train_dir.exists():
        train_dir = dataset_dir / "train_set"
    if not test_dir.exists():
        test_dir = dataset_dir / "test_set"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Folder train tidak ditemukan. Dicari di: {dataset_dir / 'train'} "
            f"atau {dataset_dir / 'train_set'}"
        )

    return train_dir, test_dir if test_dir.exists() else None


def create_generators(train_dir: Path, test_dir: Path | None):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        rotation_range=30,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        classes=CLASS_FOLDERS,
        shuffle=True,
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        classes=CLASS_FOLDERS,
        shuffle=False,
    )

    test_generator = None
    if test_dir is not None:
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            classes=CLASS_FOLDERS,
            shuffle=False,
        )

    return train_generator, validation_generator, test_generator


def build_model(num_classes: int) -> Model:
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def plot_history(history, output_path: str = PLOT_PATH):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()


def train(dataset_dir: Path):
    train_dir, test_dir = find_dataset_dirs(dataset_dir)
    train_generator, validation_generator, test_generator = create_generators(train_dir, test_dir)

    model = build_model(num_classes=len(CLASS_FOLDERS))

    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    plot_history(history)

    if test_generator is not None and Path(MODEL_PATH).exists():
        best_model = load_model(MODEL_PATH)
        test_loss, test_accuracy = best_model.evaluate(test_generator)
        print(f"Test Loss     : {test_loss:.4f}")
        print(f"Test Accuracy : {test_accuracy:.4f}")

    print(f"Model terbaik disimpan ke: {Path(MODEL_PATH).resolve()}")
    print(f"Plot training disimpan ke: {Path(PLOT_PATH).resolve()}")


def predict_image(model_path: str, image_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    model = load_model(model_path)
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)[0]
    best_idx = int(np.argmax(predictions))

    print("\nHasil Prediksi")
    print(f"Kelas       : {CLASS_FOLDERS[best_idx]}")
    print(f"Confidence  : {predictions[best_idx] * 100:.2f}%")
    print("\nProbabilitas semua kelas:")
    for class_name, score in zip(CLASS_FOLDERS, predictions):
        print(f"- {class_name:<12}: {score * 100:.2f}%")


def open_interactive_predictor(model_path: str):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(
        title="Pilih gambar daun jambu mete",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if image_path:
        predict_image(model_path, image_path)
    else:
        print("Tidak ada gambar yang dipilih.")


def main():
    parser = argparse.ArgumentParser(
        description="Klasifikasi penyakit daun jambu mete dengan ResNet50 Transfer Learning."
    )
    parser.add_argument(
        "--dataset",
        default="D:/PC/Cashew",
        help="Path dataset yang berisi folder train/train_set dan test/test_set.",
    )
    parser.add_argument(
        "--predict",
        help="Path gambar yang ingin diprediksi menggunakan model tersimpan.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Buka file picker untuk memilih gambar yang ingin diprediksi.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="Path model .h5 untuk prediksi.",
    )
    args = parser.parse_args()

    if args.predict:
        predict_image(args.model, args.predict)
    elif args.interactive:
        open_interactive_predictor(args.model)
    else:
        train(Path(args.dataset))


if __name__ == "__main__":
    main()
