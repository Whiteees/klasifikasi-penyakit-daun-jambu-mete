const input = document.querySelector("#image");
const preview = document.querySelector("#preview");
const uploadText = document.querySelector("#uploadText");
const uploadBox = document.querySelector(".upload-box");
const predictButton = document.querySelector("#predictButton");

input?.addEventListener("change", () => {
  const file = input.files?.[0];

  if (!file) {
    preview.style.display = "none";
    uploadText.style.display = "grid";
    uploadBox?.classList.remove("has-preview");
    if (predictButton) {
      predictButton.disabled = true;
    }
    return;
  }

  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
  uploadText.style.display = "none";
  uploadBox?.classList.add("has-preview");
  if (predictButton) {
    predictButton.disabled = false;
  }
});
