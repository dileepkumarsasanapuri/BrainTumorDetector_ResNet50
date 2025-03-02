document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");
    const fileNameDisplay = document.getElementById("fileName");
    const uploadBtn = document.getElementById("uploadBtn");
    const form = document.getElementById("uploadForm");
    const loading = document.getElementById("loading");

    // Reset UI on page load
    fileNameDisplay.textContent = "No file chosen";
    uploadBtn.disabled = true;

    // Enable upload button when file is selected
    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            fileNameDisplay.textContent = fileInput.files[0].name;
            uploadBtn.disabled = false;
        } else {
            fileNameDisplay.textContent = "No file chosen";
            uploadBtn.disabled = true;
        }
    });

    // Show loading spinner when form is submitted
    form.addEventListener("submit", function () {
        loading.classList.remove("hidden");
        uploadBtn.textContent = "Processing...";
        uploadBtn.disabled = true;
    });
});
