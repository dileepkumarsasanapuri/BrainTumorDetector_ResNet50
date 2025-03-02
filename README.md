# Brain Tumor Detection Using ResNet50

## 📌 Features
✅ **Pre-trained ResNet50 model** for accurate classification  
✅ **Flask backend** for real-time predictions  
✅ **User-friendly UI** using HTML, CSS, and JavaScript  
✅ **Supports image uploads** for tumor detection  
✅ **Deployable** on GitHub/Cloud platforms  

---

## 📂 Folder Structure
```
brain_tumor_project/
│── .venv/              # Virtual environment (ignored in Git)
│── dataset/            # Local dataset (ignored in Git)
│── saved_models/       # Directory for saving and loading the trained model
│   ├── brain_tumor_resnet50.keras # Trained ResNet50 model
│── static/             # Static files (CSS, JS, Images)
│   ├── uploads/        # Uploaded images for prediction
│   ├── script.js       # JavaScript for UI interaction
│   ├── style.css       # Styling for the frontend
│── templates/          # HTML templates (Flask frontend)
│   ├── index.html      # Main UI for image upload
│── .gitignore          # Ignore unnecessary files (e.g., dataset, .venv)
│── app.py              # Flask backend to handle requests
│── predict.py          # Script for making predictions
│── requirements.txt    # Dependencies list
│── train_resnet50.py   # Model training script
│── README.md           # Project documentation
```

---

## 🛠️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/braintumordetector_resnet50.git
cd braintumordetector_resnet50
```

### 2️⃣ Create a Virtual Environment *(Optional but Recommended)*
#### For Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```
#### For Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run Flask Application
```bash
python app.py
```

📌 **Ensure you have the trained model stored in** `saved_models/` **before running predictions.**

🚀 The app will run at: **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 🖼️ Usage
1. Open the web app in your browser.
2. Upload an MRI scan image.
3. The model will predict and display the tumor category.

---

## 🧠 Model Details
- **Architecture:** ResNet50 *(Pre-trained on ImageNet)*
- **Fine-tuned on:** Brain Tumor MRI Dataset
- **Classes:**
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor
