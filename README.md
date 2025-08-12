# Handwritten Digit Recognition using SVM + Flask

This project is a **Handwritten Digit Recognition** web application built with **Support Vector Machine (SVM)** for classification and **Flask** for the backend.  
It allows users to upload an image of a digit (0–9) and predicts the digit using a trained SVM model.

---

## Features
- **Machine Learning Model**: Trained with `sklearn`'s built-in `load_digits()` dataset.
- **Flask Web App**: Serves the model and handles predictions.
- **Frontend**: Simple and modern UI with image upload and preview.
- **Prediction Result**: Displays the predicted digit instantly.

---

## Tech Stack
- **Python 3**
- **Flask**
- **scikit-learn**
- **NumPy**
- **Pillow**
- **HTML, CSS, JavaScript**

---

## Project Structure
├── app.py # Flask application
├── model.py # Train & save the SVM model
├── svm_model.pkl # Saved model file
├── templates/
│ └── index.html # Frontend HTML
├── requirements.txt # Dependencies
└── README.md # Project documentation

---

## How to Run

### 1. Clone the repository
```bash
git clone <your-repo-link>
cd <your-project-folder>
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Train and save the model
```bash
python model.py
```
### 4. Run the flask app
```bash
python app.py
```
### 5. Open in browser
Visit: http://127.0.0.1:5000

---
