## Overview
This project focuses on predicting a person's **age** and **gender** from facial images using **Deep Learning**.  
The model is trained on the **UTKFace Dataset** and uses **Transfer Learning** with Convolutional Neural Networks (CNNs) to improve performance and reduce training time.

The application is also deployed as a web app where users can upload images and get real-time predictions for age and gender.

---

## Features
- Predicts **Age** from facial images
- Classifies **Gender** (Male/Female)
- Uses **Transfer Learning**
- Trained on the **UTKFace Dataset**
- Interactive web interface for predictions
- Real-time image upload and inference

---

## Dataset
The project uses the **UTKFace Dataset**, which contains face images labeled with:
- Age
- Gender
- Ethnicity

Dataset filename format:
```bash
[age]_[gender]_[race]_[date&time].jpg
```

Example:
```bash
25_0_2_20170116174525125.jpg
```
- Age = 25
- Gender = 0 (Male)
- Race = 2

---

## Model Architecture
The project uses:
- CNN-based architecture
- Transfer Learning
- Separate outputs for:
  - Age Prediction (Regression)
  - Gender Classification (Binary Classification)

---

## Evaluation Metrics

### Gender Classification Accuracy
- **85.87%**

### Age Prediction Accuracy (±3)
- **79.36%**

### Additional Metrics Used
- Mean Absolute Error (MAE)
- Loss Curves
- Confusion Matrix
- Prediction Visualization

---

## Deployed Version

You can access the deployed application here:

[Click here to view the web app](https://facereadai.streamlit.app/)

---

## Tech Stack
- Python
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/shindevaish/age_gender_classifier_cnn.git
cd age_gender_classifier_cnn
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

---