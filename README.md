# 📱 SMS Spam Detection using Naive Bayes and TF-IDF

A complete Streamlit machine-learning project for classifying SMS messages as **Spam** or **Ham (Legitimate)**.

## Features

- Text preprocessing
- TF-IDF feature extraction
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM
- Accuracy, Precision, Recall and F1 Score
- Interactive SMS prediction
- CSV dataset upload
- Supports common datasets using `label/message` or `v1/v2`

## Project structure

```text
SMS-Spam-Detection/
├── app.py
├── requirements.txt
└── README.md
```

## Dataset format

Recommended:

```csv
label,message
ham,Hey how are you?
spam,Congratulations! You won a free prize!
```

The common SMS Spam Collection format also works:

```csv
v1,v2
ham,Hello how are you?
spam,WIN a free prize now!
```

## Run locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd SMS-Spam-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Streamlit

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `README.md` to GitHub.
2. Open Streamlit Community Cloud.
3. Select your GitHub repository.
4. Select `app.py` as the main file.
5. Deploy the application.

The application includes a small demo dataset, so it can run without an uploaded CSV. For your final project, upload your actual SMS dataset through the sidebar.

## Technologies

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF
- Naive Bayes
- Logistic Regression
- Linear SVM

## Project workflow

```text
SMS Input
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Machine Learning Model
   ↓
Spam / Ham Prediction
```

## Author

Add your name here before submitting the project.
