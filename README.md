# Pulse | Sentiment Intelligence

A Streamlit workspace for classifying the emotional tone of text with classical machine-learning models.

[Live Demo](https://sentiment-analysis-with-machine-learning-by-ahnaf.streamlit.app/)

![Pulse sentiment analysis interface](docs/ui-preview.png)

## Overview

Pulse cleans and analyzes text from the included `data.txt` dataset, then predicts one of six emotions:

- Joy
- Love
- Surprise
- Anger
- Sadness
- Fear

The interface lets you choose a model, enter a message, view the prediction with an emotion emoji, and inspect the selected model's evaluation metrics.

## Models

- Naive Bayes with Bag of Words
- Naive Bayes with TF-IDF
- Logistic Regression with TF-IDF
- Support Vector Machine with TF-IDF

Logistic Regression is selected by default. SVM is also available and currently performs best on the held-out test split used by the app.

## Features

- Premium dark analysis-console UI
- Model selector with dynamic Accuracy, Precision, Recall, and F1 score
- Emotion-specific prediction colors and emoji
- Cached model training for faster Streamlit reruns
- Stratified 80/20 train-test split
- Text preprocessing with lowercasing, punctuation and number removal, ASCII normalization, and English stop-word removal

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/iamahnaf/Sentiment-Analysis-with-Machine-Learning.git
cd Sentiment-Analysis-with-Machine-Learning
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Start Streamlit

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Deploy with Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Choose **New app**.
4. Select this repository and the `main` branch.
5. Set the main file to `app.py`.
6. Deploy.

Streamlit will install the packages from `requirements.txt` and use `data.txt` from the repository at runtime.

## Project Structure

```text
.
├── app.py                    # Streamlit application
├── data.txt                  # Labeled sentiment dataset
├── requirements.txt          # Python dependencies
├── Sentiment_Analysis.ipynb  # Exploratory notebook and model comparison
└── docs/
    └── ui-preview.png       # Application screenshot
```

## Evaluation

Metrics are calculated from the held-out test data using weighted averages for precision, recall, and F1 score. The app displays metrics for the currently selected model rather than showing a full comparison table.

## License

This project is intended for educational and demonstration purposes.
