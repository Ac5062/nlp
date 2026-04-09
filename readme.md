# Question Quality Evaluator using NLP

An NLP system that automatically evaluates the quality of user-generated questions, classifying them as **Low**, **Medium**, or **High** quality based on clarity, specificity, and answerability.

## Dataset

This project uses the **[60k Stack Overflow Questions with Quality Rating](https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate)** dataset from Kaggle, containing 60,000 questions classified into three categories:

| Label | Quality | Description |
|-------|---------|-------------|
| HQ | High Quality | Score > 30, never edited |
| LQ_EDIT | Medium Quality | Low score, community-edited but still open |
| LQ_CLOSE | Low Quality | Closed by community |

## Project Structure

```
nlp/
├── data/                       # Dataset directory
│   └── sample_dataset.csv      # Generated sample (or place Kaggle CSV here)
├── models/                     # Saved trained models
│   ├── question_quality_model.pkl
│   └── feature_engineer.pkl
├── outputs/                    # Evaluation outputs
│   ├── confusion_matrix.png
│   ├── metrics_comparison.png
│   └── evaluation_report.txt
├── src/                        # Source modules
│   ├── __init__.py
│   ├── preprocessing.py        # Text cleaning & tokenization
│   ├── feature_engineering.py  # TF-IDF & handcrafted features
│   ├── model.py                # Logistic Regression model
│   └── evaluate.py             # Metrics & visualization
├── app.py                      # Streamlit web interface
├── train.py                    # Main training pipeline
├── download_dataset.py         # Kaggle dataset downloader
├── generate_sample_data.py     # Sample dataset generator
├── requirements.txt            # Python dependencies
└── README.md
```

## Setup and Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NLTK data (handled automatically on first run)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Getting the Dataset

**Option A: Download from Kaggle (recommended)**
```bash
pip install kaggle
# Set up API key: https://www.kaggle.com/settings -> API -> Create New Token
python download_dataset.py
```

**Option B: Use generated sample data**
```bash
python generate_sample_data.py
```

## Training

```bash
# Train with sample dataset
python train.py

# Train with Kaggle dataset
python train.py --data data/train.csv

# Train with hyperparameter tuning
python train.py --data data/train.csv --tune

# Custom options
python train.py --data data/train.csv --max_features 15000 --test_size 0.25
```

## Running the Web App

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Pipeline Overview

1. **Data Loading** - Load CSV dataset (Kaggle or sample)
2. **Text Preprocessing** - HTML removal, lowercasing, tokenization, stopword removal, lemmatization
3. **Feature Engineering** - TF-IDF vectorization (unigrams + bigrams) combined with handcrafted features (word count, code presence, question marks, etc.)
4. **Model Training** - Logistic Regression with balanced class weights
5. **Evaluation** - Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC AUC

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.8+ |
| ML | scikit-learn |
| NLP | NLTK |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Web UI | Streamlit |