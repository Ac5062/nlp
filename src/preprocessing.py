"""
Data Preprocessing Module
=========================
Handles text cleaning, tokenization, and stopword removal
for the Question Quality Evaluator.
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
def download_nltk_data():
    """Download required NLTK resources."""
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {e}")


class TextPreprocessor:
    """Preprocesses text data for NLP pipeline."""

    def __init__(self):
        download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def remove_html_tags(self, text):
        """Remove HTML tags from text."""
        if not isinstance(text, str):
            return ""
        clean = re.compile(r'<.*?>')
        return re.sub(clean, ' ', text)

    def remove_urls(self, text):
        """Remove URLs from text."""
        url_pattern = re.compile(
            r'https?://\S+|www\.\S+|ftp://\S+'
        )
        return url_pattern.sub(' ', text)

    def remove_code_blocks(self, text):
        """Remove code blocks (common in Stack Overflow questions)."""
        # Remove content between code tags
        text = re.sub(r'<code>.*?</code>', ' ', text, flags=re.DOTALL)
        # Remove content between backticks
        text = re.sub(r'`[^`]*`', ' ', text)
        return text

    def remove_special_characters(self, text):
        """Remove special characters and digits, keeping only letters and spaces."""
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return text

    def normalize_whitespace(self, text):
        """Normalize multiple spaces to single space and strip."""
        return ' '.join(text.split()).strip()

    def to_lowercase(self, text):
        """Convert text to lowercase."""
        return text.lower()

    def tokenize(self, text):
        """Tokenize text into words."""
        try:
            return word_tokenize(text)
        except Exception:
            return text.split()

    def remove_stopwords(self, tokens):
        """Remove stopwords from token list."""
        return [t for t in tokens if t not in self.stop_words]

    def lemmatize(self, tokens):
        """Lemmatize tokens."""
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def preprocess_text(self, text):
        """
        Full preprocessing pipeline for a single text string.

        Steps:
        1. Remove HTML tags
        2. Remove URLs
        3. Remove code blocks
        4. Convert to lowercase
        5. Remove special characters
        6. Normalize whitespace
        7. Tokenize
        8. Remove stopwords
        9. Lemmatize
        10. Rejoin tokens

        Returns:
            str: Cleaned and preprocessed text
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""

        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.remove_code_blocks(text)
        text = self.to_lowercase(text)
        text = self.remove_special_characters(text)
        text = self.normalize_whitespace(text)

        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)

        return ' '.join(tokens)

    def preprocess_dataframe(self, df, text_column='Body', title_column='Title'):
        """
        Preprocess an entire DataFrame.

        Args:
            df: pandas DataFrame with text data
            text_column: Name of the column containing question body
            title_column: Name of the column containing question title

        Returns:
            DataFrame with added 'cleaned_text' column
        """
        df = df.copy()

        # Combine title and body for richer features
        if title_column in df.columns and text_column in df.columns:
            df['combined_text'] = df[title_column].fillna('') + ' ' + df[text_column].fillna('')
        elif title_column in df.columns:
            df['combined_text'] = df[title_column].fillna('')
        elif text_column in df.columns:
            df['combined_text'] = df[text_column].fillna('')
        else:
            raise ValueError(f"Neither '{title_column}' nor '{text_column}' found in DataFrame")

        print("Preprocessing text data...")
        df['cleaned_text'] = df['combined_text'].apply(self.preprocess_text)

        # Remove empty rows after preprocessing
        empty_mask = df['cleaned_text'].str.strip() == ''
        if empty_mask.any():
            print(f"Removing {empty_mask.sum()} empty rows after preprocessing")
            df = df[~empty_mask].reset_index(drop=True)

        print(f"Preprocessing complete. {len(df)} samples ready.")
        return df


def load_dataset(filepath):
    """
    Load dataset from CSV file.

    Args:
        filepath: Path to the CSV file

    Returns:
        pandas DataFrame
    """
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df


def encode_labels(df, label_column='Y'):
    """
    Encode text labels to numeric values.

    Mapping:
        LQ_CLOSE -> 0 (Low Quality)
        LQ_EDIT  -> 1 (Medium Quality)
        HQ       -> 2 (High Quality)

    Args:
        df: DataFrame with label column
        label_column: Name of the label column

    Returns:
        DataFrame with added 'label' and 'label_name' columns
    """
    df = df.copy()
    label_map = {
        'LQ_CLOSE': 0,
        'LQ_EDIT': 1,
        'HQ': 2
    }
    label_names = {
        0: 'Low Quality',
        1: 'Medium Quality',
        2: 'High Quality'
    }

    if label_column in df.columns:
        df['label'] = df[label_column].map(label_map)
        df['label_name'] = df['label'].map(label_names)

        # Handle unmapped labels
        unmapped = df['label'].isna()
        if unmapped.any():
            print(f"Warning: {unmapped.sum()} rows with unmapped labels")
            df = df.dropna(subset=['label']).reset_index(drop=True)
            df['label'] = df['label'].astype(int)
    else:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame")

    print(f"Label distribution:\n{df['label_name'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    # Quick test
    preprocessor = TextPreprocessor()
    sample = "<p>How do I <code>sort</code> a list in Python? https://example.com</p>"
    print(f"Original: {sample}")
    print(f"Cleaned:  {preprocessor.preprocess_text(sample)}")
