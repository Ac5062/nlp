"""
Feature Engineering Module
==========================
Handles TF-IDF vectorization and additional feature extraction
for the Question Quality Evaluator.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import joblib
import os


class FeatureEngineer:
    """Extracts features from preprocessed text data."""

    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        """
        Initialize the feature engineer.

        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: Range of n-grams for TF-IDF (default: unigrams and bigrams)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,       # Apply log normalization
            min_df=2,                # Ignore very rare terms
            max_df=0.95,             # Ignore very common terms
            strip_accents='unicode'
        )
        self.is_fitted = False

    def extract_text_features(self, df, text_column='cleaned_text'):
        """
        Extract handcrafted text features from the DataFrame.

        Features extracted:
        - word_count: Number of words
        - char_count: Number of characters
        - avg_word_length: Average word length
        - sentence_count: Approximate sentence count
        - question_mark_count: Number of question marks
        - code_indicator: Whether the original text contained code
        - has_url: Whether the original text contained URLs
        - title_length: Length of the title
        - uppercase_ratio: Ratio of uppercase letters

        Args:
            df: DataFrame with text data
            text_column: Column name for cleaned text

        Returns:
            DataFrame with additional feature columns
        """
        df = df.copy()

        # Use the original combined_text for some features, cleaned for others
        original_col = 'combined_text' if 'combined_text' in df.columns else text_column
        clean_col = text_column

        # Word count from cleaned text
        df['word_count'] = df[clean_col].apply(lambda x: len(str(x).split()))

        # Character count
        df['char_count'] = df[clean_col].apply(lambda x: len(str(x)))

        # Average word length
        df['avg_word_length'] = df[clean_col].apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
        )

        # Sentence count (approximate from original text)
        df['sentence_count'] = df[original_col].apply(
            lambda x: max(1, len([s for s in str(x).split('.') if s.strip()]))
        )

        # Question mark count (from original text)
        df['question_mark_count'] = df[original_col].apply(
            lambda x: str(x).count('?')
        )

        # Code indicator (from original text)
        df['has_code'] = df[original_col].apply(
            lambda x: 1 if ('<code>' in str(x).lower() or '```' in str(x)) else 0
        )

        # URL indicator
        df['has_url'] = df[original_col].apply(
            lambda x: 1 if ('http' in str(x).lower() or 'www.' in str(x).lower()) else 0
        )

        # Title length (if available)
        if 'Title' in df.columns:
            df['title_length'] = df['Title'].apply(lambda x: len(str(x).split()))
        else:
            df['title_length'] = 0

        # Uppercase ratio (from original text)
        df['uppercase_ratio'] = df[original_col].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )

        # Unique word ratio
        df['unique_word_ratio'] = df[clean_col].apply(
            lambda x: len(set(str(x).split())) / max(len(str(x).split()), 1)
        )

        return df

    def fit_tfidf(self, texts):
        """
        Fit the TF-IDF vectorizer on training texts.

        Args:
            texts: Series or list of cleaned text strings
        """
        print(f"Fitting TF-IDF vectorizer (max_features={self.max_features}, "
              f"ngram_range={self.ngram_range})...")
        self.tfidf_vectorizer.fit(texts)
        self.is_fitted = True
        vocab_size = len(self.tfidf_vectorizer.vocabulary_)
        print(f"TF-IDF vocabulary size: {vocab_size}")

    def transform_tfidf(self, texts):
        """
        Transform texts to TF-IDF feature matrix.

        Args:
            texts: Series or list of cleaned text strings

        Returns:
            Sparse matrix of TF-IDF features
        """
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer has not been fitted. Call fit_tfidf() first.")
        return self.tfidf_vectorizer.transform(texts)

    def fit_transform_tfidf(self, texts):
        """
        Fit and transform texts to TF-IDF feature matrix.

        Args:
            texts: Series or list of cleaned text strings

        Returns:
            Sparse matrix of TF-IDF features
        """
        self.fit_tfidf(texts)
        return self.transform_tfidf(texts)

    def get_combined_features(self, df, tfidf_matrix, feature_columns=None):
        """
        Combine TF-IDF features with handcrafted features.

        Args:
            df: DataFrame with handcrafted features
            tfidf_matrix: Sparse TF-IDF matrix
            feature_columns: List of handcrafted feature column names

        Returns:
            Combined sparse feature matrix
        """
        if feature_columns is None:
            feature_columns = [
                'word_count', 'char_count', 'avg_word_length',
                'sentence_count', 'question_mark_count', 'has_code',
                'has_url', 'title_length', 'uppercase_ratio', 'unique_word_ratio'
            ]

        # Filter to columns that exist in the DataFrame
        available_cols = [c for c in feature_columns if c in df.columns]

        if available_cols:
            handcrafted = csr_matrix(df[available_cols].values.astype(float))
            combined = hstack([tfidf_matrix, handcrafted])
            print(f"Combined features shape: {combined.shape} "
                  f"(TF-IDF: {tfidf_matrix.shape[1]}, handcrafted: {len(available_cols)})")
        else:
            combined = tfidf_matrix
            print(f"Using TF-IDF features only, shape: {combined.shape}")

        return combined

    def save(self, filepath):
        """Save the fitted vectorizer to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"Feature engineer saved to {filepath}")

    def load(self, filepath):
        """Load a fitted vectorizer from disk."""
        data = joblib.load(filepath)
        self.tfidf_vectorizer = data['tfidf_vectorizer']
        self.max_features = data['max_features']
        self.ngram_range = data['ngram_range']
        self.is_fitted = data['is_fitted']
        print(f"Feature engineer loaded from {filepath}")


if __name__ == "__main__":
    # Quick test
    sample_texts = [
        "how sort list python efficiently",
        "what",
        "implement binary search tree java explain time complexity",
    ]
    fe = FeatureEngineer(max_features=100)
    tfidf = fe.fit_transform_tfidf(sample_texts)
    print(f"TF-IDF shape: {tfidf.shape}")
