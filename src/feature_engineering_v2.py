"""
Feature Engineering V2 - Improved
=================================
Key improvements over V1:
1. Extracts structural features from ORIGINAL text (before cleaning)
2. Adds 15+ new quality-signal features
3. Separate title vs body feature extraction
4. Better TF-IDF configuration with char n-grams option
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import joblib
import os


class FeatureEngineerV2:
    """Enhanced feature extraction with structural quality signals."""

    def __init__(self, max_features=15000, ngram_range=(1, 3)):
        self.max_features = max_features
        self.ngram_range = ngram_range

        # Main TF-IDF on cleaned body+title
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=3,
            max_df=0.90,
            strip_accents='unicode',
            analyzer='word'
        )

        # Character-level TF-IDF (captures patterns like code syntax)
        self.char_tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(2, 5),
            analyzer='char_wb',
            sublinear_tf=True,
            min_df=3,
            max_df=0.90
        )

        self.is_fitted = False

    def extract_structural_features(self, df):
        """
        Extract features from ORIGINAL text (before preprocessing).
        This preserves quality signals that cleaning destroys.
        """
        df = df.copy()

        # Use original Body and Title columns
        body = df['Body'].fillna('') if 'Body' in df.columns else pd.Series([''] * len(df))
        title = df['Title'].fillna('') if 'Title' in df.columns else pd.Series([''] * len(df))
        original = title + ' ' + body

        # ── Title Features ──────────────────────────────
        df['title_word_count'] = title.apply(lambda x: len(str(x).split()))
        df['title_char_count'] = title.apply(lambda x: len(str(x)))
        df['title_has_question_mark'] = title.apply(lambda x: 1 if '?' in str(x) else 0)
        df['title_starts_with_how'] = title.apply(
            lambda x: 1 if str(x).lower().strip().startswith(('how', 'what', 'why', 'when', 'where', 'which')) else 0
        )

        # ── Body Features (from ORIGINAL HTML) ──────────
        df['body_word_count'] = body.apply(lambda x: len(str(x).split()))
        df['body_char_count'] = body.apply(lambda x: len(str(x)))

        # Code blocks — STRONG quality signal
        df['code_block_count'] = body.apply(
            lambda x: len(re.findall(r'<code>', str(x), re.IGNORECASE))
        )
        df['has_code'] = (df['code_block_count'] > 0).astype(int)

        # Paragraph structure
        df['paragraph_count'] = body.apply(
            lambda x: max(1, len(re.findall(r'<p>', str(x), re.IGNORECASE)))
        )

        # Links / references
        df['link_count'] = body.apply(
            lambda x: len(re.findall(r'<a\s+href|https?://', str(x), re.IGNORECASE))
        )
        df['has_links'] = (df['link_count'] > 0).astype(int)

        # Error message patterns — quality signal
        df['has_error_message'] = body.apply(
            lambda x: 1 if re.search(
                r'(error|exception|traceback|stacktrace|failed|TypeError|ValueError|'
                r'NullPointer|undefined|segfault|FATAL|ImportError|SyntaxError)',
                str(x), re.IGNORECASE
            ) else 0
        )

        # List structure (ordered or unordered)
        df['has_list'] = body.apply(
            lambda x: 1 if re.search(r'<[uo]l>|<li>', str(x), re.IGNORECASE) else 0
        )

        # ── Punctuation & Style Features (ORIGINAL text) ──
        df['question_mark_count'] = original.apply(lambda x: str(x).count('?'))
        df['exclamation_count'] = original.apply(lambda x: str(x).count('!'))
        df['caps_word_ratio'] = original.apply(
            lambda x: sum(1 for w in str(x).split() if w.isupper() and len(w) > 1) / max(len(str(x).split()), 1)
        )

        # Multiple question marks/exclamations = low quality signal
        df['excessive_punctuation'] = original.apply(
            lambda x: 1 if re.search(r'[?!]{2,}', str(x)) else 0
        )

        # "please help" / "urgent" patterns = low quality
        df['has_plea'] = original.apply(
            lambda x: 1 if re.search(
                r'\b(plz|pls|please help|help me|urgent|asap|homework|assignment)\b',
                str(x), re.IGNORECASE
            ) else 0
        )

        # ── Tag Features ────────────────────────────────
        if 'Tags' in df.columns:
            df['tag_count'] = df['Tags'].apply(
                lambda x: len(re.findall(r'<([^>]+)>', str(x)))
            )
        else:
            df['tag_count'] = 0

        # ── Derived Ratios ──────────────────────────────
        df['code_to_text_ratio'] = df['code_block_count'] / (df['body_word_count'] + 1)
        df['title_to_body_ratio'] = df['title_word_count'] / (df['body_word_count'] + 1)

        # ── From cleaned text ───────────────────────────
        cleaned = df['cleaned_text'] if 'cleaned_text' in df.columns else original
        df['cleaned_word_count'] = cleaned.apply(lambda x: len(str(x).split()))
        df['unique_word_ratio'] = cleaned.apply(
            lambda x: len(set(str(x).split())) / max(len(str(x).split()), 1)
        )
        df['avg_word_length'] = cleaned.apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
        )

        return df

    def get_structural_feature_columns(self):
        """Return list of structural feature column names."""
        return [
            'title_word_count', 'title_char_count', 'title_has_question_mark',
            'title_starts_with_how',
            'body_word_count', 'body_char_count',
            'code_block_count', 'has_code',
            'paragraph_count', 'link_count', 'has_links',
            'has_error_message', 'has_list',
            'question_mark_count', 'exclamation_count', 'caps_word_ratio',
            'excessive_punctuation', 'has_plea', 'tag_count',
            'code_to_text_ratio', 'title_to_body_ratio',
            'cleaned_word_count', 'unique_word_ratio', 'avg_word_length'
        ]

    def fit_tfidf(self, texts):
        print(f"Fitting word TF-IDF (max={self.max_features}, ngram={self.ngram_range})...")
        self.tfidf_vectorizer.fit(texts)
        print(f"  Word vocab size: {len(self.tfidf_vectorizer.vocabulary_)}")

        print(f"Fitting char TF-IDF (max=5000, ngram=(2,5))...")
        self.char_tfidf.fit(texts)
        print(f"  Char vocab size: {len(self.char_tfidf.vocabulary_)}")

        self.is_fitted = True

    def transform_tfidf(self, texts):
        if not self.is_fitted:
            raise ValueError("Not fitted. Call fit_tfidf() first.")
        word_features = self.tfidf_vectorizer.transform(texts)
        char_features = self.char_tfidf.transform(texts)
        return hstack([word_features, char_features])

    def fit_transform_tfidf(self, texts):
        self.fit_tfidf(texts)
        return self.transform_tfidf(texts)

    def get_combined_features(self, df, tfidf_matrix):
        """Combine TF-IDF + char TF-IDF + structural features."""
        feature_cols = self.get_structural_feature_columns()
        available_cols = [c for c in feature_cols if c in df.columns]

        if available_cols:
            structural = csr_matrix(df[available_cols].values.astype(float))
            combined = hstack([tfidf_matrix, structural])
            print(f"Combined: {combined.shape} "
                  f"(TF-IDF: {tfidf_matrix.shape[1]}, structural: {len(available_cols)})")
        else:
            combined = tfidf_matrix
        return combined

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'char_tfidf': self.char_tfidf,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"Feature engineer V2 saved to {filepath}")

    def load(self, filepath):
        data = joblib.load(filepath)
        self.tfidf_vectorizer = data['tfidf_vectorizer']
        self.char_tfidf = data['char_tfidf']
        self.max_features = data['max_features']
        self.ngram_range = data['ngram_range']
        self.is_fitted = data['is_fitted']
        print(f"Feature engineer V2 loaded from {filepath}")
