"""
Model Training Module
=====================
Handles model training, hyperparameter tuning, and prediction
for the Question Quality Evaluator.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
import joblib
import os
import time


class QuestionQualityModel:
    """Logistic Regression model for question quality classification."""

    def __init__(self, C=1.0, max_iter=1000, solver='lbfgs'):
        """
        Initialize the model.

        Args:
            C: Regularization strength (inverse)
            max_iter: Maximum iterations for solver convergence
            solver: Optimization algorithm
        """
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        self.best_params = None
        self.label_names = {
            0: 'Low Quality',
            1: 'Medium Quality',
            2: 'High Quality'
        }

    def train(self, X_train, y_train):
        """
        Train the model on training data.

        Args:
            X_train: Feature matrix (sparse or dense)
            y_train: Label array
        """
        print("Training Logistic Regression model...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        self.is_trained = True
        print(f"Training complete in {elapsed:.2f} seconds")
        print(f"Training accuracy: {self.model.score(X_train, y_train):.4f}")

    def train_with_tuning(self, X_train, y_train, cv=5):
        """
        Train with hyperparameter tuning using GridSearchCV.

        Args:
            X_train: Feature matrix
            y_train: Label array
            cv: Number of cross-validation folds

        Returns:
            dict: Best parameters found
        """
        print("Starting hyperparameter tuning with GridSearchCV...")
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'saga'],
        }

        grid_search = GridSearchCV(
            LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        elapsed = time.time() - start_time

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.is_trained = True

        print(f"\nTuning complete in {elapsed:.2f} seconds")
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

        return self.best_params

    def predict(self, X):
        """
        Predict quality labels.

        Args:
            X: Feature matrix

        Returns:
            numpy array of predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained. Call train() first.")
        return self.model.predict_proba(X)

    def predict_single(self, X):
        """
        Predict quality for a single question with probabilities.

        Args:
            X: Feature vector for a single question

        Returns:
            dict with predicted label and probabilities
        """
        prediction = self.predict(X)[0]
        probabilities = self.predict_proba(X)[0]

        result = {
            'predicted_label': int(prediction),
            'predicted_quality': self.label_names[int(prediction)],
            'probabilities': {
                self.label_names[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
        }
        return result

    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.

        Args:
            X: Feature matrix
            y: Label array
            cv: Number of folds

        Returns:
            dict with mean and std for each metric
        """
        print(f"Running {cv}-fold cross-validation...")
        metrics = {}
        for scoring in ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            metrics[scoring] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'scores': scores.tolist()
            }
            print(f"  {scoring}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

        return metrics

    def save(self, filepath):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'is_trained': self.is_trained,
            'best_params': self.best_params,
            'label_names': self.label_names
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load trained model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.is_trained = data['is_trained']
        self.best_params = data['best_params']
        self.label_names = data['label_names']
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Quick test with random data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=20, n_classes=3,
                               n_informative=15, random_state=42)
    model = QuestionQualityModel()
    model.train(X, y)
    preds = model.predict(X[:5])
    print(f"Sample predictions: {preds}")
