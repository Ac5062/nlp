"""
Main Training Script
====================
Orchestrates the full pipeline: data loading, preprocessing,
feature engineering, model training, and evaluation.

Usage:
  python train.py                          # Use sample dataset
  python train.py --data data/train.csv    # Use Kaggle dataset
  python train.py --tune                   # With hyperparameter tuning
"""

import argparse
import os
import sys
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import TextPreprocessor, load_dataset, encode_labels
from src.feature_engineering import FeatureEngineer
from src.model import QuestionQualityModel
from src.evaluate import ModelEvaluator


def main(args):
    """Run the full training pipeline."""
    start_time = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  Question Quality Evaluator - Training Pipeline")
    print("=" * 60)
    print()

    # ── Step 1: Load Data ──────────────────────────────────
    data_path = os.path.join(base_dir, args.data)
    if not os.path.exists(data_path):
        print(f"Dataset not found at: {data_path}")
        print("Generating sample dataset...")
        from generate_sample_data import generate_dataset
        generate_dataset(num_per_class=500)
        data_path = os.path.join(base_dir, "data", "sample_dataset.csv")

    df = load_dataset(data_path)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # ── Step 2: Encode Labels ──────────────────────────────
    print("\n" + "─" * 40)
    print("Step 2: Encoding labels")
    df = encode_labels(df, label_column='Y')

    # ── Step 3: Preprocess Text ────────────────────────────
    print("\n" + "─" * 40)
    print("Step 3: Preprocessing text")
    preprocessor = TextPreprocessor()

    # Determine text columns
    title_col = 'Title' if 'Title' in df.columns else None
    body_col = 'Body' if 'Body' in df.columns else None

    if body_col:
        df = preprocessor.preprocess_dataframe(df, text_column=body_col,
                                                title_column=title_col or 'Title')
    elif title_col:
        df = preprocessor.preprocess_dataframe(df, text_column='Title',
                                                title_column='Title')
    else:
        print("ERROR: No 'Title' or 'Body' column found in dataset.")
        sys.exit(1)

    # ── Step 4: Feature Engineering ────────────────────────
    print("\n" + "─" * 40)
    print("Step 4: Feature engineering")
    fe = FeatureEngineer(max_features=args.max_features, ngram_range=(1, 2))

    # Extract handcrafted features
    df = fe.extract_text_features(df, text_column='cleaned_text')

    # Split data BEFORE fitting TF-IDF (to avoid data leakage)
    X_text = df['cleaned_text']
    y = df['label']

    X_text_train, X_text_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_text, y, df.index,
        test_size=args.test_size,
        random_state=42,
        stratify=y
    )

    print(f"\nTrain set: {len(X_text_train)} samples")
    print(f"Test set:  {len(X_text_test)} samples")

    # Fit TF-IDF on training data only
    tfidf_train = fe.fit_transform_tfidf(X_text_train)
    tfidf_test = fe.transform_tfidf(X_text_test)

    # Combine with handcrafted features
    X_train = fe.get_combined_features(df.loc[idx_train], tfidf_train)
    X_test = fe.get_combined_features(df.loc[idx_test], tfidf_test)

    # ── Step 5: Model Training ─────────────────────────────
    print("\n" + "─" * 40)
    print("Step 5: Model training")
    model = QuestionQualityModel()

    if args.tune:
        model.train_with_tuning(X_train, y_train, cv=5)
    else:
        model.train(X_train, y_train)

    # ── Step 6: Evaluation ─────────────────────────────────
    print("\n" + "─" * 40)
    print("Step 6: Evaluation")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    evaluator = ModelEvaluator()
    results = evaluator.evaluate(y_test, y_pred, y_proba)
    evaluator.print_report()

    # Cross-validation on full training set
    print("\n" + "─" * 40)
    print("Cross-Validation Results:")
    cv_results = model.cross_validate(X_train, y_train, cv=5)

    # ── Step 7: Save Outputs ───────────────────────────────
    print("\n" + "─" * 40)
    print("Step 7: Saving outputs")

    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # Save model and feature engineer
    model.save(os.path.join(models_dir, "question_quality_model.pkl"))
    fe.save(os.path.join(models_dir, "feature_engineer.pkl"))

    # Save evaluation plots
    evaluator.plot_confusion_matrix(
        save_path=os.path.join(outputs_dir, "confusion_matrix.png")
    )
    evaluator.plot_metrics_comparison(
        save_path=os.path.join(outputs_dir, "metrics_comparison.png")
    )

    # Save text report
    evaluator.save_report(os.path.join(outputs_dir, "evaluation_report.txt"))

    # ── Summary ────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n  Total time:    {elapsed:.1f} seconds")
    print(f"  Accuracy:      {results['accuracy']:.4f}")
    print(f"  F1 (weighted): {results['f1_weighted']:.4f}")
    print(f"\n  Model saved:   models/question_quality_model.pkl")
    print(f"  Features saved: models/feature_engineer.pkl")
    print(f"  Report saved:  outputs/evaluation_report.txt")
    print(f"  Plots saved:   outputs/confusion_matrix.png")
    print(f"                 outputs/metrics_comparison.png")
    print(f"\n  Run the app:   streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Question Quality Evaluator")
    parser.add_argument(
        "--data", type=str, default="data/sample_dataset.csv",
        help="Path to the dataset CSV file (relative to project root)"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Fraction of data for testing (default: 0.2)"
    )
    parser.add_argument(
        "--max_features", type=int, default=10000,
        help="Max TF-IDF features (default: 10000)"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Enable hyperparameter tuning with GridSearchCV"
    )
    args = parser.parse_args()
    main(args)
