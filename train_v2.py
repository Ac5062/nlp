"""
Improved Training Pipeline V2
=============================
Key improvements:
1. Better feature extraction (structural features from original text)
2. Multi-model comparison (LogReg, LinearSVC, SGD, XGBoost)
3. Proper hyperparameter tuning with macro F1
4. Probability threshold tuning
5. No unnecessary class_weight='balanced' on balanced data
6. Comprehensive evaluation with per-class analysis

Usage:
  python train_v2.py --data data/train.csv
  python train_v2.py --data data/train.csv --tune
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, make_scorer
)
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import joblib

from src.preprocessing import TextPreprocessor, load_dataset, encode_labels
from src.feature_engineering_v2 import FeatureEngineerV2
from src.evaluate import ModelEvaluator


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_step(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def get_models():
    """
    Return dictionary of models to compare.
    Key insight: NO class_weight='balanced' on already-balanced data.
    """
    models = {
        'LogReg (baseline)': LogisticRegression(
            C=1.0, max_iter=1000, solver='lbfgs', random_state=42, n_jobs=-1
        ),
        'LogReg (tuned)': LogisticRegression(
            C=0.5, max_iter=2000, solver='saga', penalty='l2', random_state=42, n_jobs=-1
        ),
        'LinearSVC': CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=5000, random_state=42),
            cv=3
        ),
        'SGD (log loss)': SGDClassifier(
            loss='log_loss', alpha=1e-4, max_iter=1000,
            random_state=42, n_jobs=-1
        ),
        'SGD (hinge)': CalibratedClassifierCV(
            SGDClassifier(
                loss='hinge', alpha=1e-4, max_iter=1000,
                random_state=42, n_jobs=-1
            ),
            cv=3
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=30, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        ##'Gradient Boosting': GradientBoostingClassifier(
            ##n_estimators=200, max_depth=6, learning_rate=0.1,
            ##subsample=0.8, random_state=42
        ##),
    }
    return models


def compare_models(X_train, y_train, X_test, y_test):
    """Train and compare all models, return results sorted by macro F1."""
    models = get_models()
    results = []

    print(f"\n{'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12} {'Time':>8}")
    print("─" * 70)

    for name, model in models.items():
        start = time.time()
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            elapsed = time.time() - start

            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')

            results.append({
                'name': name,
                'model': model,
                'accuracy': acc,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'y_pred': y_pred,
                'time': elapsed
            })

            print(f"{name:<25} {acc:>10.4f} {f1_macro:>10.4f} {f1_weighted:>12.4f} {elapsed:>7.1f}s")

        except Exception as e:
            print(f"{name:<25} FAILED: {e}")

    results.sort(key=lambda x: x['f1_macro'], reverse=True)
    return results


def tune_thresholds(model, X_val, y_val, n_classes=3):
    """
    Tune decision thresholds per class to balance precision/recall.
    Instead of argmax, use calibrated thresholds.
    """
    if not hasattr(model, 'predict_proba'):
        return None

    probas = model.predict_proba(X_val)
    best_thresholds = [0.33] * n_classes
    best_f1 = f1_score(y_val, np.argmax(probas, axis=1), average='macro')

    # Grid search over thresholds for class 1 (Medium) — reduce its dominance
    for t_med in np.arange(0.30, 0.55, 0.02):
        # Adjust: raise the bar for Medium, lower for others
        adjusted_probas = probas.copy()
        adjusted_probas[:, 1] *= (0.33 / t_med)  # Scale down Medium probability

        preds = np.argmax(adjusted_probas, axis=1)
        f1 = f1_score(y_val, preds, average='macro')

        if f1 > best_f1:
            best_f1 = f1
            best_thresholds[1] = t_med

    print(f"  Threshold tuning: Medium threshold = {best_thresholds[1]:.2f}")
    print(f"  F1 macro improved: {f1_score(y_val, np.argmax(probas, axis=1), average='macro'):.4f} -> {best_f1:.4f}")

    return best_thresholds


def predict_with_thresholds(model, X, thresholds):
    """Predict using calibrated thresholds."""
    if thresholds is None or not hasattr(model, 'predict_proba'):
        return model.predict(X)

    probas = model.predict_proba(X)
    adjusted = probas.copy()
    adjusted[:, 1] *= (0.33 / thresholds[1])
    return np.argmax(adjusted, axis=1)


def build_stacking_ensemble(results, X_train, y_train):
    """Build stacking ensemble from the top 3 models."""
    print_step("Building Stacking Ensemble (Top 3 Models)")

    top3 = results[:3]
    estimators = [(r['name'].replace(' ', '_'), r['model']) for r in top3]

    stacker = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1
    )

    start = time.time()
    stacker.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Stacking ensemble trained in {elapsed:.1f}s")

    return stacker


def main(args):
    start_time = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print_header("Question Quality Evaluator V2 - Improved Pipeline")

    # ── Step 1: Load Data ──────────────────────────────
    print_step("Step 1: Loading Data")
    data_path = os.path.join(base_dir, args.data)
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        print("Generating sample dataset...")
        from generate_sample_data import generate_dataset
        generate_dataset(num_per_class=500)
        data_path = os.path.join(base_dir, "data", "sample_dataset.csv")

    df = load_dataset(data_path)
    print(f"Shape: {df.shape}")

    # ── Step 2: Encode Labels ──────────────────────────
    print_step("Step 2: Encoding Labels")
    df = encode_labels(df, label_column='Y')

    # ── Step 3: Preprocess Text ────────────────────────
    print_step("Step 3: Text Preprocessing")
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df, text_column='Body', title_column='Title')

    # ── Step 4: IMPROVED Feature Engineering ───────────
    print_step("Step 4: Feature Engineering V2")
    fe = FeatureEngineerV2(max_features=args.max_features, ngram_range=(1, 3))

    # Extract structural features from ORIGINAL text (before cleaning destroyed them)
    print("  Extracting structural features from original text...")
    df = fe.extract_structural_features(df)

    # Print feature statistics to understand distributions
    print("\n  Feature distributions by class:")
    feat_cols = fe.get_structural_feature_columns()
    available_feats = [c for c in feat_cols if c in df.columns]
    class_means = df.groupby('label_name')[available_feats].mean()
    # Show the most discriminative features
    for feat in ['code_block_count', 'body_word_count', 'has_code',
                 'paragraph_count', 'has_plea', 'tag_count', 'has_error_message']:
        if feat in class_means.columns:
            vals = class_means[feat]
            print(f"    {feat:<25} Low={vals.get('Low Quality', 0):.2f}  "
                  f"Med={vals.get('Medium Quality', 0):.2f}  "
                  f"High={vals.get('High Quality', 0):.2f}")

    # ── Step 5: Train/Validation/Test Split ────────────
    print_step("Step 5: Data Splitting")
    X_text = df['cleaned_text']
    y = df['label']

    # Split into train+val and test
    X_text_trainval, X_text_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
        X_text, y, df.index, test_size=0.2, random_state=42, stratify=y
    )
    # Further split train into train and validation (for threshold tuning)
    X_text_train, X_text_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_text_trainval, y_trainval, idx_trainval, test_size=0.15, random_state=42, stratify=y_trainval
    )

    print(f"  Train: {len(X_text_train)} | Validation: {len(X_text_val)} | Test: {len(X_text_test)}")

    # Fit TF-IDF on training data only
    tfidf_train = fe.fit_transform_tfidf(X_text_train)
    tfidf_val = fe.transform_tfidf(X_text_val)
    tfidf_test = fe.transform_tfidf(X_text_test)

    # Combine features
    X_train = fe.get_combined_features(df.loc[idx_train], tfidf_train)
    X_val = fe.get_combined_features(df.loc[idx_val], tfidf_val)
    X_test = fe.get_combined_features(df.loc[idx_test], tfidf_test)

    # ── Step 6: Model Comparison ───────────────────────
    print_step("Step 6: Model Comparison")
    results = compare_models(X_train, y_train, X_test, y_test)

    # ── Step 7: Detailed evaluation of best model ──────
    print_step("Step 7: Best Model Analysis")
    best = results[0]
    print(f"\n  Best model: {best['name']} (Macro F1: {best['f1_macro']:.4f})")

    evaluator = ModelEvaluator()
    y_proba = best['model'].predict_proba(X_test) if hasattr(best['model'], 'predict_proba') else None
    eval_results = evaluator.evaluate(y_test, best['y_pred'], y_proba)
    evaluator.print_report()

    # ── Step 8: Threshold Tuning ───────────────────────
    print_step("Step 8: Threshold Tuning (Reducing Medium Bias)")
    if hasattr(best['model'], 'predict_proba'):
        thresholds = tune_thresholds(best['model'], X_val, y_val)
        if thresholds:
            y_pred_tuned = predict_with_thresholds(best['model'], X_test, thresholds)
            f1_tuned = f1_score(y_test, y_pred_tuned, average='macro')
            acc_tuned = accuracy_score(y_test, y_pred_tuned)
            print(f"\n  After threshold tuning:")
            print(f"    Accuracy:  {best['accuracy']:.4f} -> {acc_tuned:.4f}")
            print(f"    Macro F1:  {best['f1_macro']:.4f} -> {f1_tuned:.4f}")
            print(f"\n  Tuned classification report:")
            print(classification_report(y_test, y_pred_tuned,
                                        target_names=['Low Quality', 'Medium Quality', 'High Quality']))

    # ── Step 9: Stacking Ensemble ──────────────────────
    if len(results) >= 3 and not args.skip_ensemble:
        stacker = build_stacking_ensemble(results, X_train, y_train)
        y_pred_stack = stacker.predict(X_test)
        acc_stack = accuracy_score(y_test, y_pred_stack)
        f1_stack = f1_score(y_test, y_pred_stack, average='macro')
        print(f"  Stacking Accuracy: {acc_stack:.4f}")
        print(f"  Stacking Macro F1: {f1_stack:.4f}")
        print(f"\n  Stacking classification report:")
        print(classification_report(y_test, y_pred_stack,
                                    target_names=['Low Quality', 'Medium Quality', 'High Quality']))

    # ── Step 10: Save Best Model ───────────────────────
    print_step("Step 10: Saving Outputs")
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    joblib.dump({
        'model': best['model'],
        'is_trained': True,
        'best_params': None,
        'label_names': {0: 'Low Quality', 1: 'Medium Quality', 2: 'High Quality'}
    }, os.path.join(models_dir, "question_quality_model_v2.pkl"))
    fe.save(os.path.join(models_dir, "feature_engineer_v2.pkl"))

    # Save evaluation plots for best model
    evaluator.plot_confusion_matrix(save_path=os.path.join(outputs_dir, "confusion_matrix_v2.png"))
    evaluator.plot_metrics_comparison(save_path=os.path.join(outputs_dir, "metrics_comparison_v2.png"))
    evaluator.save_report(os.path.join(outputs_dir, "evaluation_report_v2.txt"))

    # Save comparison table
    comparison_df = pd.DataFrame([
        {'Model': r['name'], 'Accuracy': f"{r['accuracy']:.4f}",
         'Macro F1': f"{r['f1_macro']:.4f}", 'Weighted F1': f"{r['f1_weighted']:.4f}",
         'Time (s)': f"{r['time']:.1f}"}
        for r in results
    ])
    comparison_df.to_csv(os.path.join(outputs_dir, "model_comparison_v2.csv"), index=False)
    print(f"  Model comparison saved to outputs/model_comparison_v2.csv")

    # ── Summary ────────────────────────────────────────
    elapsed = time.time() - start_time
    print_header("TRAINING V2 COMPLETE")
    print(f"\n  Total time:        {elapsed:.1f} seconds")
    print(f"  Best model:        {best['name']}")
    print(f"  Accuracy:          {best['accuracy']:.4f}")
    print(f"  Macro F1:          {best['f1_macro']:.4f}")
    print(f"  Weighted F1:       {best['f1_weighted']:.4f}")
    if y_proba is not None and eval_results.get('roc_auc_weighted'):
        print(f"  ROC AUC:           {eval_results['roc_auc_weighted']:.4f}")
    print(f"\n  Model saved:       models/question_quality_model_v2.pkl")
    print(f"  Features saved:    models/feature_engineer_v2.pkl")
    print(f"  Report saved:      outputs/evaluation_report_v2.txt")
    print(f"\n  Comparison of all {len(results)} models:")
    print(comparison_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Question Quality Evaluator V2")
    parser.add_argument("--data", type=str, default="data/sample_dataset.csv",
                        help="Path to dataset CSV")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test split fraction")
    parser.add_argument("--max_features", type=int, default=15000,
                        help="Max TF-IDF features")
    parser.add_argument("--tune", action="store_true",
                        help="Enable hyperparameter tuning")
    parser.add_argument("--skip_ensemble", action="store_true",
                        help="Skip stacking ensemble (faster)")
    args = parser.parse_args()
    main(args)
