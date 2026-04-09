# Model Diagnosis Report: Question Quality Evaluator

## 1. Root Cause Analysis

### The Core Problem: Feature Space Overlap, Not Class Imbalance

Your classes are balanced (3000 each), yet the model predicts Medium for ~50% of Low and ~22% of High samples. This is NOT a class imbalance problem — it's a **feature space overlap** problem. Here's why:

**The confusion matrix tells the story:**
```
                Predicted
              Low    Med    High
Actual Low   1347   1488    165
Actual Med     36   2910     54
Actual High   688    663   1649
```

- 1488 Low questions are predicted as Medium (49.6% of Low)
- 663 High questions are predicted as Medium (22.1% of High)
- But Medium itself has near-perfect recall (97%)

This means Medium Quality sits in the CENTER of the feature space, and the decision boundaries are pulled toward it.

### Why This Happens — Specific to Your Pipeline

**Cause 1: TF-IDF collapses semantic differences between Low and Medium**

LQ_CLOSE (Low) and LQ_EDIT (Medium) are both low-scoring Stack Overflow questions. The difference between them is NOT about the words used — it's about how the community responded. A question that says "how do I sort a list?" could be Low OR Medium depending on whether the community edited it or closed it. TF-IDF, which only looks at word frequencies, cannot distinguish these.

**Cause 2: Preprocessing destroys quality signals**

Your `remove_special_characters()` strips ALL non-letter characters. This removes:
- Code formatting (`{}`, `()`, `=`, `;`) — a strong quality indicator
- Punctuation patterns (multiple `???` = low quality)
- Markdown structure (`##`, `*`, `-`)
- Numbers (error codes, version numbers = higher quality)

After preprocessing, "help plz my code doesnt work" and "How do I implement sort? I tried list.sort() but got TypeError" look much more similar than they should.

**Cause 3: Handcrafted features are extracted AFTER cleaning**

Your `extract_text_features()` calculates word_count, char_count, etc. on `cleaned_text` — after stopwords and special characters are removed. This means features like `question_mark_count` are computed on text that already had punctuation stripped. You should extract structural features from the ORIGINAL text BEFORE cleaning.

**Cause 4: Logistic Regression draws linear boundaries**

The decision boundary between these overlapping classes is likely non-linear. Logistic Regression can only draw straight lines in feature space. When Medium sits between Low and High, a linear model will struggle to carve out distinct regions for all three classes.

**Cause 5: `class_weight='balanced'` is counterproductive here**

Your data is ALREADY balanced (15K per class). Setting `class_weight='balanced'` reweights samples unnecessarily and can distort the learned boundaries. Remove it.

### Why ROC AUC (82.45%) >> Accuracy (65.46%)

ROC AUC measures ranking ability — "can the model rank High questions higher than Low ones?" Yes, it can. The probability scores are reasonably well-ordered.

But accuracy measures hard predictions at the default 0.5 threshold. The model's probability distribution for Medium is likely very broad, so at default thresholds, too many samples cross into the Medium prediction zone.

This gap confirms: **the model has learned useful signal but the decision boundaries are wrong.**

---

## 2. Immediate Fixes (High Impact, Low Effort)

### Fix A: Extract features from ORIGINAL text, not cleaned text
**Expected improvement: +3-5% accuracy**

### Fix B: Remove `class_weight='balanced'` since data IS balanced
**Expected improvement: +1-2% accuracy**

### Fix C: Add structural features that TF-IDF can't capture
**Expected improvement: +3-5% accuracy**

New features to add:
- `code_block_count` — from original Body (number of `<code>` tags)
- `paragraph_count` — number of `<p>` tags
- `has_error_message` — regex for stack traces / error patterns
- `tag_count` — number of topic tags
- `title_word_count` — separate from body length
- `body_word_count` — separate from title length
- `exclamation_count` — from original text
- `caps_word_ratio` — ratio of ALL-CAPS words (shouting = low quality)
- `link_count` — number of URLs (more links = potentially higher quality)
- `avg_sentence_length` — longer sentences correlate with quality

### Fix D: Switch to LinearSVC or SGDClassifier
**Expected improvement: +2-4% accuracy**

LinearSVC with hinge loss often outperforms Logistic Regression on high-dimensional sparse data (which is what TF-IDF produces).

### Fix E: Threshold tuning via probability calibration
**Expected improvement: +2-3% accuracy**

Instead of argmax on raw probabilities, tune per-class thresholds to balance precision/recall.

---

## 3. Advanced Improvements

### Upgrade A: Gradient Boosted Trees (XGBoost/LightGBM)
**Expected improvement: +5-10% accuracy over baseline**

Tree models handle non-linear boundaries and feature interactions natively. They can learn that "short text + no code + multiple question marks" = Low Quality without you explicitly engineering interaction features.

### Upgrade B: Stacking Ensemble
Combine LinearSVC (good on TF-IDF) + XGBoost (good on handcrafted features) via a meta-learner.

### Upgrade C: Treat as Ordinal Classification
Low < Medium < High is ordinal. Standard multiclass treats them as unrelated categories. An ordinal approach (e.g., two binary classifiers: "Low vs not-Low" and "High vs not-High") can reduce Medium-class domination.

### Upgrade D: Sentence-BERT Embeddings
If you want to go beyond TF-IDF, sentence-transformers produce 384/768-dim dense vectors that capture semantic meaning. But this requires GPU or significant CPU time and adds complexity. For a project of this scope, TF-IDF + strong handcrafted features + a good model is likely sufficient.

---

## 4. Metrics to Focus On

- **Primary:** Macro F1 (treats all classes equally — better than weighted F1 for balanced data)
- **Secondary:** Per-class recall (ensure no class is neglected)
- **Diagnostic:** Confusion matrix (watch for Medium domination)
- **Ranking:** ROC AUC (should stay above 80%)

Avoid focusing on accuracy alone — macro F1 is more informative for multi-class problems.

---

## 5. Priority Action Plan

| Priority | Action | Effort | Expected Gain |
|----------|--------|--------|---------------|
| 1 | Fix feature extraction (use original text) | Low | +3-5% |
| 2 | Add 10+ structural features | Low | +3-5% |
| 3 | Remove class_weight='balanced' | Trivial | +1-2% |
| 4 | Switch to LinearSVC | Low | +2-4% |
| 5 | Add XGBoost comparison | Medium | +5-10% |
| 6 | Threshold tuning | Medium | +2-3% |
| 7 | Stacking ensemble | High | +3-5% |

Realistic target: **75-82% accuracy** with fixes 1-6.
