"""
Streamlit Web Application
=========================
Real-time Question Quality Evaluator with an interactive web interface.

Usage:
  streamlit run app.py

Prerequisites:
  - Run train.py first to generate the model files
"""

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import TextPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model import QuestionQualityModel


# ── Page Configuration ──────────────────────────────────
st.set_page_config(
    page_title="Question Quality Evaluator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .quality-high {
        background-color: #d1fae5;
        border-left: 5px solid #10b981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .quality-medium {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .quality-low {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f9fafb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model and feature engineer."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "question_quality_model.pkl")
    fe_path = os.path.join(base_dir, "models", "feature_engineer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(fe_path):
        return None, None, None

    model = QuestionQualityModel()
    model.load(model_path)

    fe = FeatureEngineer()
    fe.load(fe_path)

    preprocessor = TextPreprocessor()

    return model, fe, preprocessor


def get_quality_display(quality, confidence):
    """Get the display elements for a quality prediction."""
    if quality == "High Quality":
        emoji = "🟢"
        css_class = "quality-high"
        color = "#10b981"
        tips = "Great question! It is clear, specific, and provides enough context."
    elif quality == "Medium Quality":
        emoji = "🟡"
        css_class = "quality-medium"
        color = "#f59e0b"
        tips = ("This question could be improved. Consider adding more context, "
                "code examples, or specifying what you have already tried.")
    else:
        emoji = "🔴"
        css_class = "quality-low"
        color = "#ef4444"
        tips = ("This question needs significant improvement. Add a clear title, "
                "describe your problem in detail, include code and error messages, "
                "and explain what you have tried so far.")

    return emoji, css_class, color, tips


def predict_question(question_text, model, fe, preprocessor):
    """Run the prediction pipeline on a single question."""
    # Preprocess
    cleaned = preprocessor.preprocess_text(question_text)

    if not cleaned.strip():
        return None

    # Create a mini DataFrame for feature extraction
    mini_df = pd.DataFrame([{
        'Title': question_text[:100],
        'Body': question_text,
        'combined_text': question_text,
        'cleaned_text': cleaned
    }])

    # Extract handcrafted features
    mini_df = fe.extract_text_features(mini_df, text_column='cleaned_text')

    # TF-IDF transform
    tfidf_features = fe.transform_tfidf([cleaned])

    # Combine features
    combined = fe.get_combined_features(mini_df, tfidf_features)

    # Predict
    result = model.predict_single(combined)
    return result


# ── Main App ────────────────────────────────────────────
def main():
    # Header
    st.markdown('<p class="main-header">📝 Question Quality Evaluator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluate the quality of your questions using NLP</p>',
                unsafe_allow_html=True)

    # Load model
    model, fe, preprocessor = load_model()

    if model is None:
        st.error("⚠️ Model not found! Please run `python train.py` first to train the model.")
        st.code("python train.py", language="bash")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This tool evaluates questions based on:
        - **Clarity** - Is the question clear?
        - **Specificity** - Is it detailed enough?
        - **Answerability** - Can it be meaningfully answered?

        **Quality Levels:**
        - 🟢 **High** - Well-structured, detailed
        - 🟡 **Medium** - Decent but could improve
        - 🔴 **Low** - Needs major improvement
        """)

        st.divider()
        st.header("📊 Model Info")
        st.markdown("""
        - **Model:** Logistic Regression
        - **Features:** TF-IDF + Handcrafted
        - **Dataset:** Stack Overflow Questions
        """)

        st.divider()
        st.header("💡 Tips for Good Questions")
        st.markdown("""
        1. Write a clear, descriptive title
        2. Describe the problem in detail
        3. Include relevant code snippets
        4. Show what you have tried
        5. Include error messages
        6. Use proper formatting
        """)

    # Main input area
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Enter Your Question")
        question_text = st.text_area(
            "Type or paste a question below:",
            height=200,
            placeholder="Example: How do I implement a binary search tree in Python? "
                        "I need support for insert, delete, and search operations..."
        )

        evaluate_button = st.button("🔍 Evaluate Quality", type="primary", use_container_width=True)

    with col2:
        st.subheader("Quick Examples")
        example_questions = {
            "🟢 High Quality": "How do I implement a binary search tree in Python with O(log n) "
                               "lookup? I need the tree to support insertion, deletion, and "
                               "search operations. Here is my current Node class implementation.",
            "🟡 Medium Quality": "Sort a list in Python. I tried sort() but it doesn't work "
                                 "for my case with mixed types.",
            "🔴 Low Quality": "help plz my code doesnt work"
        }

        for label, example in example_questions.items():
            if st.button(f"Try: {label}", use_container_width=True):
                question_text = example
                evaluate_button = True

    # Evaluation
    if evaluate_button and question_text.strip():
        with st.spinner("Analyzing question quality..."):
            result = predict_question(question_text, model, fe, preprocessor)

        if result is None:
            st.warning("Could not process the question. Please enter a meaningful question.")
        else:
            quality = result['predicted_quality']
            probs = result['probabilities']
            emoji, css_class, color, tips = get_quality_display(
                quality, max(probs.values())
            )

            st.divider()

            # Result display
            st.markdown(f"""
            <div class="{css_class}">
                <h2 style="margin:0;">{emoji} {quality}</h2>
                <p style="margin-top:0.5rem; color:#374151;">{tips}</p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence scores
            st.subheader("Confidence Scores")
            prob_cols = st.columns(3)

            colors_map = {"High Quality": "#10b981", "Medium Quality": "#f59e0b",
                          "Low Quality": "#ef4444"}

            for i, (label, prob) in enumerate(probs.items()):
                with prob_cols[i]:
                    st.metric(label=label, value=f"{prob:.1%}")
                    st.progress(prob)

            # Detailed analysis
            with st.expander("📋 Detailed Analysis"):
                cleaned = preprocessor.preprocess_text(question_text)
                words = cleaned.split()

                analysis_cols = st.columns(4)
                with analysis_cols[0]:
                    st.metric("Word Count", len(question_text.split()))
                with analysis_cols[1]:
                    st.metric("Character Count", len(question_text))
                with analysis_cols[2]:
                    st.metric("Question Marks", question_text.count('?'))
                with analysis_cols[3]:
                    has_code = 1 if ('```' in question_text or '<code>' in question_text) else 0
                    st.metric("Has Code", "Yes" if has_code else "No")

                st.markdown("**Preprocessed text:**")
                st.code(cleaned[:500] + ("..." if len(cleaned) > 500 else ""))

    elif evaluate_button:
        st.warning("Please enter a question to evaluate.")

    # Footer
    st.divider()
    st.markdown(
        "<p style='text-align:center; color:#9ca3af; font-size:0.9rem;'>"
        "Question Quality Evaluator | Built with NLP & Streamlit | "
        "Dataset: Stack Overflow Questions with Quality Rating (Kaggle)"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
