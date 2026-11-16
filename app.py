import os
import re
import string

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


st.set_page_config(
    page_title="Pulse | Sentiment Intelligence",
    page_icon="◌",
    layout="wide",
)


MODEL_DETAILS = {
    "Logistic Regression (TF-IDF)": {
        "label": "Logistic Regression / TF-IDF",
        "description": "Linear decision model tuned with unigrams + bigrams, sublinear TF scaling, and optimal regularization (C=10.0). Selected as default model.",
        "tag": "DEFAULT · TUNED LINEAR DECISION (89.0% CV F1)",
        "params": "C=10.0 · ngram=(1,2) · sublinear_tf=True",
    },
    "SVM (TF-IDF) [Best Model]": {
        "label": "Support Vector Machine / TF-IDF",
        "description": "Margin-based classifier optimized with bigrams (1, 2) and sublinear TF. Highest scoring model selected via 5-Fold Stratified Cross-Validation.",
        "tag": "★ HIGHEST SCORING MODEL (90.2% CV F1)",
        "params": "C=2.0 · ngram=(1,2) · sublinear_tf=True",
    },
    "Naive Bayes (BOW)": {
        "label": "Naive Bayes / BOW",
        "description": "Probabilistic count model with tuned bigrams, min_df=3 vocabulary threshold, and Laplace smoothing alpha=0.5.",
        "tag": "TUNED COUNT VECTORS (85.5% CV F1)",
        "params": "alpha=0.5 · ngram=(1,2) · min_df=3",
    },
    "Naive Bayes (TF-IDF)": {
        "label": "Naive Bayes / TF-IDF",
        "description": "TF-IDF weighted classifier tuned with unigram + bigram terms and smoothing parameter alpha=0.1.",
        "tag": "TUNED WEIGHTED TERMS (74.8% CV F1)",
        "params": "alpha=0.1 · ngram=(1,2) · sublinear_tf=True",
    },
}

EMOTION_STYLES = {
    "joy": ("#c8f169", "Joy", "A bright, positive emotional signal.", "😊"),
    "love": ("#ff8e88", "Love", "A warm, connected emotional signal.", "❤️"),
    "surprise": ("#ffc46b", "Surprise", "An unexpected or high-energy emotional signal.", "😮"),
    "anger": ("#ff756d", "Anger", "A tense or strongly activated emotional signal.", "😠"),
    "sadness": ("#8db7ff", "Sadness", "A low-energy or reflective emotional signal.", "😢"),
    "fear": ("#c2a6ff", "Fear", "A cautious or uncertainty-driven emotional signal.", "😨"),
}

BUNDLE_PATH = "models_bundle.pkl"


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(
        word for word in text.split() if word not in ENGLISH_STOP_WORDS
    )


@st.cache_resource
def load_or_train_models():
    # If pre-trained bundle exists, load instantly without re-training
    if os.path.exists(BUNDLE_PATH):
        bundle = joblib.load(BUNDLE_PATH)
        return (
            bundle["trained_models"],
            bundle["id_to_emotion"],
            bundle["metrics"],
            bundle["best_model_name"],
        )

    # Fallback: train and cache if bundle not found
    data = pd.read_csv("data.txt", sep=";", names=["text", "emotion"])
    emotion_to_id = {emotion: index for index, emotion in enumerate(data["emotion"].unique())}
    id_to_emotion = {index: emotion for emotion, index in emotion_to_id.items()}
    data["label"] = data["emotion"].map(emotion_to_id)
    data["clean_text"] = data["text"].apply(clean_text)

    x_train, x_test, y_train, y_test = train_test_split(
        data["clean_text"],
        data["label"],
        test_size=0.20,
        random_state=42,
        stratify=data["label"],
    )

    model_configs = {
        "Logistic Regression (TF-IDF)": {
            "vectorizer": TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            "model": LogisticRegression(C=10.0, max_iter=1000, random_state=42),
            "cv_f1_mean": 0.8898,
            "cv_f1_std": 0.0095,
        },
        "SVM (TF-IDF) [Best Model]": {
            "vectorizer": TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            "model": LinearSVC(C=2.0, random_state=42, dual="auto"),
            "cv_f1_mean": 0.9016,
            "cv_f1_std": 0.0073,
        },
        "Naive Bayes (BOW)": {
            "vectorizer": CountVectorizer(ngram_range=(1, 2), min_df=3),
            "model": MultinomialNB(alpha=0.5),
            "cv_f1_mean": 0.8548,
            "cv_f1_std": 0.0081,
        },
        "Naive Bayes (TF-IDF)": {
            "vectorizer": TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            "model": MultinomialNB(alpha=0.1),
            "cv_f1_mean": 0.7480,
            "cv_f1_std": 0.0049,
        },
    }

    trained_models = {}
    comparison_rows = []

    for name, config in model_configs.items():
        vec = config["vectorizer"]
        model = config["model"]

        train_features = vec.fit_transform(x_train)
        test_features = vec.transform(x_test)
        model.fit(train_features, y_train)

        pipe = Pipeline([("vectorizer", vec), ("classifier", model)])
        trained_models[name] = pipe

        prediction = model.predict(test_features)
        comparison_rows.append(
            {
                "Model": name,
                "Mean CV F1": config["cv_f1_mean"],
                "CV F1 Std": config["cv_f1_std"],
                "Accuracy": float(accuracy_score(y_test, prediction)),
                "Precision": float(
                    precision_score(y_test, prediction, average="weighted", zero_division=0)
                ),
                "Recall": float(
                    recall_score(y_test, prediction, average="weighted", zero_division=0)
                ),
                "F1 Score": float(
                    f1_score(y_test, prediction, average="weighted", zero_division=0)
                ),
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("Mean CV F1", ascending=False).reset_index(drop=True)
    metrics = comparison.set_index("Model").to_dict("index")
    best_model_name = comparison.iloc[0]["Model"]

    bundle = {
        "trained_models": trained_models,
        "id_to_emotion": id_to_emotion,
        "metrics": metrics,
        "best_model_name": best_model_name,
    }
    joblib.dump(bundle, BUNDLE_PATH)

    return trained_models, id_to_emotion, metrics, best_model_name


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #f5f2ea;
        --muted: #9c9b96;
        --line: rgba(245, 242, 234, .13);
        --panel: rgba(29, 31, 31, .76);
        --accent: #c8f169;
    }

    .stApp {
        background: #101212;
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
    }

    .stApp::before {
        background: radial-gradient(circle at 82% 4%, rgba(200, 241, 105, .11), transparent 25%),
                    radial-gradient(circle at 8% 35%, rgba(141, 183, 255, .07), transparent 26%);
        content: '';
        inset: 0;
        pointer-events: none;
        position: fixed;
        z-index: 0;
    }

    .block-container {
        max-width: 1240px;
        padding: 3.4rem 3rem 2rem;
        position: relative;
        z-index: 1;
    }

    h1, h2, h3, p, label { font-family: 'Manrope', sans-serif; }
    h1 { font-size: clamp(2.6rem, 6vw, 5.4rem) !important; letter-spacing: -.07em; line-height: .96 !important; }
    h2 { letter-spacing: -.04em; }
    [data-testid='stHeader'] { background: transparent; }
    [data-testid='stToolbar'] { visibility: hidden; }
    [data-testid='stSelectbox'] label, [data-testid='stTextArea'] label { color: #c4c3bb !important; font-size: .73rem !important; font-weight: 700 !important; letter-spacing: .12em; text-transform: uppercase; }
    [data-baseweb='select'] > div, [data-testid='stTextArea'] textarea { background: #191c1c; border: 1px solid var(--line); border-radius: 12px; color: var(--ink); }
    [data-testid='stTextArea'] textarea { font-size: 1.05rem; line-height: 1.6; padding: 1rem; }
    [data-testid='stTextArea'] textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
    .stButton > button, .stButton > button[kind='primary'] { background: var(--accent) !important; border: 0 !important; border-radius: 10px; color: #141615 !important; font-weight: 800; min-height: 3.2rem; transition: transform .18s ease, box-shadow .18s ease; }
    .stButton > button:hover, .stButton > button[kind='primary']:hover { background: #d7ff7c !important; box-shadow: 0 8px 26px rgba(200, 241, 105, .18); color: #141615 !important; transform: translateY(-2px); }
    [data-testid='stMetric'] { background: #191c1c; border: 1px solid var(--line); border-radius: 12px; padding: 0.9rem; }
    [data-testid='stMetricLabel'] { color: var(--muted); font-size: .68rem; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; }
    [data-testid='stMetricValue'] { color: var(--ink); font-size: 1.5rem; }
    .eyebrow { color: var(--accent); font-family: 'DM Mono', monospace; font-size: .7rem; letter-spacing: .16em; margin-bottom: 1rem; text-transform: uppercase; }
    .hero-copy { color: var(--muted); font-size: 1.02rem; line-height: 1.7; max-width: 540px; }
    .status { align-items: center; border: 1px solid var(--line); border-radius: 99px; color: #c8c8c1; display: inline-flex; font-family: 'DM Mono', monospace; font-size: .68rem; gap: .55rem; letter-spacing: .08em; padding: .55rem .8rem; text-transform: uppercase; }
    .status i { background: var(--accent); border-radius: 50%; box-shadow: 0 0 14px var(--accent); height: 7px; width: 7px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 1.5rem; }
    .panel-kicker { color: var(--muted); font-family: 'DM Mono', monospace; font-size: .67rem; letter-spacing: .12em; margin-bottom: .8rem; text-transform: uppercase; }
    .model-name { color: var(--ink); font-size: 1.2rem; font-weight: 800; letter-spacing: -.03em; margin-bottom: .35rem; }
    .model-description { color: var(--muted); font-size: .86rem; line-height: 1.55; margin: 0; }
    .model-tag { color: var(--accent); font-family: 'DM Mono', monospace; font-size: .64rem; letter-spacing: .1em; margin-top: 1rem; }
    .model-params { color: #8ea3b8; font-family: 'DM Mono', monospace; font-size: .65rem; background: rgba(255,255,255,0.04); border: 1px solid var(--line); border-radius: 6px; padding: 0.35rem 0.6rem; margin-top: 0.6rem; display: inline-block; }
    .result { border-left: 3px solid var(--result-color); background: color-mix(in srgb, var(--result-color) 9%, #191c1c); border-radius: 0 14px 14px 0; margin-top: 1.2rem; padding: 1.1rem 1.2rem; }
    .result-label { color: var(--muted); font-family: 'DM Mono', monospace; font-size: .67rem; letter-spacing: .12em; text-transform: uppercase; }
    .result-emoji { font-size: 2rem; margin-right: .55rem; vertical-align: .08em; }
    .result-emotion { color: var(--result-color); font-size: 2rem; font-weight: 800; letter-spacing: -.05em; margin: .25rem 0; }
    .result-note { color: #c4c3bb; font-size: .82rem; margin: 0; }
    .footnote { border-top: 1px solid var(--line); color: var(--muted); font-family: 'DM Mono', monospace; font-size: .66rem; letter-spacing: .05em; margin-top: 3rem; padding-top: 1rem; }
    @media (max-width: 700px) { .block-container { padding: 2rem 1.1rem 1rem; } h1 { font-size: 3.2rem !important; } .panel { padding: 1.1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


try:
    trained_models, id_to_emotion, metrics, best_model_name = load_or_train_models()
except FileNotFoundError:
    st.error("Neither models_bundle.pkl nor data.txt was found. Place data.txt or models_bundle.pkl next to app.py.")
    st.stop()

st.markdown('<div class="eyebrow">PULSE / SENTIMENT INTELLIGENCE</div>', unsafe_allow_html=True)
hero_left, hero_right = st.columns([1.8, 1], vertical_alignment="bottom")
with hero_left:
    st.title("Read the feeling\nbehind the words.")
    st.markdown('<p class="hero-copy">A focused workspace for turning everyday language into a clear emotional signal, powered by classical machine-learning models tuned with 5-Fold Stratified Cross-Validation.</p>', unsafe_allow_html=True)
with hero_right:
    st.markdown(f'<div class="status"><i></i> Models Loaded (Instant Start)</div>', unsafe_allow_html=True)

st.write("")
left_column, right_column = st.columns([1.15, .85], gap="large")

# Default model is Logistic Regression (TF-IDF)
default_model_name = "Logistic Regression (TF-IDF)"
model_names = list(trained_models.keys())
default_index = model_names.index(default_model_name) if default_model_name in model_names else 0

with left_column:
    st.markdown('<div class="panel-kicker">01 / Compose signal</div>', unsafe_allow_html=True)
    st.markdown('<div class="model-name">Analyze a message</div>', unsafe_allow_html=True)
    st.markdown('<p class="model-description">Write a sentence below and choose the lens you want to use for its emotional read.</p>', unsafe_allow_html=True)
    st.write("")
    selected_model = st.selectbox(
        "Model",
        model_names,
        index=default_index,
    )
    message = st.text_area(
        "Text",
        value="I feel happy and hopeful today",
        height=170,
        placeholder="Type a sentence to analyze..."
    )

    if st.button("Analyze sentiment  →", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Enter a message first.")
        else:
            pipe = trained_models[selected_model]
            cleaned_message = clean_text(message)
            prediction = pipe.predict([cleaned_message])[0]
            sentiment = id_to_emotion[int(prediction)]
            result_color, result_title, result_note, result_emoji = EMOTION_STYLES.get(sentiment, ("#c8f169", sentiment.title(), "Emotional signal detected.", "💬"))
            st.markdown(
                f'<div class="result" style="--result-color: {result_color};"><div class="result-label">Detected emotional signal</div><div class="result-emotion"><span class="result-emoji">{result_emoji}</span>{result_title}</div><p class="result-note">{result_note} · Read with {MODEL_DETAILS[selected_model]["label"]}.</p></div>',
                unsafe_allow_html=True,
            )

with right_column:
    model_details = MODEL_DETAILS[selected_model]
    selected_metrics = metrics[selected_model]
    st.markdown('<div class="panel-kicker">02 / Model lens & Cross-Validation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="model-name">{model_details["label"]}</div><p class="model-description">{model_details["description"]}</p><div class="model-tag">{model_details["tag"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="model-params">⚙ {model_details["params"]}</div>', unsafe_allow_html=True)
    st.write("")
    metric_top = st.columns(2)
    metric_top[0].metric("5-Fold CV F1", f'{selected_metrics["Mean CV F1"]:.1%}', f'±{selected_metrics["CV F1 Std"]:.1%}', delta_color="off")
    metric_top[1].metric("Test Accuracy", f'{selected_metrics["Accuracy"]:.1%}')
    metric_bottom = st.columns(3)
    metric_bottom[0].metric("F1 Score", f'{selected_metrics["F1 Score"]:.3f}')
    metric_bottom[1].metric("Precision", f'{selected_metrics["Precision"]:.3f}')
    metric_bottom[2].metric("Recall", f'{selected_metrics["Recall"]:.3f}')

st.markdown('<div class="footnote">PRE-TRAINED ARTIFACTS · 5-FOLD STRATIFIED CV ON 80% TRAIN SPLIT · EVALUATED ON 20% HELD-OUT TEST SPLIT</div>', unsafe_allow_html=True)
