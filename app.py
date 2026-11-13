import re
import string

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


st.set_page_config(
    page_title="Pulse | Sentiment Intelligence",
    page_icon="◌",
    layout="wide",
)


MODEL_DETAILS = {
    "Naive Bayes (BOW)": {
        "label": "Naive Bayes / BOW",
        "description": "A lightweight count-based classifier built on word occurrence patterns.",
        "tag": "COUNT VECTORS",
    },
    "Naive Bayes (TF-IDF)": {
        "label": "Naive Bayes / TF-IDF",
        "description": "A probabilistic classifier that weighs informative words above common ones.",
        "tag": "WEIGHTED TERMS",
    },
    "Logistic Regression (TF-IDF)": {
        "label": "Logistic Regression / TF-IDF",
        "description": "A linear decision model tuned for stable, interpretable text classification.",
        "tag": "LINEAR DECISION",
    },
    "SVM (TF-IDF)": {
        "label": "Support Vector Machine / TF-IDF",
        "description": "A margin-based classifier that is the strongest performer on this dataset split.",
        "tag": "MARGIN CLASSIFIER",
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


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(
        word for word in text.split() if word not in ENGLISH_STOP_WORDS
    )


@st.cache_resource
def train_models():
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

    bow = CountVectorizer(ngram_range=(1, 1))
    tfidf = TfidfVectorizer()
    x_train_bow = bow.fit_transform(x_train)
    x_test_bow = bow.transform(x_test)
    x_train_tfidf = tfidf.fit_transform(x_train)
    x_test_tfidf = tfidf.transform(x_test)

    trained_models = {
        "Naive Bayes (BOW)": (MultinomialNB(), x_train_bow, x_test_bow),
        "Naive Bayes (TF-IDF)": (MultinomialNB(), x_train_tfidf, x_test_tfidf),
        "Logistic Regression (TF-IDF)": (
            LogisticRegression(max_iter=1000),
            x_train_tfidf,
            x_test_tfidf,
        ),
        "SVM (TF-IDF)": (LinearSVC(), x_train_tfidf, x_test_tfidf),
    }

    predictions = {}
    comparison_rows = []
    for name, (model, train_features, test_features) in trained_models.items():
        model.fit(train_features, y_train)
        prediction = model.predict(test_features)
        predictions[name] = prediction
        comparison_rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, prediction),
                "Precision": precision_score(
                    y_test, prediction, average="weighted", zero_division=0
                ),
                "Recall": recall_score(
                    y_test, prediction, average="weighted", zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test, prediction, average="weighted", zero_division=0
                ),
            }
        )

    comparison = (
        pd.DataFrame(comparison_rows)
        .sort_values("F1 Score", ascending=False)
        .reset_index(drop=True)
    )
    metrics = comparison.set_index("Model").to_dict("index")
    return trained_models, bow, tfidf, id_to_emotion, metrics


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
    [data-testid='stMetric'] { background: #191c1c; border: 1px solid var(--line); border-radius: 12px; padding: 1rem; }
    [data-testid='stMetricLabel'] { color: var(--muted); font-size: .7rem; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; }
    [data-testid='stMetricValue'] { color: var(--ink); font-size: 1.65rem; }
    .eyebrow { color: var(--accent); font-family: 'DM Mono', monospace; font-size: .7rem; letter-spacing: .16em; margin-bottom: 1rem; text-transform: uppercase; }
    .hero-copy { color: var(--muted); font-size: 1.02rem; line-height: 1.7; max-width: 540px; }
    .status { align-items: center; border: 1px solid var(--line); border-radius: 99px; color: #c8c8c1; display: inline-flex; font-family: 'DM Mono', monospace; font-size: .68rem; gap: .55rem; letter-spacing: .08em; padding: .55rem .8rem; text-transform: uppercase; }
    .status i { background: var(--accent); border-radius: 50%; box-shadow: 0 0 14px var(--accent); height: 7px; width: 7px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 1.5rem; }
    .panel-kicker { color: var(--muted); font-family: 'DM Mono', monospace; font-size: .67rem; letter-spacing: .12em; margin-bottom: .8rem; text-transform: uppercase; }
    .model-name { color: var(--ink); font-size: 1.2rem; font-weight: 800; letter-spacing: -.03em; margin-bottom: .35rem; }
    .model-description { color: var(--muted); font-size: .86rem; line-height: 1.55; margin: 0; }
    .model-tag { color: var(--accent); font-family: 'DM Mono', monospace; font-size: .64rem; letter-spacing: .1em; margin-top: 1.3rem; }
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
    trained_models, bow, tfidf, id_to_emotion, metrics = train_models()
except FileNotFoundError:
    st.error("data.txt was not found. Place it next to app.py and restart the app.")
    st.stop()

st.markdown('<div class="eyebrow">PULSE / SENTIMENT INTELLIGENCE</div>', unsafe_allow_html=True)
hero_left, hero_right = st.columns([1.8, 1], vertical_alignment="bottom")
with hero_left:
    st.title("Read the feeling\nbehind the words.")
    st.markdown('<p class="hero-copy">A focused workspace for turning everyday language into a clear emotional signal, powered by four classical machine-learning models.</p>', unsafe_allow_html=True)
with hero_right:
    st.markdown('<div class="status"><i></i> Model suite ready</div>', unsafe_allow_html=True)

st.write("")
left_column, right_column = st.columns([1.15, .85], gap="large")

with left_column:
    st.markdown('<div class="panel-kicker">01 / Compose signal</div>', unsafe_allow_html=True)
    st.markdown('<div class="model-name">Analyze a message</div>', unsafe_allow_html=True)
    st.markdown('<p class="model-description">Write a sentence below and choose the lens you want to use for its emotional read.</p>', unsafe_allow_html=True)
    st.write("")
    selected_model = st.selectbox(
        "Model",
        list(trained_models),
        index=list(trained_models).index("Logistic Regression (TF-IDF)"),
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
            model, _, _ = trained_models[selected_model]
            cleaned_message = clean_text(message)
            features = tfidf.transform([cleaned_message])
            if selected_model == "Naive Bayes (BOW)":
                features = bow.transform([cleaned_message])
            prediction = model.predict(features)[0]
            sentiment = id_to_emotion[int(prediction)]
            result_color, result_title, result_note, result_emoji = EMOTION_STYLES.get(sentiment, ("#c8f169", sentiment.title(), "Emotional signal detected.", "💬"))
            st.markdown(
                f'<div class="result" style="--result-color: {result_color};"><div class="result-label">Detected emotional signal</div><div class="result-emotion"><span class="result-emoji">{result_emoji}</span>{result_title}</div><p class="result-note">{result_note} · Read with {MODEL_DETAILS[selected_model]["label"]}.</p></div>',
                unsafe_allow_html=True,
            )

with right_column:
    model_details = MODEL_DETAILS[selected_model]
    selected_metrics = metrics[selected_model]
    st.markdown('<div class="panel-kicker">02 / Model lens</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="model-name">{model_details["label"]}</div><p class="model-description">{model_details["description"]}</p><div class="model-tag">{model_details["tag"]}</div>', unsafe_allow_html=True)
    st.write("")
    metric_top = st.columns(2)
    metric_top[0].metric("Accuracy", f'{selected_metrics["Accuracy"]:.1%}')
    metric_top[1].metric("F1 score", f'{selected_metrics["F1 Score"]:.3f}')
    metric_bottom = st.columns(2)
    metric_bottom[0].metric("Precision", f'{selected_metrics["Precision"]:.3f}')
    metric_bottom[1].metric("Recall", f'{selected_metrics["Recall"]:.3f}')

st.markdown('<div class="footnote">TRAINED FROM DATA.TXT · 80/20 STRATIFIED SPLIT · METRICS FROM HELD-OUT TEST DATA</div>', unsafe_allow_html=True)
