import re
import string

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📱",
    layout="wide",
)

st.title("📱 SMS Spam Detection")
st.markdown(
    "### Naive Bayes + TF-IDF Text Classification\n"
    "Classify an SMS as **Spam** or **Legitimate (Ham)**."
)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def prepare_dataset(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    label_names = ["label", "v1", "category", "class", "target"]
    text_names = ["message", "v2", "text", "sms", "body"]

    label_col = next((c for c in label_names if c in df.columns), None)
    text_col = next((c for c in text_names if c in df.columns), None)

    if label_col is None or text_col is None:
        if len(df.columns) >= 2:
            label_col, text_col = df.columns[:2]
        else:
            raise ValueError(
                "CSV needs two columns: one label column and one message/text column."
            )

    data = df[[label_col, text_col]].copy()
    data.columns = ["label", "message"]
    data = data.dropna()

    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["label"] = data["label"].replace(
        {"legitimate": "ham", "legit": "ham", "not spam": "ham"}
    )
    data = data[data["label"].isin(["ham", "spam"])]

    data["clean_message"] = data["message"].astype(str).apply(clean_text)
    data = data[data["clean_message"].str.len() > 0]

    if data["label"].nunique() < 2:
        raise ValueError("Dataset must contain both 'ham' and 'spam' messages.")

    return data.reset_index(drop=True)


demo = pd.DataFrame(
    {
        "label": [
            "ham", "ham", "spam", "ham", "spam",
            "ham", "spam", "ham", "spam", "ham",
            "spam", "ham", "spam", "ham", "spam",
            "ham", "spam", "ham", "ham", "spam",
        ],
        "message": [
            "Hey, are we still meeting at 6 pm?",
            "Can you send me the notes from today's class?",
            "Congratulations! You won a free cash prize. Call now to claim.",
            "Please call me when you reach home.",
            "URGENT! You have won a free vacation. Reply WIN now.",
            "Happy birthday! Have a great day.",
            "You have been selected for a £1000 reward. Claim immediately.",
            "The assignment is due tomorrow morning.",
            "WINNER! Claim your exclusive free prize by calling this number.",
            "I'll be late for college today.",
            "Free entry in a weekly competition. Text WIN now.",
            "Can you pick up some milk on your way home?",
            "You have won a guaranteed cash bonus. Call now!",
            "Your appointment is confirmed for tomorrow at 10 AM.",
            "SPECIAL OFFER! Get a free gift voucher today.",
            "Don't forget to bring your ID card.",
            "You have received a bonus reward. Call now!",
            "Let's meet near the station after class.",
            "Thanks for helping me with the project.",
            "WIN a free prize now. Reply YES to claim.",
        ],
    }
)

st.sidebar.header("⚙️ Settings")
uploaded = st.sidebar.file_uploader("Upload SMS dataset (.csv)", type=["csv"])

if uploaded:
    try:
        data = prepare_dataset(pd.read_csv(uploaded))
        st.sidebar.success(f"Loaded {len(data)} SMS messages.")
    except Exception as exc:
        st.sidebar.error(str(exc))
        data = prepare_dataset(demo)
else:
    data = prepare_dataset(demo)
    st.sidebar.info("Demo dataset is being used.")

test_size = st.sidebar.slider("Test set size", 0.10, 0.40, 0.20, 0.05)
random_state = st.sidebar.number_input("Random state", 0, 999, 42)

c1, c2, c3 = st.columns(3)
c1.metric("Total SMS", len(data))
c2.metric("Spam", int((data["label"] == "spam").sum()))
c3.metric("Ham", int((data["label"] == "ham").sum()))

st.subheader("📊 Dataset Preview")
st.dataframe(data[["label", "message"]].head(10), use_container_width=True)

X = data["clean_message"]
y = data["label"]

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=int(random_state),
        stratify=y,
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )

models = {
    "Multinomial Naive Bayes": Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("model", MultinomialNB()),
        ]
    ),
    "Logistic Regression": Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    ),
    "Linear SVM": Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("model", LinearSVC()),
        ]
    ),
}

trained = {}
rows = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    trained[name] = model

    rows.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(
                y_test, pred, pos_label="spam", zero_division=0
            ),
            "Recall": recall_score(
                y_test, pred, pos_label="spam", zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, pred, pos_label="spam", zero_division=0
            ),
        }
    )

results = pd.DataFrame(rows)

st.subheader("📈 Model Comparison")
formatted = results.copy()
for column in ["Accuracy", "Precision", "Recall", "F1 Score"]:
    formatted[column] = formatted[column].map(lambda x: f"{x:.2%}")
st.dataframe(formatted, use_container_width=True, hide_index=True)

st.subheader("🔍 Test an SMS")
sms = st.text_area(
    "Enter your SMS:",
    placeholder="Congratulations! You won a free prize. Call now!",
    height=120,
)

model_name = st.selectbox("Select model", list(trained.keys()))

if st.button("🚀 Predict", type="primary"):
    if not sms.strip():
        st.warning("Please enter an SMS message.")
    else:
        model = trained[model_name]
        cleaned = clean_text(sms)
        prediction = model.predict([cleaned])[0]

        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ LEGITIMATE MESSAGE (HAM)")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([cleaned])[0]
            probability_table = pd.DataFrame(
                {"Class": model.classes_, "Probability": probabilities}
            )
            probability_table["Probability"] = probability_table[
                "Probability"
            ].map(lambda x: f"{x:.2%}")
            st.dataframe(
                probability_table, use_container_width=True, hide_index=True
            )
        else:
            st.info("Linear SVM does not provide probability scores by default.")

st.divider()
st.caption("SMS Spam Detection using Naive Bayes and TF-IDF")
