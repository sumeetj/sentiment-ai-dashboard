import streamlit as st
from transformers import pipeline
import pandas as pd

st.set_page_config(page_title="🤖 AI NLP Analyzer Pro", page_icon="💭", layout="wide")

@st.cache_resource
def load_pipelines():
    return {
        "sentiment": pipeline("sentiment-analysis", 
                             model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
                             top_k=3),
        "zero_shot": pipeline("zero-shot-classification", 
                             model="facebook/bart-large-mnli"),
        "emotion": pipeline("text-classification", 
                           model="j-hartmann/emotion-english-distilroberta-base")
    }

pipes = load_pipelines()

st.title("🤖 Production AI NLP Analyzer")
st.markdown("*Sentiment · Custom Tones · Emotions* - Stable on Streamlit Cloud")

tab1, tab2, tab3 = st.tabs(["📊 Sentiment", "🎯 Custom Tones", "😊 Emotions"])

with tab1:
    text1 = st.text_area("Paste review/tweet:", key="sentiment_input", height=150, 
                        placeholder="This product is amazing!")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Analyze Sentiment", type="primary", key="sentiment_btn"):
            if text1:
                res = pipes["sentiment"](text1)
                df = pd.DataFrame(res[0])
                df["score"] = df["score"].map("{:.1%}".format)
                st.dataframe(df, use_container_width=True)
                st.metric("Top Sentiment", df.iloc[0]["label"], f"{df.iloc[0]['score']}")

with tab2:
    text2 = st.text_area("Paste text:", key="tone_input", height=150)
    labels_str = st.text_input("Tone labels:", "positive,negative,neutral,joyful,angry,sarcastic", key="tone_labels")
    labels = [l.strip() for l in labels_str.split(",")]
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Classify Tones", key="tone_btn"):
            if text2 and labels:
                res = pipes["zero_shot"](text2, candidate_labels=labels, multi_label=True)
                tone_df = pd.DataFrame({
                    "Tone": res["labels"][:6], 
                    "Score": [f"{s:.1%}" for s in res["scores"][:6]]
                })
                st.bar_chart(tone_df.set_index("Tone")["Score"].str[:-1].astype(float)/100)
                st.dataframe(tone_df)

with tab3:
    text3 = st.text_area("Paste text:", key="emotion_input", height=150)
    if st.button("Detect Emotions", type="primary", key="emotion_btn"):
        if text3:
            res = pipes["emotion"](text3)
            df_em = pd.DataFrame([{"Emotion": r["label"], "Score": f"{r['score']:.1%}"} for r in res])
            df_em = df_em.sort_values("Score", ascending=False)
            st.dataframe(df_em, use_container_width=True)
            top = df_em.iloc[0]
            st.balloons()
            st.metric("Dominant Emotion", top["Emotion"], top["Score"])

# Sidebar skills showcase
with st.sidebar:
    st.markdown("### 🛠️ Tech Stack")
    st.write("- **Transformers**: RoBERTa, BART, DistilRoBERTa")
    st.write("- **Zero-shot classification**")
    st.write("- **Streamlit tabs + caching**")
    st.write("- **Pandas + viz**")
    st.markdown("[Source GitHub](https://github.com/sumeetj/sentiment-ai-dashboard)")

st.markdown("---")
st.caption("🤖 Advanced NLP Analysis Tool | Powered by Hugging Face Transformers")
