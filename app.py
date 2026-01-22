import streamlit as st
from transformers import pipeline
import pandas as pd
import torch

st.set_page_config(page_title="Advanced AI NLP Analyzer", page_icon="💭", layout="wide")

@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis", 
                        model="distilbert-base-uncased-finetuned-sst-2-english", 
                        top_k=None)  # Both scores, no warning
    zero_shot = pipeline("zero-shot-classification", 
                        model="facebook/bart-large-mnli")
    emotion = pipeline("text-classification", 
                      model="bhadresh-savani/distilbert-base-uncased-emotion")
    summarizer = pipeline("summarization", 
                         model="sshleifer/distilbart-cnn-12-6")
    return sentiment, zero_shot, emotion, summarizer

sentiment_pipe, zero_pipe, emotion_pipe, summarizer = load_models()

st.title("💭 Advanced AI NLP Analyzer 🚀")
st.markdown("Multi-model: Sentiment (2 labels), Tones (zero-shot), Emotions (6), Summary.")

tab1, tab2, tab3, tab4 = st.tabs(["1. Sentiment", "2. Custom Tone", "3. Emotions", "4. Summary"])

with tab1:
    text = st.text_area("Text:", height=200)
    if st.button("Analyze", type="primary") and text:
        results = sentiment_pipe(text)
        df = pd.DataFrame([{"Label": r["label"], "Score": f"{r['score']:.1%}"} for r in results[0]])
        st.dataframe(df, use_container_width=True)
        best = max(results[0], key=lambda x: x['score'])
        st.success(f"**{best['label']}** ({best['score']:.1%})")

with tab2:
    text2 = st.text_area("Text:", height=200)
    tones = st.text_input("Tones (comma sep.):", "joyful,sarcastic,neutral,angry,excited")
    tones = [t.strip() for t in tones.split(",")]
    if st.button("Detect Tone") and text2:
        result = zero_pipe(text2, candidate_labels=tones, multi_label=True)
        df_tone = pd.DataFrame({"Tone": result['labels'][:5], "Score": [f"{s:.1%}" for s in result['scores'][:5]]})
        st.bar_chart(df_tone.set_index('Tone')['Score'].str.rstrip('%').astype(float)/100)
        st.dataframe(df_tone)

with tab3:
    text3 = st.text_area("Text:", height=200)
    if st.button("Detect Emotions") and text3:
        results = emotion_pipe(text3)
        df_em = pd.DataFrame([{"Emotion": r["label"], "Score": f"{r['score']:.1%}"} for r in results])
        st.dataframe(df_em, use_container_width=True)
        top_em = max(results, key=lambda x: x['score'])
        st.balloons()
        st.success(f"**{top_em['label'].upper()}** ({top_em['score']:.1%})")

with tab4:
    text4 = st.text_area("Long text:", height=250)
    if st.button("Summarize") and text4:
        with st.spinner("Summarizing..."):
            summary = summarizer(text4, max_length=100, min_length=20, do_sample=False)[0]['summary_text']
            st.info(f"**Summary**: {summary}")

st.markdown("---")
st.caption("Production: Explicit models, no warnings. #Python #AI #NLP")
