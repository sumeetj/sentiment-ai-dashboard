import streamlit as st
from transformers import pipeline
import pandas as pd

st.set_page_config(page_title="Advanced AI NLP Analyzer", page_icon="💭", layout="wide")

@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis", 
                        model="distilbert-base-uncased-finetuned-sst-2-english", 
                        top_k=None)
    zero_shot = pipeline("zero-shot-classification", 
                        model="facebook/bart-large-mnli")
    emotion = pipeline("text-classification", 
                      model="bhadresh-savani/distilbert-base-uncased-emotion")
    summarizer = pipeline("summarization", 
                         model="sshleifer/distilbart-cnn-12-6")
    return sentiment, zero_shot, emotion, summarizer

sentiment_pipe, zero_pipe, emotion_pipe, summarizer = load_models()

st.title("💭 Advanced AI NLP Analyzer 🚀")
st.markdown("Sentiment + Tones + Emotions + Summary (Fixed tabs).")

tab1, tab2, tab3, tab4 = st.tabs(["1. Sentiment", "2. Custom Tone", "3. Emotions", "4. Summary"])

with tab1:
    text = st.text_area("Enter text here:", height=200, key="text_sentiment")
    if st.button("Analyze Sentiment", type="primary", key="btn_sentiment") and text:
        results = sentiment_pipe(text)
        df = pd.DataFrame([{"Label": r["label"], "Score": f"{r['score']:.1%}"} for r in results[0]])
        st.dataframe(df, use_container_width=True)
        best = max(results[0], key=lambda x: x['score'])
        st.success(f"**{best['label']}** ({best['score']:.1%})")

with tab2:
    text2 = st.text_area("Enter text for tones:", height=200, key="text_tone")
    tones = st.text_input("Custom tones (comma-separated):", "joyful,sarcastic,neutral,angry,excited", key="input_tones")
    tones_list = [t.strip() for t in tones.split(",")]
    if st.button("Detect Tone", key="btn_tone") and text2 and tones_list:
        result = zero_pipe(text2, candidate_labels=tones_list, multi_label=True)
        df_tone = pd.DataFrame({"Tone": result['labels'][:5], "Score": [f"{s:.1%}" for s in result['scores'][:5]]})
        st.bar_chart(df_tone.set_index('Tone')['Score'].str.rstrip('%').astype(float)/100)
        st.dataframe(df_tone)

with tab3:
    text3 = st.text_area("Enter text for emotions:", height=200, key="text_emotion")
    if st.button("Detect Emotions", key="btn_emotion") and text3:
        results = emotion_pipe(text3)
        df_em = pd.DataFrame([{"Emotion": r["label"], "Score": f"{r['score']:.1%}"} for r in results])
        st.dataframe(df_em, use_container_width=True)
        top_em = max(results, key=lambda x: x['score'])
        st.balloons()
        st.success(f"**{top_em['label'].upper()}** ({top_em['score']:.1%})")

with tab4:
    text4 = st.text_area("Long text to summarize:", height=250, key="text_summary")
    if st.button("Generate Summary", key="btn_summary") and text4:
        with st.spinner("Summarizing..."):
            summary = summarizer(text4, max_length=100, min_length=20, do_sample=False)[0]['summary_text']
            st.info(f"**Summary**: {summary}")

st.markdown("---")
st.caption("✅ No duplicate IDs | Multi-model NLP demo | Live on Streamlit Cloud")
