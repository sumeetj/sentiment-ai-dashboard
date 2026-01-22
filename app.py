import streamlit as st
from transformers import pipeline
import pandas as pd

st.set_page_config(page_title="Advanced AI Sentiment Analyzer", page_icon="💭", layout="wide")

@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis", return_all_scores=True)
    zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    emotion = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
    return sentiment, zero_shot, emotion

sentiment_pipe, zero_pipe, emotion_pipe = load_models()

st.title("💭 Advanced AI NLP Analyzer")
st.markdown("Sentiment + Tone + Emotions + Summary. Updated for multi-label analysis.")

tab1, tab2, tab3, tab4 = st.tabs(["Sentiment", "Custom Tone", "Emotions", "Summary"])

with tab1:
    text = st.text_area("Text:", height=200)
    if st.button("Analyze Sentiment", type="primary") and text:
        results = sentiment_pipe(text)[0]
        df = pd.DataFrame(results)
        st.dataframe(df.style.format({"score": "{:.1%}"}), use_container_width=True)
        best = max(results, key=lambda x: x['score'])
        st.success(f"**{best['label']}** ({best['score']:.1%})")

with tab2:
    text2 = st.text_area("Text for tone:", height=200)
    tones = st.text_input("Custom tones (comma-separated):", "joyful,sarcastic,neutral,angry,excited")
    tones = [t.strip() for t in tones.split(",")]
    if st.button("Detect Tone") and text2 and tones:
        result = zero_pipe(text2, candidate_labels=tones, multi_label=True)
        df_tone = pd.DataFrame({"Tone": result['labels'], "Score": [f"{s:.1%}" for s in result['scores']]})
        st.bar_chart(df_tone.set_index('Tone')['Score'].map(lambda x: float(x.strip('%'))/100))
        st.dataframe(df_tone)

with tab3:
    text3 = st.text_area("Text for emotions:", height=200)
    if st.button("Detect Emotions") and text3:
        results = emotion_pipe(text3)
        df_em = pd.DataFrame([r for r in results])
        st.dataframe(df_em.style.format({"score": "{:.1%}"}), use_container_width=True)
        top_em = max(results, key=lambda x: x['score'])
        st.success(f"**Dominant Emotion**: {top_em['label'].upper()} ({top_em['score']:.1%})")

with tab4:
    from transformers import pipeline
    @st.cache_resource
    def load_summarizer():
        return pipeline("summarization")
    summarizer = load_summarizer()
    text4 = st.text_area("Long text to summarize:", height=250)
    if st.button("Summarize") and text4:
        summary = summarizer(text4, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        st.write("**Summary**:", summary)

st.markdown("---")
st.caption("Enhanced: Zero-shot tones, 6 emotions, summarization. Deployed live!")
