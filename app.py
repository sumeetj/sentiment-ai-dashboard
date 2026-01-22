import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="💭",
    layout="wide"
)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

st.title("💭 AI Sentiment Analysis Dashboard")
st.markdown("Analyze text sentiment with Hugging Face Transformers. Demo for Python/AI skills.")

model = load_model()

col1, col2 = st.columns([3, 1])
with col1:
    text = st.text_area("Enter text:", height=200, help="Paste reviews, tweets, etc.")
with col2:
    st.markdown("### Batch")
    texts = st.text_area("Multi-text (\n separated):", height=200)

if st.button("🔍 Analyze", type="primary") and text:
    with st.spinner("Processing..."):
        result = model(text)[0]
        label = result["label"].title()
        score = result["score"] * 100
        st.balloons()
        st.success(f"**{label}** sentiment ({score:.1f}% confidence)")
        st.metric("Confidence", f"{score:.1f}%")

if st.button("📊 Batch Analyze") and texts.strip():
    batch = [t.strip() for t in texts.split("\n") if t.strip()]
    if batch:
        results = model(batch)
        import pandas as pd
        df_data = [{"Text": batch[i][:80]+"..." if len(batch[i])>80 else batch[i], 
                    "Sentiment": r["label"].title(), 
                    "Confidence": f"{r["score"]*100:.1f}%"} 
                   for i, r in enumerate(results)]
        st.dataframe(pd.DataFrame(df_data))

st.markdown("---")
st.caption("Production-ready | GitHub deploy to Streamlit Cloud | #Python #AI #Streamlit")
