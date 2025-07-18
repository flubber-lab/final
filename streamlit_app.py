import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="Budget vs Actual Analyzer", layout="centered")
st.title("📊 Budget vs Actual Variance Analyzer with AI Suggestions")

# Load model and tokenizer
@st.cache_resource

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to("cuda")
    return tokenizer, model

tokenizer, model = load_model()

def generate_explanation_prompt(row):
    return (
        f"Category: {row['Category']}\n"
        f"Actual: {row['Actual']}\n"
        f"Budget: {row['Budget']}\n"
        f"Variance: {row['Variance']}\n"
        f"% Variance: {row['% Variance']:.1f}\n"
        f"Explain why there was a variance and give suggestions."
    )

def get_llm_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = {"Category", "Month", "Type", "Amount"}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must contain the following columns: {', '.join(required_cols)}")
    else:
        pivot_df = df.pivot_table(index=["Category", "Month"], columns="Type", values="Amount").reset_index()
        pivot_df = pivot_df.fillna(0)
        pivot_df["Variance"] = pivot_df["Actual"] - pivot_df["Budget"]
        pivot_df["% Variance"] = (pivot_df["Variance"] / pivot_df["Budget"].replace(0, 1)) * 100

        st.subheader("📋 Variance Summary")
        st.write("Generating explanations using a local LLM (Flan-T5)...")

        explanations = []
        for _, row in pivot_df.iterrows():
            prompt = generate_explanation_prompt(row)
            explanation = get_llm_response(prompt)
            explanations.append(explanation)

        pivot_df["AI Explanation"] = explanations

        st.dataframe(pivot_df[["Category", "Month", "Actual", "Budget", "Variance", "% Variance", "AI Explanation"]])

        csv = pivot_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Summary as CSV", csv, "variance_summary.csv", "text/csv")
else:
    st.info("Please upload a CSV file with 'Category', 'Month', 'Type' (Actual/Budget), and 'Amount' columns.")
