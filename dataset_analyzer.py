import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from dotenv import load_dotenv
import os
from openai import OpenAI
import json

# Load OpenAI API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="Learn About Your Dataset!",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Learn About Your Dataset!")

# File upload
uploaded_file = st.file_uploader(
    "Upload CSV file - Must be under 1MB (though Streamlit supports up to 200 MB)",
    type=["csv"]
)

if uploaded_file:
    if uploaded_file.size > 1_000_000:
        st.error("File too large. Please upload a CSV under 1 MB.")
        st.stop()

    # Reset chat if new file uploaded
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = uploaded_file.name
        st.session_state.chat_history = []
    elif st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.chat_history = []
        st.session_state.last_uploaded_file = uploaded_file.name

    df = pd.read_csv(uploaded_file)

    # Dataset summary
    dataset_summary = {
        "columns": df.columns.tolist(),
        "numeric_summary": df.describe().to_dict(),
        "categorical_summary": df.describe(include=['object']).to_dict(),
        "sample_rows": df.head(10).to_dict(orient='records')
    }

    # Chunking
    def chunk_dataframe(df, max_rows=50):
        chunks = []
        for start in range(0, len(df), max_rows):
            chunks.append(df.iloc[start:start + max_rows])
        return chunks

    chunks = chunk_dataframe(df, max_rows=50)
    chunk_summaries = []

    # Few-shot chunk summarization
    for i, chunk in enumerate(chunks):
        chunk_csv = chunk.to_csv(index=False)
        chunk_prompt = f"""
        You are a data analyst AI. Summarize the following dataset chunk.
        Respond STRICTLY in JSON format. Do NOT include any extra text outside JSON.

        Required JSON structure:
        {{
        "chunk_summary": "1-3 sentence high-level summary of this chunk",
        "numeric_summary": {{
            "mean": {{}},
            "median": {{}},
            "std": {{}},
            "min": {{}},
            "max": {{}},
            "outliers": {{}}
        }},
        "categorical_summary": {{
            "top_values": {{}},
            "rare_values": {{}},
            "missing_count": {{}}
        }},
        "missing_data": {{}}
        }}

        Example:
        Chunk CSV:
        col1,col2,col3
        1,apple,10
        2,banana,15
        3,apple,12

        JSON Output:
        {{
        "chunk_summary": "Small chunk with 3 rows. Numeric values show low variation; 'col2' mostly contains 'apple'.",
        "numeric_summary": {{
            "mean": {{"col1": 2, "col3": 12.33}},
            "median": {{"col1": 2, "col3": 12}},
            "std": {{"col1": 1, "col3": 2.52}},
            "min": {{"col1": 1, "col3": 10}},
            "max": {{"col1": 3, "col3": 15}},
            "outliers": {{}}
        }},
        "categorical_summary": {{
            "top_values": {{"col2": ["apple"]}},
            "rare_values": {{"col2": ["banana"]}},
            "missing_count": {{"col2": 0}}
        }},
        "missing_data": {{"col1": 0, "col2": 0, "col3": 0}}
        }}

        Now summarize this chunk:

        {chunk_csv}
        """
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": chunk_prompt}]
        )
        try:
            summary_json = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            summary_json = {
                "chunk_summary": response.choices[0].message.content
            }
        chunk_summaries.append(summary_json)

    # Streamlit Tabs
    tabs = st.tabs(["Chatbot", "AI Insights", "Data Preview & Stats"])

    # Chatbot Tab
    with tabs[0]:
        st.header("Ask Questions About Your Dataset")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Ask anything about your dataset")

        if st.button("Send"):
            if user_input:
                messages = [
                    {"role": "system", "content": "You are a helpful data analyst AI."}]
                for h in st.session_state.chat_history:
                    messages.append(h)

                chatbot_prompt = f"""
                You are a helpful data analyst AI. Use the summaries of dataset chunks to answer the user's question.
                Respond concisely and clearly. Include trends, correlations, anomalies, and missing data warnings. Suggest further analysis if useful. Mention Responsible AI considerations if relevant.

                Example:
                Chunk summaries:
                [
                {{
                    "chunk_summary": "Chunk 1 shows numeric values increasing steadily.",
                    "numeric_summary": {{"mean": {{"col1": 5}}, "median": {{"col1": 5}}, "std": {{"col1": 0}}, "min": {{"col1": 5}}, "max": {{"col1": 5}}, "outliers": {{}}}},
                    "categorical_summary": {{"top_values": {{"col2": ["A"]}}, "rare_values": {{}}, "missing_count": {{"col2": 0}}}},
                    "missing_data": {{"col1": 0, "col2": 0}}
                }}
                ]
                User question: "Are there any anomalies in the dataset?"
                Answer: "Chunk 1 shows a perfectly steady trend for 'col1', no numeric anomalies detected. 'col2' is dominated by category 'A'. No missing data is present. Suggest checking for potential skew in small sample sizes."

                Now use the following chunk summaries to answer the user's question:
                {json.dumps(chunk_summaries, indent=2)}

                User question: {user_input}
                """
                messages.append({"role": "user", "content": chatbot_prompt})

                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=messages
                )
                answer = response.choices[0].message.content

                # Save chat history
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input})
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer})

        # Display chat history (newest first)
        if st.session_state.chat_history:
            st.markdown("---")
            for i in range(len(st.session_state.chat_history) - 2, -1, -2):
                user_msg = st.session_state.chat_history[i]["content"]
                ai_msg = st.session_state.chat_history[i+1]["content"]
                st.markdown(f"**You:** {user_msg}")
                st.markdown(f"**AI:** {ai_msg}")
                st.markdown("-----")

    # AI Insights Tab
    with tabs[1]:
        st.header("Generate AI Overview and Insights")
        if st.button("Generate"):
            with st.spinner("AI analyzing your dataset and generating insights..."):
                aggregated_prompt = f"""
                You are a data analyst AI. Here are summaries from dataset chunks:
                {json.dumps(chunk_summaries, indent=2)}

                Please provide a detailed textual analysis including:
                - Overall trends, patterns, and correlations
                - Key anomalies or outliers
                - Missing data insights
                - Interesting categorical and numeric observations
                - Suggestions for further analysis or preprocessing
                - Responsible AI considerations (bias, fairness, uncertainty)

                Respond ONLY in text (no code or plots). Format clearly with bullet points where helpful.
                """
                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": aggregated_prompt}]
                )

                ai_output_text = response.choices[0].message.content
                st.subheader("Detailed AI Overview")
                st.markdown(ai_output_text)

    # Data Preview & Stats Tab
    with tabs[2]:
        st.header("Dataset Preview & Statistics")
        st.subheader("Preview of Data")
        st.dataframe(df.head(20))
        st.subheader("Basic Statistics")
        st.write(df.describe())
