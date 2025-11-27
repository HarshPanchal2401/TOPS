import streamlit as st
from transformers import pipeline
from transformers.pipelines import Pipeline
from typing import Dict
import pandas as pd
import io

st.set_page_config(page_title="Multi-Task NLP Assistant", layout="wide")

# --- Helper: cached loader for pipelines ---
@st.cache_resource
def get_pipeline(task: str, model_name: str = None, device: int = -1) -> Pipeline:
    """Load and cache a transformers pipeline.

    device: -1 for CPU, >=0 for GPU device index.
    """
    if task == "summarization":
        model = model_name or "facebook/bart-large-cnn"
        return pipeline("summarization", model=model, device=device)
    if task == "question-answering":
        model = model_name or "deepset/roberta-base-squad2"
        return pipeline("question-answering", model=model, device=device)
    if task == "ner":
        model = model_name or "dbmdz/bert-large-cased-finetuned-conll03-english"
        return pipeline("ner", model=model, aggregation_strategy="simple", device=device)
    if task == "translation":
        # model_name must be provided for translation pipeline caller
        model = model_name or "Helsinki-NLP/opus-mt-en-fr"
        return pipeline("translation", model=model, device=device)
    raise ValueError(f"Unsupported task: {task}")


# --- UI layout ---
st.sidebar.title("Multi-Task NLP Assistant")
selected_task = st.sidebar.selectbox(
    "Select task",
    ("Text Summarization", "Question Answering", "Named Entity Recognition", "Translation (English → FR/ES/DE)")
)

st.sidebar.markdown("---")
use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=False)

st.title("Multi-Task NLP Assistant")
st.write("Choose a task from the sidebar, paste your text, and run the model.")

# --- Input area ---
if selected_task == "Question Answering":
    st.subheader("Question Answering")
    context = st.text_area("Context / Paragraph (required)", height=200)
    question = st.text_input("Question (required)")
    run_btn = st.button("Run QA")

elif selected_task == "Text Summarization":
    st.subheader("Text Summarization")
    context = st.text_area("Text to summarize (required)", height=300)
    max_len = st.slider("Max summary length", min_value=20, max_value=300, value=120)
    min_len = st.slider("Min summary length", min_value=5, max_value=100, value=30)
    run_btn = st.button("Summarize")

elif selected_task == "Named Entity Recognition":
    st.subheader("Named Entity Recognition (showing PERSON & ORG only)")
    context = st.text_area("Text for NER (required)", height=250)
    download_csv = st.checkbox("Save NER output to CSV", value=False)
    run_btn = st.button("Extract Entities")

else:  # Translation
    st.subheader("Translation (English → French / Spanish / German)")
    context = st.text_area("English text to translate (required)", height=250)
    target_lang = st.selectbox("Target language", ("French", "Spanish", "German"), index=0)
    run_btn = st.button("Translate")

# --- Device selection for pipeline ---
# transformer pipelines accept device index; -1 for CPU.
device = 0 if use_gpu else -1

# --- Run selected task ---
if run_btn:
    if not context.strip():
        st.error("Please provide input text in the text area before running the model.")
    else:
        try:
            with st.spinner("Loading model and running... This may take a few seconds."):
                if selected_task == "Text Summarization":
                    pipe = get_pipeline("summarization", device=device)
                    # run
                    summary = pipe(context, max_length=max_len, min_length=min_len, do_sample=False)
                    out = summary[0]["summary_text"]
                    st.subheader("Summary")
                    st.markdown(out)

                elif selected_task == "Question Answering":
                    if not question.strip():
                        st.error("Please type a question for the QA task.")
                    else:
                        pipe = get_pipeline("question-answering", device=device)
                        result = pipe({"question": question, "context": context})
                        st.subheader("Answer")
                        st.markdown(f"**{result['answer']}**")
                        st.write(f"Score: {result['score']:.4f}")

                elif selected_task == "Named Entity Recognition":
                    pipe = get_pipeline("ner", device=device)
                    entities = pipe(context)
                    # Filter only PERSON and ORG (different models use labels like PER / ORG)
                    df_rows = []
                    for ent in entities:
                        label = ent.get("entity_group") or ent.get("entity")
                        text = ent.get("word") or ent.get("entity")
                        score = ent.get("score", 0.0)
                        # unify label names
                        label_norm = label.upper() if label else ""
                        if label_norm.startswith("PER") or label_norm.startswith("PERSON"):
                            typ = "PERSON"
                        elif label_norm.startswith("ORG"):
                            typ = "ORG"
                        else:
                            typ = None
                        if typ:
                            df_rows.append({"Entity": text, "Type": typ, "Confidence": round(score, 4)})

                    if not df_rows:
                        st.info("No PERSON or ORG entities were found in the given text.")
                    else:
                        df = pd.DataFrame(df_rows)
                        st.subheader("Entities (PERSON / ORG)")
                        st.table(df)

                        if download_csv:
                            csv_buf = io.StringIO()
                            df.to_csv(csv_buf, index=False)
                            csv_bytes = csv_buf.getvalue().encode()
                            st.download_button("Download CSV", data=csv_bytes, file_name="ner_entities.csv")

                else:  # Translation
                    # choose model according to target language
                    model_map = {
                        "fr": "Helsinki-NLP/opus-mt-en-fr",
                        "es": "Helsinki-NLP/opus-mt-en-es",
                        "de": "Helsinki-NLP/opus-mt-en-de",
                    }
                    model_name = model_map.get(target_lang, "Helsinki-NLP/opus-mt-en-fr")
                    pipe = get_pipeline("translation", model_name=model_name, device=device)
                    translations = pipe(context, max_length=400)
                    translated = translations[0]["translation_text"]
                    st.subheader(f"Translated to {target_lang}")
                    st.markdown(translated)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


# Footer / Usage notes
st.markdown("---")
st.caption("Models are loaded lazily and cached. On first run the app may take time to load models. For GPU support, ensure CUDA and a compatible PyTorch are installed.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Tips**:\n- For long inputs, summarization may take longer.\n- Toggle GPU only if available on the host machine.")
