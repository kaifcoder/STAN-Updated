# main.py
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
import gc


# create text2text pipeline


def llm_pipeline(text):
    # model and tokenizer loading
    print("loading model in homepage")
    checkpoint = "LaMini-Flan-T5-783M"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    base_model = T5ForConditionalGeneration.from_pretrained(
        checkpoint, device_map='auto', torch_dtype=torch.float32)
    pipe_sum = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=400,
        min_length=50)
    result = pipe_sum(text)
    result = result[0]['generated_text']
    gc.collect()
    return result


# Sidebar navigation
st.set_page_config(
    page_title="Stan: Summarization and Text Analysis", page_icon="✒️")
st.title("Stan: Summarization and Text Analysis")
st.write("Welcome to Stan, your tool for summarizing and analyzing text.")

# User Input Section
st.header("Input Text")
user_input = st.text_area("Enter the text you want to analyze:", height=200)


# Button to initiate analysis
if st.button("Analyze"):
    if user_input:
        # Perform text analysis based on user's selection
        with st.spinner("Summarizing..."):
            summary = llm_pipeline(f"Summarize : {user_input}")
            st.subheader("Summary:")
            placeholder = st.empty()
            full_response = ''
            for item in summary:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)

    else:
        st.warning("Please enter some text for analysis.")
