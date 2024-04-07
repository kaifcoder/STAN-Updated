import streamlit as st
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
import os
import gc
import torch
# App title
st.set_page_config(page_title="Chat with LaMini", page_icon="🤖")

# load model lists from root directory
model_list = os.listdir("models")

default_model = "LaMini-Flan-T5-248M"


@st.cache_resource(experimental_allow_widgets=True)
def ChatModel(selected_model):
    print(selected_model)
    # return AutoModelForSeq2SeqLM.from_pretrained(
    #     "LaMini-Flan-T5-248M",
    #     # model_file="codellama-7b.Q2_K.gguf",
    #     # model_type='T5',
    #     # temperature=temperature,
    #     # top_p=top_p,
    # )
    if selected_model is None:
        pipe = pipeline('text2text-generation',
                        model=f"models/{default_model}")
    pipe = pipeline('text2text-generation', model=f"models/{selected_model}")
    return pipe


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?\n\nIf you are looking for document information retrieval please upload a document."}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?\n\nIf you are looking for document information retrieval please upload a document."}]


def clear_resources():
    gc.collect()
    ChatModel.clear()
    torch.cuda.empty_cache()


with st.sidebar:
    st.title('Chatbot')
    selected = st.sidebar.selectbox(
        'Select Model', (model_list), placeholder=default_model,
        help="Select a model from the list to chat with it."

    )

    if selected != default_model:
        clear_resources()
        chat_model = ChatModel(selected_model=selected)

    else:
        chat_model = ChatModel(selected_model=default_model)

    use_modal = chat_model
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        with open(os.path.join("models", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Saved File")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


def generate_llama2_response(prompt_input):
    # string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    # for dict_message in st.session_state.messages:
    #     if dict_message["role"] == "user":
    #         string_dialogue += "User: " + dict_message["content"] + "\\n\\n"
    #     else:
    #         string_dialogue += "Assistant: " + \
    #             dict_message["content"] + "\\n\\n"
    output = use_modal(f"{prompt_input}", max_length=512,
                       do_sample=True)[0]['generated_text']
    return output


# create a upload file button and upload the file to the root directory.


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
