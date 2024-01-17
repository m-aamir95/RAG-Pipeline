import streamlit as st

st.set_page_config(page_title='PDF-GPT', layout = 'wide', initial_sidebar_state = 'auto')
with st.sidebar:

    st.title("PDF-GPT Demo LLM App")
    st.markdown(
        '''
        ## About
        A simple Open-AI powered application to chat and explore the contents of PDFs, the app leverages the following tech.
        1. OpenAI   
        2. LangChain (High level API to interact with LLMs)
        3. Streamlit (For quick GUI)
        4. FAISS (Vector Database)
        '''
    )

    st.markdown("## Inspired by the Youtube tutorial of [Prompt Engineer](https://www.youtube.com/watch?v=RIWbalZ7sTo&ab_channel=PromptEngineering)")
    