import streamlit as st
from PyPDF2 import PdfReader

st.set_page_config(page_title='PDF-GPT', layout = 'wide', initial_sidebar_state = 'auto')
with st.sidebar:

    st.title("PDF-GPT Demo LLM App")
    st.markdown(
        '''
        ## About
        A simple Open-AI powered application to chat and explore the contents of PDFs, the app leverages the following tech.
        1. OpenAI (For the embedding and llm models) 
        2. LangChain (High level API to interact with LLMs)
        3. Streamlit (For quick GUI)
        4. FAISS (Vector Database)
        '''
    )

    st.markdown("## Inspired by the Youtube tutorial of [Prompt Engineer](https://www.youtube.com/watch?v=RIWbalZ7sTo&ab_channel=PromptEngineering)")


def main():

    st.header("Chat with the PDF")
    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.write(text)

if __name__ == "__main__":
    main()