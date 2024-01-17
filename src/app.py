import streamlit as st

from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter

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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # Tokens
            chunk_overlap=200, # Overlap between the adjacent chunks
                               # Both adjacent chunks will share 200 tokens 
                               # This helps in not loosing the context of the 
                               # conversation 
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        st.write(chunks)


if __name__ == "__main__":
    main()