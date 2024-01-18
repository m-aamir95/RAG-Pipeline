import os
import pickle

import streamlit as st

from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import faiss
from langchain.llms import openai
from langchain.chains.question_answering  import load_qa_chain
from langchain.callbacks import get_openai_callback

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
        filename = str(pdf.name).split('.')[0]
        embeddings_file = os.path.join("openai_computed_embeddings", f"{filename}")

        embedding_already_computed = os.path.exists(embeddings_file)

        # Vector store for our knowledge base
        vector_store = None

        if not embedding_already_computed:

            st.write(f"Computing Embeddings For -> {filename}")


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
            
            
            #Embeddings 
            # TODO,warning this will hit OpenAI API with each RUN
            embeddings = OpenAIEmbeddings()
            vector_store = faiss.FAISS.from_texts(chunks, embeddings)

            # Will create a new directory for each embedding
            embeddings_file = os.path.join("openai_computed_embeddings", f"{filename}")

            vector_store.save_local(embeddings_file)
        else:
            vector_store = faiss.FAISS.load_local(embeddings_file, OpenAIEmbeddings())
            

        # Get user input
        query = st.text_input("Ask Questions")

        if query:
            # K is important because a large K might make us go above the LLM context
            similar_docs = vector_store.similarity_search(query=query, k=3)

            llm = openai.OpenAI(temperature=0)

            with get_openai_callback() as openai_callback_with_info:
                # TODO, there are different types of chains
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                resp = chain.run(input_documents=similar_docs, question=query)
                print(openai_callback_with_info)

            st.write(resp)




if __name__ == "__main__":
    main()