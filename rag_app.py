import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

st.title("🧠 Local RAG Chatbot (Mistral + FAISS)")

uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

if uploaded_file:
    with open("knowledge.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = TextLoader("knowledge.txt")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)

    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=gen_pipeline)

    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    question = st.text_input("Ask your question:")
    if question:
        with st.spinner("Thinking..."):
            answer = rag_chain.run(question)
        st.success("Answer:")
        st.write(answer)
