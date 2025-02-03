# main.py

import os
import streamlit as st
from rag_utility import process_document_to_chroma_db, answer_question

# Set the working directory
working_dir = os.getcwd()

st.title("🐋 DeepSeek-R1 - Document RAG")

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Define save path
    save_path = os.path.join(working_dir, uploaded_file.name)
    # Save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_document = process_document_to_chroma_db(uploaded_file.name)
    st.info("Document Processed Successfully")

# Text widget to get user input
user_question = st.text_input("Ask your question about the document")

if st.button("Answer"):
    if user_question:
        answer = answer_question(user_question)
        st.markdown("### DeepSeek-R1 Response")
        st.markdown(answer)
    else:
        st.warning("Please enter a question.")


        