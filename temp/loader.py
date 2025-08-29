# loader.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path, chunk_size=500, chunk_overlap=50):
    # 1. PDF 로딩
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(documents)
    return split_docs
