import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DATA_DIR="data/raw"
FAISS_INDEX_PATH=os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP=int(os.getenv("CHUNK_OVERLAP",50))

def load_documents(data_dir: str = DATA_DIR):
    loader=DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents=loader.load()
    print(f"loaded {len(documents)} pages")
    return documents

def chunk_documents(documents, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    splitter= RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n","\n","."," ",""]
    )
    chunks=splitter.split_documents(documents)
    print(f"created {len(chunks)} chunks")
    return chunks

def get_embedding_model(model_name: str= EMBEDDING_MODEL):
    print(f"loading embedding model: {model_name}")
    embeddings= HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings":True}
    )
    print("embedding model loaded")
    return embeddings

def build_faiss_index(chunks, index_path: str= FAISS_INDEX_PATH):
    os.makedirs(index_path, exist_ok=True)
    embeddings= get_embedding_model()
    print(f"building faiss index for {len(chunks)} chunks...")
    vectorstore=FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"faiss index saved to  {index_path}")
    return vectorstore

def load_faiss_index(index_path: str=FAISS_INDEX_PATH):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"faiss index not found at {index_path}. Run ingestion.py first")
    embeddings=get_embedding_model()
    vectorstore=FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
        )
    print(f"FAISS index loaded from {index_path}")
    return vectorstore

if __name__ == "__main__" :
    print("===starting ingestion===")
    docs=load_documents()
    chunks= chunk_documents(docs)
    build_faiss_index(chunks)
    print("===ingestion complete===")