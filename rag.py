import os
import re
import fitz  # PyMuPDF for extracting text from PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel


app=FastAPI()
load_dotenv()

class RAG:
    def __init__(self, pdf_path="resume.pdf", model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", embedding_model="sentence-transformers/all-MiniLM-L6-v2", chunk_size=1000, chunk_overlap=200):
        self.pdf_path = pdf_path
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None
        self.qa_chain = None
        self.huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    def extract_text_from_pdf(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"The file at {self.pdf_path} was not found.")
        
        try:
            doc = fitz.open(self.pdf_path)
        except fitz.FileDataError as e:
            raise Exception(f"Failed to open the file. Please check the file path or ensure that the file is a valid PDF.") from e

        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()

        return text
    
    def create_documents_from_text(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        return documents
    
    def index_documents(self, documents):
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vectorstore = FAISS.from_documents(documents, embeddings)
    
    def setup_retrieval_qa(self):
        if not self.vectorstore:
            raise Exception("Vector store is not initialized. Index documents first.")
        
        retriever = self.vectorstore.as_retriever()
        llm = HuggingFaceHub(repo_id=self.model_name, huggingfacehub_api_token=self.huggingface_token)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=retriever
        )
    
    def ask_question(self, question):
        if not self.qa_chain:
            raise Exception("QA chain is not initialized. Set up retrieval QA first.")
        
        answer = self.qa_chain.run(question)
        return answer
    
    def run_pipeline(self, question):
        text = self.extract_text_from_pdf()
        print("Text extracted successfully.")
        
        documents = self.create_documents_from_text(text)
        self.index_documents(documents)
        
        self.setup_retrieval_qa()
        
        answer = self.ask_question(question)
        return answer

rag_system = RAG() 

def extract_after_helpful_answer(text):
    match = re.search(r"Helpful Answer:(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""

class QuestionRequest(BaseModel):
    question: str
@app.post("/ask_question")
def get_answer(question:QuestionRequest):
    try:
        answer=rag_system.run_pipeline(question.question)
        answer=extract_after_helpful_answer(answer)
        print(answer)
        return {"answer":answer}
    except Exception as e:
        return {"error":e.__str__()}
