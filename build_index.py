import os
import glob
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("err: GEMINI_API_KEY is not set in .env")
    exit(1)
docs_dir = 'all_docs'
pdf_dir = os.path.join(docs_dir, 'ПДФ')

documents = []

print("load pdfs")
if os.path.exists(pdf_dir):
    for root, dirs, files in os.walk(pdf_dir):
        for file in files:
            if file.lower().endswith('.pdf') and not file.startswith('~'):
                path = os.path.join(root, file)
                print(f"Loading {path}...")
                try:
                    loader = PyPDFLoader(path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {path}: {e}")
else:
    print(f"dir {pdf_dir} not found.")

print("\nload excel files")
for root, dirs, files in os.walk(docs_dir):
    if 'archive' in root.lower() and 'archive-2' not in root.lower():
        continue
        
    for file in files:
        if file.lower().endswith('.xlsx') and not file.startswith('~'):
            path = os.path.join(root, file)
            print(f"loading {path}...")
            try:
                excel_data = pd.read_excel(path, sheet_name=None)
                for sheet_name, df in excel_data.items():
                    text = df.to_string(index=False)
                    text = f"Файл: {file}, Лист: {sheet_name}\n\n{text}"
                    documents.append(Document(page_content=text, metadata={"source": path}))
            except Exception as e:
                print(f"err loading {path}: {e}")

print(f"\loaded {len(documents)} documents")

print("\nsplitting text")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(f" chunks: {len(docs)}")

print("\ngen emb")

embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

vectorstore = FAISS.from_documents(docs, embeddings)

print("\nsave")
vectorstore.save_local("faiss_index")
print("done")
