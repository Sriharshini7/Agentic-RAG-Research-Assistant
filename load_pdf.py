# load_pdf.py
from langchain_community.document_loaders import PyPDFLoader
from chunking import retriever  # Import the configured retriever

PDF_PATH = "constitution_pdf.pdf"

def load_pdf_to_vector_db():
    print(f"📄 Loading raw document: {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    raw_documents = loader.load()
    
    print("✂️ Generating parent-child links, vectorizing, and populating DB stores...")
    # Slices into parents and children, indexes children in Chroma, and links text in SQLite
    retriever.add_documents(raw_documents, ids=None)
    
    print("✅ Success! Your Parent-Child Database is fully built and synchronized on disk.")

if __name__ == "__main__":
    load_pdf_to_vector_db()