# custom_tools.py
import json
import sqlite3
from typing import Iterator, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever

from chunking import retriever



# ==========================================
#  MODULAR AGENT SEARCH TOOL
# ==========================================

@tool
def legal_agent(query: str) -> str:
    """Searches the comprehensive legal terms as per the constitution.
    This fetches broad page context blocks using parent-child retrieval."""
    
    # Hits Chroma radar first, then maps back to the SQLite bookshelf
    docs = retriever.invoke(query)
    
    formatted_context = []
    for i, doc in enumerate(docs[:3]):  # Extract top 3 deep parent contexts
        page_num = doc.metadata.get("page", "Unknown")
        formatted_context.append(f"[Context Block #{i+1} from Page {page_num}]:\n{doc.page_content}")
        
    return "\n\n".join(formatted_context)



