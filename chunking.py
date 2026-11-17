# chunking.py
import json
import sqlite3
from typing import Iterator, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever






# ==========================================
# SQLite Document Store
# ==========================================
class SQLiteDocStore(BaseStore[str, Document]):  #str = key and dcoument is or parent chunk
    """A simple sqlite3-based docstore for ParentDocumentRetriever."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_table()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
    #getting values 
    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        with self._get_conn() as conn:
            placeholders = ",".join("?" for _ in keys)
            cursor = conn.execute(
                f"SELECT key, value FROM parent_chunks WHERE key IN ({placeholders})",
                list(keys),
            )
            rows = {row[0]: row[1] for row in cursor.fetchall()}

        results = []
        for k in keys:
            if k in rows:
                obj = json.loads(rows[k])
                results.append(Document(page_content=obj["page_content"], metadata=obj["metadata"]))
            else:
                results.append(None)
        return results

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        with self._get_conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO parent_chunks (key, value) VALUES (?, ?)",
                [(k, json.dumps({"page_content": v.page_content, "metadata": v.metadata})) for k, v in key_value_pairs],
            )

    def mdelete(self, keys: Sequence[str]) -> None:
        with self._get_conn() as conn:
            placeholders = ",".join("?" for _ in keys)
            conn.execute(
                f"DELETE FROM parent_chunks WHERE key IN ({placeholders})",
                list(keys),
            )

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        with self._get_conn() as conn:
            if prefix:
                cursor = conn.execute("SELECT key FROM parent_chunks WHERE key LIKE ?", (f"{prefix}%",))
            else:
                cursor = conn.execute("SELECT key FROM parent_chunks")
            for row in cursor:
                yield row[0]


# ==========================================
# Core Engine: Embeddings + Stores
# ==========================================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Chroma manages the 384-dimension child vectors
#creating db - as chroma_db and applying embedding function . later we will use vectorstore to retreive unique id associated with child chunk
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# SQLiteDocStore saves parent chunks as JSON in a plain sqlite3 database
#coverting our parent table into database so that it is easy and fast to retrieve data using sqlite
docstore = SQLiteDocStore(db_path="parent_store.db")

# ==========================================
# Splitter Configurations
# ==========================================
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# ==========================================
# Unified Parent-Child Retriever
# ==========================================
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

