import os  
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
from langchain_chroma import Chroma
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

VECTOR_DIR = "vector_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_vector_store():
    return Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embeddings
    )


def store_run_vector(run_id: str, text: str):
    db = get_vector_store()
    db.add_texts(
        texts=[text],
        metadatas=[{"run_id": run_id}],
        ids=[run_id]
    )
    # db.persist()


def retrieve_context(query: str, k: int = 2) -> str:
    db = get_vector_store()
    results = db.similarity_search(query, k=k)
    return "\n\n".join(r.page_content for r in results)

