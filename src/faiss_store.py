import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from rag import Encoder

encoder = Encoder()

def store_documents_in_faiss(documents, faiss_index_path, embedding_function):
    index = faiss.IndexFlatL2(len(encoder.embedding_function.embed_query('hello world')))
    vector_store = FAISS(embedding_function=embedding_function,index=index,docstore=InMemoryDocstore(),index_to_docstore_id={})
    vector_store.add_documents(documents)
    vector_store.save_local(faiss_index_path)
    return vector_store


def load_faiss_index(faiss_index_path, embedding_function):
    return FAISS.load_local(faiss_index_path,embedding_function, allow_dangerous_deserialization=True)