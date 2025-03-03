from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.docstore.document import Document
from langchain_groq import ChatGroq

load_dotenv()

CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)

class Encoder:
    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L12-v2", device="cpu"
    ):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=CACHE_DIR,
            model_kwargs={"device": device},
        )

class FaissDb:
    def __init__(self, embedding_function, db=None):
        self.db = db
        self.embedding_function = embedding_function
        # FAISS.from_documents(
        #     docs, embedding_function, distance_strategy=DistanceStrategy.COSINE
        # )

    def similarity_search(self, question: str, k: int = 5):
        if self.db is None:
            return ""
        retrieved_docs = self.db.similarity_search(question, k=k)
        context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
        return context
    
def load_and_split_txt_files(folder_path: str, chunk_size: int = 256):
    # List all text files in the specified folder
    file_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.txt')]

    # Load the content of each text file
    documents = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            documents.append(Document(page_content=content, metadata={"source": file_path}))

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size // 10),
        strip_whitespace=True,
    )

    # Split the texts into chunks
    docs = text_splitter.split_documents(documents)
    return docs

def retrieve_documents(query, faiss_db, k=5):
    context = faiss_db.similarity_search(query, k=k)
    return context

def generate_response_with_groq(query, context):
#     if context is None or context == "":
#         prompt = f"""Give a detailed answer to the following question. Question: {query}"""
#     else:
#         prompt = f"""Using the information contained in the context, give a detailed answer to the question.
# Context: {context}.
# Question: {query}"""
        messages = []
        if context:
            messages.append(SystemMessage(content=f"""You are a helpful travelling assistant. You suggest places to users based on their budget and interest. Keep in
            mind the following points while giving a response
            - Only answer questions related to travel recommendations
            - Do not answer any question which does not involve travel related queries
            - Do not answer questions which can start a small talk
            - If you do not understand the context, simply ask the user to ask the question again and do not add any information
            - Only answer in English language
            - Do not suggest a single place all the time, be creative in suggesting places
            - Only use facts provided in the context and do not make things on your own
            - Do not use any special symbol in the response like $
            Context: {context}"""))
        messages.append(HumanMessage(content=f"Question: {query}"))
        llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0)
        print(messages)
        response = llm.invoke(messages)

        return response
    
def rag_query(query, faiss_db):
    context = retrieve_documents(query, faiss_db)
    response = generate_response_with_groq(query, context)
    return response.content