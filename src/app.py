import os
import streamlit as st
import torch
from transcriptions import transcribe_new_audio_files
from langchain.text_splitter import RecursiveCharacterTextSplitter
from faiss_store import store_documents_in_faiss, load_faiss_index
from rag import rag_query, Encoder, FaissDb, load_and_split_txt_files
from langchain.docstore.document import Document
from download_audio import download_youtube_audio

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

st.set_page_config(
    page_title="Travel AI Assistant",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_encoder():
    encoder = Encoder(
        model_name="sentence-transformers/all-MiniLM-L12-v2", device="cpu"
    )
    return encoder

@st.cache_resource
def initialize_faiss(_encoder):
    faiss_index_path = os.path.join("..", "data", "faiss_index")
    
    if not os.path.exists(faiss_index_path):
        os.makedirs(faiss_index_path, exist_ok=True)
        initial_docs = load_and_split_txt_files("..//transcript")
        vector_store = store_documents_in_faiss(initial_docs, faiss_index_path, encoder.embedding_function)
    else:
        vector_store = load_faiss_index(faiss_index_path, encoder.embedding_function)
    
    return FaissDb(embedding_function=encoder.embedding_function, db=vector_store)

# def load_existing_transcripts():
#     documents = []
#     transcript_folder = os.path.join("..", "transcript")
#     for filename in os.listdir(transcript_folder):
#         if filename.endswith('.txt'):
#             file_path = os.path.join(transcript_folder, filename)
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 content = file.read()
#                 # Create a LangChain Document with content and metadata
#                 doc = Document(page_content=content, metadata={"source": filename})
#                 documents.append(doc)
#     return documents

def process_uploaded_files(uploaded_files, _encoder):
    if uploaded_files:
        with st.status("Processing files...", expanded=True) as status:
            st.write("🔊 Transcribing audio content...")
            new_transcripts = transcribe_new_audio_files(uploaded_files)
            st.write("📚 Updating knowledge base...")
            # Convert new transcripts to LangChain Documents
            new_documents = [Document(page_content=transcript, metadata={"source": "uploaded"}) for     transcript in new_transcripts]
            # all_documents = load_existing_transcripts() + new_documents
            # embeddings = embed_transcripts(all_documents)  # Assuming embed_transcripts can handle    LangChain Documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=256,
                chunk_overlap=int(256 // 10),
                strip_whitespace=True,
            )
            split_docs = text_splitter.split_documents(new_documents)
            faiss_index_path = os.path.join("..", "data", "faiss_index")
            faiss_index = initialize_faiss(_encoder)
            faiss_index.db.add_documents(split_docs)
            faiss_index.db.save_local(faiss_index_path)
            status.update(label="✅ Processing complete!", state="complete", expanded=False)
    
encoder = load_encoder()
faiss_db = initialize_faiss(encoder)

st.title("Travel RAG System")

if 'file_uploader_key' not in st.session_state:
    st.session_state['file_uploader_key'] = 0

with st.sidebar:
    st.header("📁 Audio Processing")
    with st.form("audio_upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload Audio Files", accept_multiple_files=True, type=["mp3"], help="Upload audio files to transcribe and add to knowledge base")
        submitted = st.form_submit_button("Upload")
        if submitted:
            process_uploaded_files(uploaded_files, encoder)
            st.session_state['file_uploader_key'] += 1
            st.rerun()
    # uploaded_files = st.file_uploader("Upload Audio Files", accept_multiple_files=True, type=["mp3"], help="Upload audio files to transcribe and add to knowledge base")
    # process_uploaded_files(uploaded_files,encoder)
    # if uploaded_files:
    #     st.session_state['file_uploader_key'] += 1
    #     st.rerun()

    st.header("🎵 YouTube Download")
    with st.form("youtube_form", clear_on_submit=True):
        youtube_url = st.text_input("YouTube Video URL", key="youtube_url")
        download_submitted = st.form_submit_button("Download Audio")
        if download_submitted and youtube_url:
            try:
                with st.status("Downloading audio...", expanded=True) as status:
                    st.write("⏬ Downloading audio from YouTube...")
                    downloaded_path = download_youtube_audio(youtube_url)
                    st.write("✅ Audio downloaded successfully!")
                    # Process the downloaded file
                    st.write("🔊 Transcribing audio...")
                    from transcriptions import process_existing_audio_file
                    split_docs = process_existing_audio_file(
                        downloaded_path,
                        transcripts_folder='../transcript',
                        temp_folder='../temp_audio',
                        chunk_folder='../chunk_audio'
                    )
                    st.write("📚 Updating knowledge base...")
                    faiss_db.db.add_documents(split_docs)
                    faiss_db.db.save_local(os.path.join("..", "data", "faiss_index"))
                    status.update(label="✅ Processing complete!", state="complete", expanded=False)
                    st.rerun()
            except Exception as e:
                st.error(f"Error downloading audio: {str(e)}")


st.header("🌍 Travel Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
       with st.chat_message(message["role"]):
           st.markdown(message["content"])
   
if prompt := st.chat_input("Ask me about travel destinations..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        user_prompt = st.session_state.messages[-1]["content"]
        # context = faiss_db.similarity_search(user_prompt)
        answer = rag_query(user_prompt, faiss_db)
        response = st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})