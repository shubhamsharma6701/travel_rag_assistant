import os
import base64
from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
import subprocess
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

load_dotenv()

groq = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

def audio_to_base64(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
        base64_audio = base64.b64encode(audio_bytes).decode()
    return base64_audio

def reencode_audio_to_ogg(input_file, output_file="encoded_audio.ogg"):
    command = [
        "C:\\Users\\eshub\\codes\\chatbot\\Lib\\site-packages\\imageio\\plugins\\ffmpeg.exe", "-y",  # Add the '-y' flag to overwrite without asking
        "-i", input_file, "-vn", "-map_metadata", "-1", 
        "-ac", "1", "-c:a", "libopus", "-b:a", "12k", "-application", "voip", output_file
    ]
    subprocess.run(command, check=True)

def split_audio(input_file, chunk_length_ms=60000):
    audio = AudioSegment.from_file(input_file)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

def transcribe_audio(audio_chunk, chunk_index, temp_folder):
    chunk_file = os.path.join(temp_folder, f"temp_chunk_{chunk_index}.ogg")
    audio_chunk.export(chunk_file, format="ogg")
    with open(chunk_file, "rb") as audio_file:
        transcript = groq.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            response_format="text"
        )
    return transcript

def transcribe_new_audio_files(audio_files, transcripts_folder='..//transcript', temp_folder='..//temp_audio', chunk_folder='..//chunk_audio',audio_folder='..//audio'):
    transcripts = []

    if not os.path.exists(transcripts_folder):
        os.makedirs(transcripts_folder)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    if not os.path.exists(chunk_folder):
        os.makedirs(chunk_folder)
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)

    for audio_file in audio_files:
        audio_path = f"{audio_folder}//{audio_file.name}"
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())

        encoded_file = os.path.join(temp_folder, f"{os.path.splitext(audio_file.name)[0]}_encoded.ogg")
        reencode_audio_to_ogg(audio_path, encoded_file)

        audio_chunks = split_audio(encoded_file)
        complete_transcript = ""

        for i, chunk in enumerate(audio_chunks):
            try:
                transcript = transcribe_audio(chunk, i, chunk_folder)
                complete_transcript += transcript + " "
            except Exception as e:
                print(f"Error transcribing chunk {i} of {audio_file.name}: {e}")

        txt_file = os.path.join(transcripts_folder, os.path.basename(audio_file.name).replace('.mp3', '.txt'))
        with open(txt_file, 'w', encoding='utf-8') as file:
            file.write(complete_transcript)
        transcripts.append(complete_transcript)

    return transcripts

def process_uploaded_files(uploaded_files):
    if uploaded_files:
        transcribe_new_audio_files(uploaded_files)
    else:
        print("No new files uploaded.")

def process_existing_audio_file(file_path, transcripts_folder, temp_folder, chunk_folder):
    # Process the audio file
    encoded_file = os.path.join(temp_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_encoded.ogg")
    reencode_audio_to_ogg(file_path, encoded_file)
    audio_chunks = split_audio(encoded_file)
    complete_transcript = ""
    
    for i, chunk in enumerate(audio_chunks):
        try:
            transcript = transcribe_audio(chunk, i, chunk_folder)
            complete_transcript += transcript + " "
        except Exception as e:
            print(f"Error transcribing chunk {i} of {file_path}: {e}")
    
    # Save transcript
    txt_file = os.path.join(transcripts_folder, os.path.basename(file_path).replace('.mp3', '.txt'))
    with open(txt_file, 'w', encoding='utf-8') as file:
        file.write(complete_transcript)
    
    # Split into documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=int(256 // 10),
        strip_whitespace=True,
    )
    
    split_docs = text_splitter.split_documents([
        Document(page_content=complete_transcript, metadata={"source": os.path.basename(file_path)})
    ])
    
    return split_docs