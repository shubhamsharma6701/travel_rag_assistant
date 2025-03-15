import os
import base64
from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
import subprocess

# Load environment variables
load_dotenv()

# Initialize OpenAI client for Groq API
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

def transcribe_audio(audio_chunk,chunk_index,temp_folder):
    chunk_file = os.path.join(temp_folder, f"temp_chunk_{chunk_index}.ogg")
    with open(chunk_file, "wb") as f:
        audio_chunk.export(f, format="ogg")
    with open(chunk_file, "rb") as audio_file:
        transcript = groq.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            response_format="text"
        )
    return transcript

def process_folder(input_folder, output_folder, temp_folder, chunk_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    if not os.path.exists(chunk_folder):
        os.makedirs(chunk_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mp3"):
            input_file = os.path.join(input_folder, file_name)
            encoded_file = os.path.join(temp_folder, f"{os.path.splitext(file_name)[0]}_encoded.ogg")
            reencode_audio_to_ogg(input_file, encoded_file)

            audio_chunks = split_audio(encoded_file)
            complete_transcript = ""

            for i, chunk in enumerate(audio_chunks):
                try:
                    transcript = transcribe_audio(chunk,i,chunk_folder)
                    complete_transcript += transcript + " "
                except Exception as e:
                    print(f"Error transcribing chunk {i} of {file_name}: {e}")
            
            print(f"Complete transcript for {file_name}: {complete_transcript}")
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.txt")
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write(complete_transcript)
            print(f"Transcript for {file_name} saved to {output_file_path}")

# input_file = "..\\audio\\ULTIMATE Vietnam Travel Guide 2025 - 14 Days in Vietnam - A Travel Documentary.mp3"
# reencode_audio_to_ogg(input_file, "encoded_audio.ogg")

# audio_chunks = split_audio("encoded_audio.ogg")
# complete_transcript = ""

# for i, chunk in enumerate(audio_chunks):
#     try:
#         transcript = transcribe_audio(chunk,i,".")
#         complete_transcript += transcript
#     except Exception as e:
#         print(f"Error transcribing chunk {i}: {e}")

input_folder = "../audio"
output_folder = "../transcript"
temp_folder = "../temp_audio"
chunk_folder = "../chunk_audio"
process_folder(input_folder, output_folder, temp_folder, chunk_folder)