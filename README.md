# Travel RAG Assistant

A Retrieval-Augmented Generation (RAG) system for travel recommendations, powered by audio processing and LLMs.


## Features

- Upload and transcribe travel audio recordings
- Semantic search powered by FAISS vector database
- Natural language Q&A with Groq's LLM (Llama3-70B)
- Persistent document storage and indexing
- Modern Streamlit UI with responsive design

## Tech Stack

- **Backend**: Python 3.10+
- **NLP**: HuggingFace Transformers, LangChain
- **Vector DB**: FAISS
- **LLM**: Groq API (Llama3-70B)
- **UI**: Streamlit

## Installation

1. Clone repository:
```bash
   git clone https://github.com/ViceMarauder/Audio-Transcript-Chatbot.git
   cd Audio-Transcript-Chatbot
```

2. Create virtual environment:
```bash
   python -m venv myvenv
   source myvenv/bin/activate  # Windows: myvenv\Scripts\activate
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Set up environment:
```bash
   echo -e "GROQ_API_KEY=your_api_key_here\nHF_TOKEN=your-huggingface-key" > .env
```

## Folder Structure
```bash
.
├── src/
│   ├── app.py                # Main Streamlit app
│   ├── rag.py                # RAG pipeline
│   ├── faiss_store.py        # Vector DB operations
│   ├── model.py              # LLM interface
│   ├── transcriptions.py     # Audio processing
│   ├── download_audio.py     # Download mp3 from youtube links
│   └── transcribe_audio.py   # Creates the transcript folder from the audio folder
├── data/
│   └── faiss_index/          # Vector DB storage
├── transcript/               # Processed text files
├── models/                   # Local model cache
├── .env                      # Environment variables
├── sample_file/              # Contains a sample file to test the transcription model
└── audio/                    # Contains the mp3 files
```

## Configuration

1. Get Groq API key from [console.groq.com](https://console.groq.com/)

2. Create .env file:
```env
   GROQ_API_KEY=<your-groq-key>
   HF_TOKEN=<your-huggingface-key>
```
## Usage

Start the application:
```bash
   streamlit run src/app.py
```

### Workflow:

1. Upload MP3 audio files (left panel)
2. Ask travel-related questions (chat interface)
3. Get AI-powered recommendations with sources