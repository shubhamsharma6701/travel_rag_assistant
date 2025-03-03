import yt_dlp
import os

def download_youtube_audio(url, output_dir="../audio"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the options for downloading the audio
    ydl_opts = {
        'format': 'bestaudio/best',  # Downloads the best audio quality
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',  # Converts audio to mp3 format
            'preferredcodec': 'mp3',  # Convert to mp3
            'preferredquality': '192',  # Set the quality
        }],
        'outtmpl': os.path.join(output_dir,'%(title)s.%(ext)s'),  # Save with the video title as the filename
        'nocheckcertificate': True,  # Disable SSL certificate verification
        # 'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0',  # Custom User-Agent string
        'verbose': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0'
        }
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
        return filename


# URL of the YouTube video
# video_url = 'https://youtu.be/kXYiU_JCYtU?si=AI9Uy1RKp3ZF9l7E'

# Download the audios
# with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#     ydl.download([video_url])