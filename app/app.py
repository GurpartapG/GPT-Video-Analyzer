from flask import Flask, render_template, request, jsonify
import openai
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import whisper
from pytube import YouTube
from pytube.exceptions import VideoUnavailable
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__, template_folder='../templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.json
    video_url = data.get('video_url')

    # Extract the YouTube video ID from the URL
    video_id = extract_video_id(video_url)

    try:
        # Try to fetch the transcript from YouTube
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([t['text'] for t in transcript])
    except TranscriptsDisabled:
        print(f"Transcripts are disabled for the video: {video_url}")
        transcript_text = transcribe_audio(video_url)
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'No transcript available and unable to process video.'}), 500
    
    return jsonify({'transcript': transcript_text})

@app.route('/ask_gpt', methods=['POST'])
def ask_gpt():
    data = request.json
    prompt = data.get('prompt')
    transcript = data.get('transcript')

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"The video content is:\n{transcript}\n\nUser's question: {prompt}",
        max_tokens=150,
        temperature=0.7,
    )
    answer = response.choices[0].text.strip()
    
    return jsonify({'response': answer})

def extract_video_id(url):
    # Extract the video ID from a YouTube URL
    if 'v=' in url:
        return url.split('v=')[-1]
    return url.split('/')[-1]

def transcribe_audio(video_url):
    try:
        # Download the YouTube video using Pytube
        yt = YouTube(video_url)
        stream = yt.streams.filter(only_audio=True).first()
        audio_file = stream.download(filename='audio.mp4')
        
        # Use Whisper to transcribe the audio
        model = whisper.load_model('base')
        result = model.transcribe(audio_file)
        return result['text']
    
    except VideoUnavailable:
        print(f"Video unavailable: {video_url}")
        return "This video is unavailable or restricted. Please try another video."

    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while processing the video."

if __name__ == "__main__":
    app.run(debug=True)
