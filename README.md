# MusAIc

Well, spotify's recommendations sucks. So I thought why not make it better:
MusAIc is an AI-powered music application that bridges emotion recognition with personalized music recommendations.
The app detects your mood in real time using facial expressions and then plays songs that match your emotions through the Spotify API.

## Demo
https://github.com/user-attachments/assets/5c7e5180-4230-4a4e-be4f-ce029998b3bc

## How to use MusAIc

#### Install Required Python Packages

Open your terminal and run:
```
pip install pywebview spotipy transformers torch opencv-python pillow
```

### Set Up Spotify Developer App

1. Go to Spotify [Developer Dashboard](https://developer.spotify.com/dashboard).

2. Log in with your Spotify account.

3. Click “Create an App”.

4. Copy your Client ID and Client Secret.

5. Add a Redirect URI → this must match your ngrok link (see next step).

### Set Up ngrok

1. We need ngrok to expose a local redirect URI for Spotify authentication.

2. Download [ngrok](https://ngrok.com/download).

3. Run ngrok in your terminal:
```
ngrok http 5000
```
4. Copy the generated link, e.g.:
```
https://12143e131154.ngrok-free.app/
```

5. Add this same link to your Spotify Developer app Redirect URIs.

### Configure Spotify Authentication

6. In your Python code, authenticate with Spotify (Replace Client ID, Secret and redirect URL):
```
from spotipy.oauth2 import SpotifyOAuth
import spotipy

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    redirect_uri="YOUR_NGROK_URI",
    scope="user-read-playback-state,user-modify-playback-state,user-read-currently-playing"
))
```
### Mood Detection Pipelines

MusAIc uses Hugging Face Transformers for emotion detection:
```
from transformers import pipeline

# Text emotion classification
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Facial emotion detection
emotion_classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
```
### Run the App

MusAIc launches a webview interface:
```
import webview

window = webview.create_window(
    "MusAIc - Feel the Music of Your Mood",
    url="index.html",
    js_api=api   # your Python ↔ JS bridge
)

webview.start(debug=True)
```

## How it works
The app detects your mood (via text input or webcam).

It builds a Spotify search query from:

- Your mood → "happy upbeat pop", "sad acoustic", etc.

- Your Spotify playlists & top artists.

- AI-generated enhancements (via GPT-2).

- It plays a recommended track directly in your Spotify app.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Known Issues
- Setup is a bit of a pain right now (requires ngrok, Spotify dev app, and several API keys).  
- Queries can take a long time to run because of multiple API calls + AI model inference.  
- Requires a Spotify Premium account for playback control.  
- Ngrok free version may change URL each restart (you’ll need to update redirect URI).  
- Emotion detection can be inaccurate in low light or with multiple faces.  


## Credits

Frountend: Ishaan Verma\
Backend: Samarth Ryan Edward
