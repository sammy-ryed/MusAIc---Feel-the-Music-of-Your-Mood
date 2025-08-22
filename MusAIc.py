# pip install pywebview spotipy transformers torch

import webview
import random
import webbrowser
from transformers import pipeline
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ---- Spotify API auth ----
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="ENTER YOUR CLIENT ID",
    client_secret="ENTER YOUR CLIENT SECRET",
    redirect_uri="ENTER REDIRECT URL",
    scope="user-read-playback-state,user-modify-playback-state,user-read-currently-playing,user-read-playback-position,user-library-read,user-top-read,playlist-read-private"
))

# ---- Hugging Face pipelines ----
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
generator = pipeline("text-generation", model="gpt2")

# ---- Mood → base query map ----
mood_map = {
    "joy": "happy upbeat pop",
    "sadness": "emotional sad acoustic",
    "anger": "heavy rock metal",
    "fear": "calm piano instrumental",
    "neutral": "chill lofi hip hop",
    "surprise": "energetic edm",
    "disgust": "dark experimental",
}

# ---- Spotify helpers ----
def get_playlist_tracks(limit_playlists=20, limit_tracks_per_playlist=50):
    playlist_tracks = []
    try:
        playlists = sp.current_user_playlists(limit=limit_playlists)["items"]
        for pl in playlists:
            tracks = sp.playlist_items(pl["id"], limit=limit_tracks_per_playlist)["items"]
            for t in tracks:
                track = t.get("track")
                if track:
                    playlist_tracks.append(track["name"])
    except Exception as e:
        print(f"⚠️ Could not fetch playlist tracks: {e}")
    return playlist_tracks[:100]

def get_user_top_tracks_and_artists(limit=5):
    try:
        top_tracks = sp.current_user_top_tracks(limit=limit, time_range="medium_term")["items"]
        top_artists = [t["artists"][0]["name"] for t in top_tracks]
        top_track_names = [t["name"] for t in top_tracks]
        return top_artists, top_track_names
    except:
        return [], []

def build_final_query(mood, user_artists, user_tracks, playlist_tracks):
    base_query = mood_map.get(mood, "chill music")

    prompt = f"Suggest one music style or artist for someone feeling {mood}."
    ai_output = generator(prompt, max_new_tokens=15, truncation=True,
                          num_return_sequences=1, do_sample=True,
                          top_k=50, top_p=0.9)[0]["generated_text"]
    ai_query = ai_output.replace(prompt, "").strip().split("\n")[0]
    ai_query = " ".join(ai_query.split()[:5])

    # 🎵 Added "mood_only" mode
    mode = random.choices(
        ["playlist", "top_artists", "top_tracks", "all_combined", "mood_only"],
        weights=[0.25, 0.3, 0.1, 0.4, 0.69],  # adjust as you like
        k=1
    )[0]

    if mode == "playlist":
        final_query = f"{base_query} {ai_query} {' '.join(playlist_tracks)}"[:240]
    elif mode == "top_artists":
        final_query = f"{base_query} {ai_query} {' '.join(user_artists)}"[:240]
    elif mode == "top_tracks":
        final_query = f"{base_query} {ai_query} {' '.join(user_tracks)}"[:240]
    elif mode == "all_combined":
        final_query = f"{base_query} {ai_query} {' '.join(playlist_tracks)} {' '.join(user_artists)} {' '.join(user_tracks)}"[:240]
    else:  # mood_only
        final_query = base_query  # ✅ just mood music

    return final_query


# ---- JS ↔ Python bridge ----
import cv2
import time
import numpy as np
from transformers import pipeline
import base64

# --- Load face detection & emotion recognition ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
emotion_classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

class API:
    def analyze_mood(self, text):
        result = classifier(text)[0]
        mood = result["label"].lower()

        playlist_tracks = get_playlist_tracks()
        user_artists, user_tracks = get_user_top_tracks_and_artists()

        final_query = build_final_query(mood, user_artists, user_tracks, playlist_tracks)
        tracks = sp.search(q=final_query, type="track", limit=5)["tracks"]["items"]

        if tracks:
            chosen = random.choice(tracks)
            song_name = chosen["name"]
            artist = chosen["artists"][0]["name"]
            url = chosen["external_urls"]["spotify"]
            uri = chosen["uri"]

            devices = sp.devices()
            if devices["devices"]:
                device_id = devices["devices"][0]["id"]
                sp.start_playback(device_id=device_id, uris=[uri])

            return {
                "mood": mood,
                "confidence": round(result["score"], 2),
                "query": final_query,
                "song": f"{song_name} - {artist}",
                "url": url
            }
        else:
            return {"error": "No matching track found."}

    def facial_recognition(self, image_data):
        # Decode base64 → image
        img_data = base64.b64decode(image_data.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Could not decode image"}

        # Detect face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # improve detection: scaleFactor smaller, minNeighbors lower
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,   # detect smaller faces
            minNeighbors=3,    # less strict
            minSize=(60, 60)   # ignore tiny noise
        )


        if len(faces) == 0:
            cv2.imwrite("debug_snapshot.jpg", frame)
            print(f"Saved snapshot: {frame.shape}")

            return {"error": "No face detected."}

        # Pick first face
        (x, y, w, h) = faces[0]
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # ✅ Convert to PIL
        from PIL import Image
        face_pil = Image.fromarray(face_rgb)

        # Predict
        preds = emotion_classifier(face_pil)
        mood = preds[0]["label"].lower()


        # Get Spotify recommendations
        playlist_tracks = get_playlist_tracks()
        user_artists, user_tracks = get_user_top_tracks_and_artists()
        final_query = build_final_query(mood, user_artists, user_tracks, playlist_tracks)
        tracks = sp.search(q=final_query, type="track", limit=5)["tracks"]["items"]

        if not tracks:
            return {"error": "No track found."}

        chosen = random.choice(tracks)
        song_name = chosen["name"]
        artist = chosen["artists"][0]["name"]
        url = chosen["external_urls"]["spotify"]
        uri = chosen["uri"]

        devices = sp.devices()
        if devices["devices"]:
            device_id = devices["devices"][0]["id"]
            sp.start_playback(device_id=device_id, uris=[uri])

        return {
            "mood": mood,
            "query": final_query,
            "song": f"{song_name} - {artist}",
            "url": url
        }


# ---- Launch Webview ----
if __name__ == "__main__":
    api = API()
    window = webview.create_window(
        "MusAIc - Feel the Music of Your Mood",
        url="index.html",
        js_api=api
    )

    # When window is closed, exit Python
    def on_closed():
        import sys
        sys.exit(0)

    window.events.closed += on_closed

    webview.start(debug=True)