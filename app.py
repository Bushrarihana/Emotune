from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import os
import sys
import platform
import threading
from tensorflow.keras.models import load_model
from music_player import play_music_by_emotion
from voice_emotion import detect_voice_emotion as detect_voice_emotion_fn
import requests

app = Flask(__name__)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "1b969085c1299779644dc57569e7445c")
base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))

# --- Load model ---
model_path = os.path.join(base_path, "emotion_model.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")
model = load_model(model_path, compile=False)

emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

# --- State ---
detected_emotion = None
emotion_locked   = False
face_score       = None
voice_score      = None
final_score      = None

# --- Camera control ---
camera_active = False
camera_lock   = threading.Lock()
cap           = None   # global VideoCapture


# ── Start camera ──────────────────────────────────────────
@app.route("/start_camera")
def start_camera():
    global camera_active, cap, detected_emotion, emotion_locked, face_score
    with camera_lock:
        if not camera_active:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return jsonify({"status": "error", "message": "Cannot open camera"}), 500
            detected_emotion = None
            emotion_locked   = False
            face_score       = None
            camera_active    = True
            print("📷 Camera started")
    return jsonify({"status": "started"})


# ── Stop camera ───────────────────────────────────────────
@app.route("/stop_camera")
def stop_camera():
    global camera_active, cap
    with camera_lock:
        camera_active = False
        if cap and cap.isOpened():
            cap.release()
            cap = None
            print("🔒 Camera stopped")
    return jsonify({"status": "stopped"})


# ── Camera status ─────────────────────────────────────────
@app.route("/camera_status")
def camera_status():
    return jsonify({"active": camera_active})


# ── Frame generator ───────────────────────────────────────
def gen_frames():
    global detected_emotion, emotion_locked, face_score, cap

    while camera_active:
        with camera_lock:
            if cap is None or not cap.isOpened():
                break
            success, frame = cap.read()

        if not success:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))

        if not emotion_locked:
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = np.reshape(roi, (1, 48, 48, 1))

                preds   = model.predict(roi, verbose=0)[0]
                emotion = emotion_labels[np.argmax(preds)]
                score   = round(float(np.max(preds) * 100), 2)

                detected_emotion = emotion
                face_score       = score

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 230, 255), 2)
                cv2.putText(frame, f"{emotion.upper()} {score:.0f}%", (x, y-12),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 230, 255), 2)

                print(f"✅ Face Emotion: {emotion} | Score: {score}%")
                emotion_locked = True
                break
        else:
            if detected_emotion:
                cv2.putText(frame, f"{detected_emotion.upper()}", (20, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 230, 255), 3)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 230, 255), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    if not camera_active:
        return Response(status=204)
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Weather ───────────────────────────────────────────────
@app.route("/detect_weather_emotion")
def detect_weather_emotion():
    global detected_emotion, emotion_locked, final_score

    lat  = request.args.get("lat")
    lon  = request.args.get("lon")
    city = request.args.get("city")

    if lat and lon:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    elif city:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    else:
        return jsonify({"error": "No location provided"}), 400

    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        weather_main = data["weather"][0]["main"].lower()
        weather_desc = data["weather"][0]["description"]          # keep original capitalisation
        icon_code    = data["weather"][0].get("icon", "01d")
        temp_c       = data.get("main", {}).get("temp")
        feels_like   = data.get("main", {}).get("feels_like")
        humidity     = data.get("main", {}).get("humidity")
        wind_speed   = data.get("wind", {}).get("speed")          # m/s
        city_name    = data.get("name", "")
        country      = data.get("sys", {}).get("country", "")

        weather_to_emotion = {
            "clear":        "happy",
            "clouds":       "neutral",
            "rain":         "sad",
            "drizzle":      "sad",
            "thunderstorm": "fear",
            "snow":         "surprise",
            "mist":         "neutral",
            "smoke":        "neutral",
            "haze":         "neutral",
            "fog":          "neutral",
            "dust":         "disgust",
            "sand":         "disgust",
            "ash":          "fear",
            "squall":       "fear",
            "tornado":      "fear",
        }
        detected         = weather_to_emotion.get(weather_main, "neutral")
        final_score      = 100
        detected_emotion = detected
        emotion_locked   = True

        return jsonify({
            "weather":      weather_main,
            "description":  weather_desc,
            "icon_code":    icon_code,
            "temp_c":       temp_c,
            "feels_like":   feels_like,
            "humidity":     humidity,
            "wind_speed":   wind_speed,
            "city":         city_name,
            "country":      country,
            "emotion":      detected,
            "final_score":  final_score
        })
    except requests.RequestException as e:
        return jsonify({"error": f"Weather API failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# ── Core routes ───────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", python_version=platform.python_version())


@app.route("/get_emotion")
def get_emotion():
    return jsonify({
        "emotion":     detected_emotion,
        "final_score": final_score,
        "face_score":  face_score,
        "voice_score": voice_score
    })


@app.route("/reset_emotion")
def reset_emotion():
    global detected_emotion, emotion_locked, face_score, voice_score, final_score
    detected_emotion = None
    emotion_locked   = False
    face_score       = None
    voice_score      = None
    final_score      = None
    return jsonify({"status": "reset"})


@app.route("/play_song")
def play_song():
    if detected_emotion:
        play_music_by_emotion(detected_emotion)
        return jsonify({"status": "playing"})
    return jsonify({"status": "no_emotion"})


@app.route("/detect_voice_emotion")
def detect_voice_emotion_route():
    global detected_emotion, emotion_locked, voice_score

    result = detect_voice_emotion_fn()

    if isinstance(result, dict):
        detected_emotion = result.get("emotion")
        voice_score      = result.get("score")
    elif isinstance(result, tuple):
        detected_emotion = result[0]
        if len(result) > 1:
            voice_score = result[1]

    emotion_locked = True
    return jsonify({
        "emotion":     detected_emotion,
        "voice_score": voice_score,
        "final_score": final_score
    })


if __name__ == "__main__":
    print("✅ Python version:", platform.python_version())
    print("✅ App running at http://127.0.0.1:5000")
    app.run(debug=True)