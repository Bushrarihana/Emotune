# music_player.py
import webbrowser

emotion_to_song = {
    "happy": "https://open.spotify.com/playlist/1PUo37Kvq4eynwGPcVWwiN?si=wAUClmkKQVixQu8D_XzFCA",
    "sad": "https://open.spotify.com/playlist/0zPQzOlmVcUDfSgVEecOZQ?si=pbvNZTSXT9awr2S11Il1jw",
    "angry": "https://open.spotify.com/playlist/0nawsjYqKZKuXBIXFGTHyH?si=VhHGJ68cShy468PXT90APg&pi=YlSOjqOzQ5mL2",
    "neutral": "https://open.spotify.com/playlist/0BYneVGzIdUoixJ08XgvHg?si=mZN6VBFcTVqrCA8k95QWsg&pi=ShwQzEchRgmfg",
    "fear":"https://open.spotify.com/playlist/6RSBdnhJmWGaI1SKvdSLaR?si=kuC8-vozTru8-eQPeSvWJg",
    "surprise":"https://open.spotify.com/playlist/7jEjIc9ZAsH6aJZ8YAS5AA?si=7EbTwIZKSni37aSl9cCxAA"
}

def play_music_by_emotion(emotion):

    url = emotion_to_song.get(emotion)
    if url:
        print(f"🎵 Playing for emotion: {emotion}")
        webbrowser.open(url)
    else:
        print(f"No song found for emotion: {emotion}")
