import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Spotify App Credentials
CLIENT_ID = "a761e971e99b4da892e0a5580b6e4a4a"
CLIENT_SECRET = "2c9c88af535d47ccb831c64aa2719601"
REDIRECT_URI = "http://127.0.0.1:8888/callback"

# Scopes allow control over playback
SCOPE = "user-read-playback-state user-modify-playback-state user-read-currently-playing"

# Create Spotipy client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE
))

# Test: Print current playback info
current = sp.current_playback()
if current and current['is_playing']:
    print("Now playing:", current['item']['name'], "by", current['item']['artists'][0]['name'])
else:
    print("Nothing is playing.")
