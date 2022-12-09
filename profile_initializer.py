import spotipy
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
import time

def get_song_ids(username, client_id, client_secret, playlist_id):
    # Use the Spotify API to get song information
    client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    ids = []
    playlist = sp.user_playlist(username, playlist_id)
    for item in playlist['tracks']['items']:
        track = item['track']
        ids.append(track['id'])

    return ids, sp

def get_song_features(id, sp):
    features = sp.audio_features(id)

    # features
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    acousticness = features[0]['acousticness']
    instrumentalness = features[0]['instrumentalness']
    liveness = features[0]['liveness']
    valence = features[0]['valence']
    tempo = features[0]['tempo']
    
    track_feats = [danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo]
    return np.array(track_feats)

def get_user_profile(song_ids, sp):
    num_features = 9
    profile = np.zeros((len(song_ids), num_features))

    for idx, song_id in enumerate(song_ids):
        track_features = get_song_features(song_id, sp)
        profile[idx] = track_features
    
    return profile
    