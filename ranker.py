import pandas as pd
import numpy as np
import nltk
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

data_root = "data/processed_data.csv"

def process_data():
    songs_df = pd.read_csv(data_root)
    #print(playlistDF.head())
    songs_df_processed = drop_duplicate_songs(songs_df)
    if len(pd.unique(songs_df_processed.artists_song)) != len(songs_df_processed):
        raise Exception("Data contains duplicate songs!")
    
    #print(songs_df_processed.head())
    return songs_df_processed

def drop_duplicate_songs(df):
    # Drop duplicate songs
    df['artists_song'] = df.apply(lambda row: row['artist_name'] + row['track_name'], axis = 1)
    # Reset index after drop
    return df.drop_duplicates('artists_song').reset_index()

def train_nearest_neighbor_model(train_data):
    # Train a nearest neighbor model to recommend songs
    # This approach uses content-based filtering for recommendation

    features = train_data[['danceability', 'energy',
                           'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                           'liveness', 'valence', 'tempo']]
    # Need to use MinMax Scaler to normalize train and test data
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Cosine similarity used to compare vectors/song features
    model = NearestNeighbors(metric="cosine").fit(scaled_features)

    return model, scaler, train_data

def recommend_songs(user_profile, model, scaler, train_data):
    # Normalize test data from the given user profile
    unseen_scaled_features = scaler.transform(user_profile)

    indices = []
    rec_song_names = []
    for i in range(unseen_scaled_features.shape[0]):
        # find the nearest neighbors of the i-th test point
        distances, neighbors = model.kneighbors(unseen_scaled_features[i].reshape(1, -1))
        # indices of the nearest neighbors
        indices.append(neighbors[0])
    
    # Store index of nearest neighbor of current song on user playlist
    indices = [x[0] for x in indices]

    # Display output
    for idx in indices:
        rec_song_names.append((train_data.loc[idx])['track_name'])
        print("{} by {}".format((train_data.loc[idx])['track_name'], (train_data.loc[idx])['artist_name']))
    
    return rec_song_names

def recs_based_on_mood(list_song_names, mood):
    sid = SentimentIntensityAnalyzer()
    songs_based_on_mood = []
    for song in list_song_names:
        # Four categories of sentiments
        # 'neg', 'neu', 'pos', 'compound'
        scores = sid.polarity_scores(song)

        if (mood == "neg") and (scores['neg'] > scores['neu']) and (scores['neg'] > scores['pos']):
            songs_based_on_mood.append(song)
        elif (mood == "neu") and (scores['neu'] > scores['neg']) and (scores['neu'] > scores['pos']):
            songs_based_on_mood.append(song)
        elif (mood == "pos") and (scores['pos'] > scores['neu']) and (scores['pos'] > scores['neg']):
            songs_based_on_mood.append(song)
    return songs_based_on_mood
