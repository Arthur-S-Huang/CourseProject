# Spotify Song Recommender with Mood Filter- Project Overview

This project provides an executable script to recommend some songs based on an initial user profile. The user profile is required to be on Spotify and this initial profile needs to have a playlist. The main challenge was to come up with features used to train a model for recommendations. The Nearest Neighbor model from scikit-learn will help with this process. In addition, this project explores a novel idea to filter/recommend songs based on a given mood. The initial recommendation list is based on the user's profile, but the user can actively filter the list based on a given mood. Sentiment analysis will be done on the list of song names.


## Installation of Libraries

Python is a core prerequisite. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraries (follow the linked page to install pip itself if not already installed). [Numpy](https://numpy.org/install/), [Pandas](https://pandas.pydata.org/docs/getting_started/install.html), [NLTK](https://www.nltk.org/api/nltk.sentiment.sentiment_analyzer.html), [scikit-learn](https://scikit-learn.org/stable/modules/neighbors.html), and [spotipy](https://spotipy.readthedocs.io/en/2.21.0/) were used to implement this project. In addition, follow [this page](https://developer.spotify.com/dashboard/login) to set up the Spotify API keys.

```bash
pip install numpy
```
```bash
pip install nltk
python -m nltk.downloader vader_lexicon
```
```bash
pip install pandas
```
```bash
pip install -U scikit-learn
```

```bash
pip install spotipy --upgrade
```

## General code structure
This section will explain how core functions/components work. 
```python
import libraries...

# process data, drop duplicate songs and reset index
cleaned_data = process_data()

# train a model for recommendation
# extract audio features such as tempo and energy
# use cosine similarity
nearest_neighbor_model, scaler, data = train_nearest_neighbor_model(cleaned_data)

# get user profile data
song_ids, sp = get_song_ids(username=spotify_username, client_id=client_id,
                                client_secret=client_secret, playlist_id=playlist_id)
# generate user playlist features
user_profile = get_user_profile(song_ids, sp)

# make recommendations
recommend_songs(user_profile, nearest_neighbor_model, scaler, data)
```

## Example Usage
In a terminal window, navigate to the root of the project folder. Run the following command, whenever a prompt shows up, type in the information required, such as Spotify username and client id.

```bash
python3 recommender.py
```

## References
[Recommender tutorial by ugis22](https://github.com/ugis22/music_recommender/blob/master/content%20based%20recommedation%20system/content_based_music_recommender.ipynb)

[Recommender tutorial by madhavthaker](https://github.com/madhavthaker/spotify-recommendation-system/blob/main/spotify-recommendation-engine.ipynb)

[Million Song Dataset](http://millionsongdataset.com/)

