from ranker import process_data, train_nearest_neighbor_model, recommend_songs, recs_based_on_mood
from profile_initializer import get_song_ids, get_user_profile

def main():
    print("Enter your Spotify Username: ")
    spotify_username = str(input())

    print()
    print("Enter your Spotify Client ID: ")
    client_id = str(input())
    
    print()
    print("Enter your Spotify Client Secret: ")
    client_secret = str(input())

    cleaned_data = process_data()
    # nearest neighbor model using cosine similarity
    nearest_neighbor_model, scaler, data = train_nearest_neighbor_model(cleaned_data)

    # Initialize user profile used for recommendation
    print()
    print("Enter your the playlist URL: ")
    playlist_id = str(input())

    song_ids, sp = get_song_ids(username=spotify_username, client_id=client_id,
                                client_secret=client_secret, playlist_id=playlist_id)

    user_profile = get_user_profile(song_ids, sp)
    
    # Now that we have trained a model and have some initial user profile/preferecesm
    # we can recommend some songs
    print()
    print("Here are your recommended songs:")
    print()
    recs = recommend_songs(user_profile, nearest_neighbor_model, scaler, data)
    print()

    print()
    print("Enter a mood as a filter (type neu, pos, or neg): ")
    mood = str(input())

    rec_songs_based_on_mood = recs_based_on_mood(recs, mood)
    print()
    print("Recommended songs based on mood: ", rec_songs_based_on_mood)

if __name__ == "__main__":
    main()
    