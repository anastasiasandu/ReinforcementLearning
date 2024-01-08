#using  KMeans clustering
import os
import pandas as pd
from sklearn.cluster import KMeans

def read_movie_data(folder_path):
    movie_file_path = os.path.join(folder_path, 'movies.dat')
    movies = pd.read_csv(movie_file_path, sep='::', engine='python', header=None, names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
    return movies

def read_rating_data(folder_path):
    rating_file_path = os.path.join(folder_path, 'ratings.dat')
    ratings = pd.read_csv(rating_file_path, sep='::', engine='python', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='latin-1')
    return ratings

def read_user_data(folder_path):
    user_file_path = os.path.join(folder_path, 'users.dat')
    users = pd.read_csv(user_file_path, sep='::', engine='python', header=None, names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='latin-1')
    return users

def recommend_movies(user_id, model, ratings_df, movies_df, top_n=5):
    user_rated_movies = ratings_df[ratings_df['UserID'] == user_id]['MovieID']
    unrated_movies = movies_df[~movies_df['MovieID'].isin(user_rated_movies)]

    user_data = ratings_df[ratings_df['UserID'] == user_id][['MovieID', 'Rating']]

    model.fit(user_data[['MovieID']])

    cluster_labels = model.predict(unrated_movies[['MovieID']])

    unrated_movies = unrated_movies.copy()
    unrated_movies['Cluster'] = cluster_labels

    cluster_counts = unrated_movies['Cluster'].value_counts()
    top_clusters = cluster_counts.sort_values(ascending=False).head(top_n).index

    recommendations = unrated_movies[unrated_movies['Cluster'].isin(top_clusters)].copy()

    recommended_movie_names = recommendations['Title'].tolist()[:top_n]  # Extracting only top N movie names

    return recommended_movie_names

if __name__ == "__main__":
    folder_path = "ml-1m"  # Change this to the correct folder path

    movies = read_movie_data(folder_path)
    ratings = read_rating_data(folder_path)
    users = read_user_data(folder_path)

    merged_data = pd.merge(ratings, movies, on='MovieID')
    users = pd.merge(users, merged_data, on='UserID')
    users['Genres'] = users['Genres'].str.split('|')

    model = KMeans(n_clusters=5, random_state=42, n_init=10)

    user_id_to_recommend = int(input("Enter User ID: "))
    recommended_movies = recommend_movies(user_id_to_recommend, model, ratings, movies, top_n=5)
    print(f'\nTop Recommendations for User {user_id_to_recommend}:')
    print(recommended_movies)
