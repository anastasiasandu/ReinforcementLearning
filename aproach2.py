import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

np.random.seed(42)

def read_movie_data(folder_path):
    movie_file_path = os.path.join(folder_path, 'movies.dat')
    movies = pd.read_csv(movie_file_path, sep='::', engine='python', header=None, names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
    return movies

def read_rating_data(folder_path):
    rating_file_path = os.path.join(folder_path, 'ratings.dat')
    ratings = pd.read_csv(rating_file_path, sep='::', engine='python', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='latin-1')
    return ratings

class SARSAMovieRecommender:
    def __init__(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings
        self.Q_values = {}

    def get_user_history(self, user_id):
        user_ratings = self.ratings[self.ratings['UserID'] == user_id][['MovieID', 'Rating']]
        return user_ratings

    def update_Q_values(self, state, action, reward, next_state, next_action, alpha, gamma):
        current_Q = self.Q_values.get((state, action), 0)
        next_Q = self.Q_values.get((next_state, next_action), 0)
        updated_Q = current_Q + alpha * (reward + gamma * next_Q - current_Q)
        self.Q_values[(state, action)] = updated_Q

    def select_action(self, state, epsilon):
        np.random.seed(42)
        if np.random.rand() < epsilon:
            return np.random.choice(self.movies['MovieID'])
        else:
            possible_actions = self.movies['MovieID'].tolist()
            return np.random.choice(possible_actions, size=1)[0]

    def recommend_movies_sarsa(self, user_id, top_n=5, alpha=0.1, gamma=0.9, epsilon=0.1):
        user_history = self.get_user_history(user_id)

        for i in range(len(user_history) - 1):
            state = user_history.iloc[i]['MovieID']
            action = user_history.iloc[i + 1]['MovieID']
            reward = user_history.iloc[i + 1]['Rating']
            next_state = user_history.iloc[i + 1]['MovieID']

            next_action = self.select_action(next_state, epsilon)
            self.update_Q_values(state, action, reward, next_state, next_action, alpha, gamma)

        current_state = user_history.iloc[-1]['MovieID']
        recommended_movie_ids = [self.select_action(current_state, epsilon) for _ in range(top_n)]
        recommended_movie_names = self.movies[self.movies['MovieID'].isin(recommended_movie_ids)]['Title'].tolist()

        return recommended_movie_names

if __name__ == "__main__":
    folder_path = "ml-1m"

    movies = read_movie_data(folder_path)
    ratings = read_rating_data(folder_path)

    sarsa_recommender = SARSAMovieRecommender(movies, ratings)

    user_id_to_recommend = int(input("Enter User ID: "))
    recommended_movies = sarsa_recommender.recommend_movies_sarsa(user_id_to_recommend, top_n=5)
    print(f'\nTop Recommendations for User {user_id_to_recommend}:')
    print(recommended_movies)
