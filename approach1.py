import os
import pandas as pd
import numpy as np


def read_movie_data(folder_path):
    movie_file_path = os.path.join(folder_path, 'movies.dat')
    movies = pd.read_csv(movie_file_path, sep='::', engine='python', header=None, names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
    return movies

def read_rating_data(folder_path):
    rating_file_path = os.path.join(folder_path, 'ratings.dat')
    ratings = pd.read_csv(rating_file_path, sep='::', engine='python', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='latin-1')
    return ratings

class TDMovieRecommender:
    def __init__(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings
        self.Q_values = {}

    def get_user_history(self, user_id):
        user_ratings = self.ratings[self.ratings['UserID'] == user_id][['MovieID', 'Rating']]
        return user_ratings

    def initialize_Q_values(self):
        all_movie_ids = self.movies['MovieID'].tolist()
        for movie_id in all_movie_ids:
            self.Q_values[movie_id] = np.random.rand()

    def update_Q_values(self, state, reward, next_state, alpha, gamma):
        current_Q = self.Q_values.get(state, np.random.rand())
        next_Q = self.Q_values.get(next_state, np.random.rand())
        updated_Q = current_Q + alpha * (reward + gamma * next_Q - current_Q)
        self.Q_values[state] = updated_Q

    def select_action(self, state, epsilon):
        possible_actions = self.movies['MovieID'].tolist()
        if np.random.rand() < epsilon:
            q_values = [self.Q_values.get((state, action), 0) for action in possible_actions]
            best_action = possible_actions[np.argmax(q_values)]
            return best_action
        else:
            return np.random.choice(possible_actions, size=1)[0]

    def recommend_movies_td(self, user_id, top_n=5, alpha=0.1, gamma=0.9, epsilon=0.1):
        user_history = self.get_user_history(user_id)

        self.initialize_Q_values()

        for i in range(len(user_history) - 1):
            state = user_history.iloc[i]['MovieID']
            reward = user_history.iloc[i + 1]['Rating']
            next_state = user_history.iloc[i + 1]['MovieID']

            self.update_Q_values(state, reward, next_state, alpha, gamma)

        current_state = user_history.iloc[-1]['MovieID']
        recommended_movie_ids = [self.select_action(current_state, epsilon) for _ in range(top_n)]
        recommended_movie_names = self.movies[self.movies['MovieID'].isin(recommended_movie_ids)]['Title'].tolist()

        return recommended_movie_names

if __name__ == "__main__":
    folder_path = "ml-1m"

    movies = read_movie_data(folder_path)
    ratings = read_rating_data(folder_path)

    td_recommender = TDMovieRecommender(movies, ratings)

    user_id_to_recommend = int(input("Enter User ID: "))
    recommended_movies = td_recommender.recommend_movies_td(user_id_to_recommend, top_n=5)
    print(f'\nTop Recommendations for User {user_id_to_recommend}:')
    print(recommended_movies)
