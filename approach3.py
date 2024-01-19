import os
import re
import random
import pandas as pd
import numpy as np
from collections import defaultdict

class DataLoader():
    def __init__(self):
        self.DATASETS_FOLDER = 'ml-1m'
        self.MOVIES_FILE = 'movies.dat'
        self.RATINGS_FILE = 'ratings.dat'
        self.MOVIES_COLUMNS = ['movie_id', 'title', 'genres']
        self.RATINGS_COLUMNS = ['user_id', 'movie_id', 'rating', 'timestamp']


    def read_dataset(self, file_name, columns):
        file_path = os.path.join(self.DATASETS_FOLDER, file_name)
        df = pd.read_csv(file_path, sep='::', engine='python', header=None, names=columns, encoding='latin-1')

        return df


    def load_datasets(self):
        movies = self.read_dataset(self.MOVIES_FILE, self.MOVIES_COLUMNS)
        ratings = self.read_dataset(self.RATINGS_FILE, self.RATINGS_COLUMNS)

        return (movies, ratings)


    def extract_year(self, title):
        year_match = re.search(r'\(\d{4}\)', title)
        if year_match:
            return int(year_match.group()[1:-1])

        return None


    def delete_year(self, title):
        if re.search(r'\(\d{4}\)', title):
            return title[:-6].strip()

        return title


    def calculate_period(self, year):
        return (year // 5) * 5


    def add_binary_genre_columns_to_movies(self, movies_df):
        movies_df = movies_df.copy(deep=True)

        if 'genres' not in movies_df:
            return movies_df, []

        movies_df['genres'] = movies_df['genres'].str.split('|').apply(
            lambda genre_columns: [genre.replace("'", "").lower() for genre in genre_columns])
        genre_columns = movies_df['genres'].explode().unique()
        for genre in genre_columns:
            movies_df[genre] = movies_df['genres'].apply(lambda x: genre in x).astype(int)
        movies_df = movies_df.drop('genres', axis=1)

        return movies_df, genre_columns


    def add_mean_rating_to_movies(self, movies_df, ratings_df):
        movies_df = movies_df.copy(deep=True)

        mean_ratings = ratings_df.groupby('movie_id')['rating'].mean().reset_index()
        mean_ratings.rename(columns={'rating': 'mean_rating'}, inplace=True)
        movies_df = pd.merge(movies_df, mean_ratings, on='movie_id', how='left')

        return movies_df


    def add_number_of_ratings_to_movies(self, movies_df, ratings_df):
        movies_df = movies_df.copy(deep=True)

        ratings_count = ratings_df['movie_id'].value_counts().reset_index()
        ratings_count.columns = ['movie_id', 'num_ratings']
        movies_df = pd.merge(movies_df, ratings_count, on='movie_id', how='left')

        return movies_df


    def preprocess(self):
        movies_df, ratings_df = self.load_datasets()
        movies_df['year'] = movies_df['title'].apply(self.extract_year)
        movies_df['period'] = movies_df['year'].apply(self.calculate_period)
        movies_df, genre_columns = self.add_binary_genre_columns_to_movies(movies_df)

        return (movies_df, ratings_df, genre_columns)


class PeriodGenreBasedMovieRecommender:
    def __init__(self):
        self.data_loader = DataLoader()
        self.movies, self.ratings, self.genre_columns = self.data_loader.preprocess()
        self.Q_values = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    def get_user_history(self, user_id):
        user_ratings = self.ratings[self.ratings['user_id'] == user_id].sort_values(by='timestamp',
                                                                                    ascending=True).merge(self.movies,
                                                                                                          on='movie_id')
        return user_ratings[['movie_id', 'rating', 'period', *self.genre_columns]]

    def initialize_Q_values(self):
        all_periods = self.movies['period'].unique()
        for period in all_periods:
            for genre in self.genre_columns:
                self.Q_values[(period, genre)] = 0

    def select_genre(self, movie_row):
        available_genres = [genre for genre in self.genre_columns if movie_row[genre] == 1]
        return np.random.choice(available_genres) if available_genres else np.random.choice(self.genre_columns)

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            filtered_movies = self.movies[(self.movies['period'] == state[0]) & (self.movies[state[1]] == 1)]
            return random.choice(filtered_movies['movie_id'].tolist()) if not filtered_movies.empty else None
        else:
            filtered_movies = self.movies[(self.movies['period'] == state[0]) & (self.movies[state[1]] == 1)]
            best_movie = None
            best_Q_value = -np.inf
            for _, movie in filtered_movies.iterrows():
                movie_id = movie['movie_id']
                Q_value = self.Q_values.get((state[0], state[1]), 0)
                if Q_value > best_Q_value:
                    best_Q_value = Q_value
                    best_movie = movie_id
            return best_movie

    def update_Q_values(self, current_state, reward, next_state):
        current_Q = self.Q_values.get(current_state, 0)
        max_next_Q = max([self.Q_values.get((next_state[0], g), 0) for g in self.genre_columns])
        updated_Q = current_Q + self.alpha * (reward + self.gamma * max_next_Q - current_Q)
        self.Q_values[current_state] = updated_Q

    def train(self, user_id):
        user_history = self.get_user_history(user_id)
        self.initialize_Q_values()

        for i in range(len(user_history) - 1):
            current_row = user_history.iloc[i]
            next_row = user_history.iloc[i + 1]

            current_state = (current_row['period'], self.select_genre(current_row))
            next_state = (next_row['period'], self.select_genre(next_row))

            current_movie = self.select_action(current_state, self.epsilon)
            next_movie = self.select_action(next_state, self.epsilon)

            reward = next_row['rating']
            self.update_Q_values(current_state, reward, next_state)

    def recommend_movies(self, user_id, top_n=5):
        self.train(user_id)
        watched_movie_ids = set(self.get_user_history(user_id)['movie_id'])

        sorted_Q_values = sorted(self.Q_values.items(), key=lambda item: item[1], reverse=True)
        recommended_movies_titles = []
        for (period, genre), _ in sorted_Q_values[:top_n]:
            recommended_movies = self.movies[(self.movies['period'] == period) & (self.movies[genre] == 1) & (
                ~self.movies['movie_id'].isin(watched_movie_ids))]
            if not recommended_movies.empty:
                recommended_movie_id = recommended_movies.iloc[0]['movie_id']
                watched_movie_ids.add(recommended_movie_id)
                recommended_movies_titles.append(recommended_movies.iloc[0]['title'])

        return recommended_movies_titles


if __name__ == "__main__":
    recommender = PeriodGenreBasedMovieRecommender()
    movies = recommender.recommend_movies(4169)
    print(movies)