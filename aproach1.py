#model using SVD
import os
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

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

def recommend_movies(user_id, model, ratings_df, top_n=5):
    user_rated_movies = ratings_df[ratings_df['UserID'] == user_id]['MovieID']
    all_movies = ratings_df['MovieID'].unique()
    unrated_movies = [movie_id for movie_id in all_movies if movie_id not in user_rated_movies]

    recommendations = pd.DataFrame({'UserID': [user_id] * len(unrated_movies), 'MovieID': unrated_movies})
    recommendations['Rating'] = recommendations.apply(lambda row: model.predict(user_id, row['MovieID']).est, axis=1)

    top_recommendations = recommendations.sort_values(by='Rating', ascending=False).head(top_n)
    return top_recommendations

if __name__ == "__main__":
    folder_path = "ml-1m"

    movies = read_movie_data(folder_path)
    ratings = read_rating_data(folder_path)
    users = read_user_data(folder_path)

    merged_data = pd.merge(ratings, movies, on='MovieID')
    users = pd.merge(users, merged_data, on='UserID')
    users['Genres'] = users['Genres'].str.split('|')

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['UserID', 'MovieID', 'Rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD()
    model.fit(trainset)

    predictions = model.test(testset)
    accuracy.rmse(predictions)

    user_id_to_recommend = 1
    recommended_movies = recommend_movies(user_id_to_recommend, model, ratings, top_n=5)
    print(f'\nTop Recommendations for User {user_id_to_recommend}:')
    print(recommended_movies)
