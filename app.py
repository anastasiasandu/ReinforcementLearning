from flask import Flask, request, render_template
import os

from approach1 import read_movie_data, read_rating_data, TDMovieRecommender
from approach2 import SARSAMovieRecommender
from approach3 import PeriodGenreBasedMovieRecommender
app = Flask(__name__)


folder_path = "ml-1m"
movies = read_movie_data(folder_path)
ratings = read_rating_data(folder_path)
td_recommender = TDMovieRecommender(movies, ratings)
sarsa_recommender = SARSAMovieRecommender(movies, ratings)
q_recommender = PeriodGenreBasedMovieRecommender()


@app.route('/', methods=['GET'])
def index():
    user_id = request.args.get('user_id')
    selected_model = request.args.get('model')
    if user_id:
        try:
            user_id = int(user_id)
            if selected_model == 'td':
                recommended_movies = td_recommender.recommend_movies_td(user_id, top_n=5)
            elif selected_model == 'sarsa':
                recommended_movies = sarsa_recommender.recommend_movies_sarsa(user_id, top_n=5)
            elif selected_model == 'qlearning':
                recommended_movies = q_recommender.recommend_movies(user_id)    
            else:
                raise ValueError("Invalid model selection")
            return render_template('recommendations.html', movies=recommended_movies, user_id=user_id)
        except ValueError as e:
            return render_template('recommendations.html', error=str(e))
        except Exception as e:
            return render_template('recommendations.html', error=f"An error occurred: {e}")
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False)
