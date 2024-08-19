from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from recommendations import train_collaborative_model, get_collaborative_recommendations, get_content_recommendations

app = Flask(__name__, static_folder='static')

# In-memory storage for users and their ratings
users = {}

# Load data
ratings_df = pd.read_csv('datasets/movielens/user_rating_history.csv')
collaborative_algo = train_collaborative_model(ratings_df)


@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/register', methods=['POST'])
def register():
    user_id = request.json['user_id']
    if user_id in users:
        return jsonify({"message": "User already exists!"}), 400
    users[user_id] = []
    return jsonify({"message": "User registered successfully!"}), 201


@app.route('/rate', methods=['POST'])
def rate():
    user_id = request.json['user_id']
    movie_id = request.json['movie_id']
    rating = request.json['rating']
    if user_id not in users:
        return jsonify({"message": "User not found!"}), 404
    users[user_id].append((movie_id, rating))
    return jsonify({"message": "Rating submitted successfully!"}), 201


@app.route('/collaborative/<int:user_id>', methods=['GET'])
def collaborative_recommendations(user_id):
    recommendations = get_collaborative_recommendations(collaborative_algo, user_id, ratings_df)
    return jsonify(recommendations)


@app.route('/content/<string:title>', methods=['GET'])
def content_recommendations(title):
    recommendations = get_content_recommendations(title)
    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)
