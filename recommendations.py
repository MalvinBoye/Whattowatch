import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv('datasets/movielens/movies.csv')
ratings = pd.read_csv('datasets/movielens/user_rating_history.csv')

# Fill any missing values in the movies DataFrame
movies['genres'] = movies['genres'].fillna('')

# Collaborative Filtering
def train_collaborative_model(ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.25)
    algo = SVD()
    algo.fit(trainset)
    return algo

def get_collaborative_recommendations(algo, user_id, ratings, n=10):
    user_ratings = ratings[ratings['userId'] == user_id]
    user_unrated_movies = ratings[~ratings['movieId'].isin(user_ratings['movieId'])]
    user_predictions = [algo.predict(user_id, movie_id) for movie_id in user_unrated_movies['movieId']]
    user_predictions.sort(key=lambda x: x.est, reverse=True)
    top_n_predictions = user_predictions[:n]
    return [(pred.iid, pred.est) for pred in top_n_predictions]

# Content-Based Filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Function to get content-based recommendations dynamically
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_content_recommendations(title, n=10):
    idx = indices[title]
    tfidf_matrix_sparse = tfidf_matrix[idx]
    cosine_sim = cosine_similarity(tfidf_matrix_sparse, tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()
