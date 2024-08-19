from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split


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
