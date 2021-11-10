from re import A
import pandas as pd;
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np;

user_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./data/ml-100k/u.user', sep='|', names=user_cols, encoding='latin-1')

n_users = users.shape[0]

print ('Number of users: ', n_users)

rating_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('./data/ml-100k/ua.base', sep='\t', names=rating_cols, encoding='latin-1')
ratings_test = pd.read_csv('./data/ml-100k/ua.test', sep='\t', names=rating_cols, encoding='latin-1')

# print(type(ratings_base))
ratings_train_arr = ratings_base.values
# print(type(ratings_train_arr))
ratings_test_arr = ratings_test.values


print('ratings_train_shape: ', ratings_train_arr.shape)
print('ratings_test_shape: ', ratings_test_arr.shape)

movie_theme_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv('./data/ml-100k/u.item', sep='|', names=movie_theme_cols, encoding='latin-1')

no_movies = movies.shape[0]
print('No movie themes: ', no_movies)

X_train = movies.values[:, -19:]
print(X_train)

transformer = TfidfTransformer(smooth_idf=True, norm='l2')
tfidf = transformer.fit_transform(X_train).toarray()

print("tfidf: ", tfidf.shape)

def get_movies_rated_by_user(utility_matrix, user_id):
    user_id_list = utility_matrix[:, 0]
    
    row_ids = np.where(user_id_list == user_id + 1)[0]
    movie_id_list = utility_matrix[row_ids, 1] - 1
    rating_list = utility_matrix[row_ids, 2]
    return (movie_id_list, rating_list)

from sklearn.linear_model import Ridge
from sklearn import linear_model

no_movie_theme = tfidf.shape[1]
w = np.zeros((no_movie_theme, n_users))
b = np.zeros((1, n_users))

for i in range(n_users):
    movie_id_list, rating_list = get_movies_rated_by_user(ratings_train_arr, i)
    ridge = Ridge(alpha=0.01, fit_intercept=True)

    tfdif_by_user = tfidf[movie_id_list, :]
    ridge.fit(tfdif_by_user, rating_list)

    w[:, i] = ridge.coef_
    b[0, i] = ridge.intercept_

Y = tfidf.dot(w)  + b;
n = 4
np.set_printoptions(precision=2)
movie_id_list, rating_list = get_movies_rated_by_user(ratings_test_arr, n)

print('True Rating List: ', rating_list)
print('Predict Rating List: ', Y[movie_id_list, n])






