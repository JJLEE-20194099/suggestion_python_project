from os import sep
import pandas as pd;
import numpy as np;
from scipy import sparse;
from sklearn.metrics.pairwise import cosine_similarity

class collaborative_filering:
    def __init__(self, Y, k_neighbors, distance_func = cosine_similarity, mode = 1):
        self.mode = mode;
        if self.mode == 1:
            self.Y = Y
        else: 
            self.Y = Y[:, [1, 0, 2]]
        
        self.k_neighbors = k_neighbors
        self.distance_func = distance_func
        self.Y_utility = None
        self.no_users = int(np.max(self.Y[:, 0])) + 1
        self.no_movies = int(np.max(self.Y[:, 1])) + 1

    def insert(self, data):
        self.Y = np.concatenate((self.Y, data), axis = 0)

    def normalize_data(self):
        users = self.Y[:, 0]
        self.Y_utility = self.Y.copy()
        self.mean_user = np.zeros((self.no_users, ))
        for i in range(self.no_users):
            user_id_list = np.where(users == i)[0].astype(np.int32)
            item_id_list = self.Y[user_id_list, 1]
            rating_list = self.Y[user_id_list, 2]
            m = np.mean(rating_list)
            if np.isnan(m):
                m = 0
            self.mean_user[i] = m
            self.Y_utility[user_id_list, 2] = rating_list - self.mean_user[i]
            
        self.Y_utility_sparse = sparse.coo_matrix((self.Y_utility[:, 2], (self.Y_utility[:, 1], self.Y_utility[:, 0])),  (self.no_movies, self.no_users))
        self.Y_utility_sparse = self.Y_utility_sparse.tocsr()
    

    def cal_similarity(self):
        self.similarity_matrix = self.distance_func(self.Y_utility_sparse.T, self.Y_utility_sparse.T)
       
    
    def refresh(self):
        self.normalize_data()
        self.cal_similarity()
    
    def fit(self):
        self.refresh()
    

    def predict_utils(self, user, i, normalized = 1):
        user_id_list = np.where(self.Y[:, 1] == i)[0].astype(np.int32)
        user_id_list = (self.Y[user_id_list, 0]).astype(np.int32)
        similar = self.similarity_matrix[user, user_id_list]

        k_user_id_nearest = np.argsort(similar)[-self.k_neighbors: ]

        k_simlilar_nearest = similar[k_user_id_nearest]
        
        r = self.Y_utility_sparse[i, user_id_list[k_user_id_nearest]]
        
        if normalized:
            return (r * k_simlilar_nearest)[0] / (np.abs(k_simlilar_nearest).sum() + 1e-8) + self.mean_user[user]

        return (r * k_simlilar_nearest)[0] / (np.abs(k_simlilar_nearest).sum() + 1e-8)
 

    def predict(self, user, i, normalized = 1):
        
        if self.mode:
            return self.predict_utils(user, i, normalized)
        return self.predict_utils(i, user, normalized)
    

    def suggest(self, user, normalized = 1):

        row_id_list = np.where(self.Y[:, 0] == user)[0]
        movie_list = self.Y[row_id_list, 1].tolist()

        res = []

        for i in range(self.no_movies):
            if i not in movie_list:
                rating = self.predict_utils(user, i)
                
                if rating > self.mean_user[user]:
                    res.append(i)
        
        return res
    
    def print(self):
        for i in range(self.no_users):
            res = self.suggest(i)
            if self.mode:
                print ('Recommend item(s):', res, 'to user', i)
            else: 
                print ('Recommend item', i, 'to user(s) : ', res)

cols = ['user_id', 'item_id', 'rating']
ratings = pd.read_csv('./data/ml-100k/me_test.dat', sep=' ', names=cols, encoding='latin-1')
Y = ratings.values

CF_model = collaborative_filering(Y, k_neighbors = 2, mode= 1)
CF_model.fit()

CF_model.print()
