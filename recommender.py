'''
SOME RESOURCES
https://blog.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/
https://stats.stackexchange.com/questions/28406/is-cosine-similarity-a-classification-or-a-clustering-technique
https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/index.html
https://www.quora.com/Can-scikit-learn-be-used-to-build-a-recommendation-system
https://muricoca.github.io/crab/
'''

import numpy
import pandas

header = ['userID', 'gameID', 'rating']
df = pandas.read_csv('boardgame-elite-users.csv', names=header)
n_users = df.userID.unique().shape[0]
n_gameIDs = df.gameID.unique().shape[0]

# TRAIN / TEST DATA SPLIT
# should maybe grab some ideal test data (users with few ratings)
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.2)

# CREATE PIVOT TABLES
all_data_ptable = pandas.pivot_table(df, index='userID', columns='gameID', values='rating', fill_value=0)
train_data_ptable = pandas.pivot_table(train_data, index='userID', 
columns='gameID', values='rating', fill_value=0)
test_data_ptable = pandas.pivot_table(test_data, index='userID', columns='gameID', values='rating', fill_value=0)

# FIND USERS AND ITEMS THAT ARE SIMILAR
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_ptable, metric='cosine')
item_similarity = pairwise_distances(train_data_ptable.T, metric='cosine')

# PREDICTION MATRIX
def predict(ratings_array, similarity, type='user'):
    if type == 'user':
        # Normalize user ratings
        mean_user_rating = ratings_array.mean(axis=1)
        ratings_diff = (ratings_array - mean_user_rating[:, numpy.newaxis])
        pred = mean_user_rating[:, numpy.newaxis] + similarity.dot(ratings_diff) / numpy.array([numpy.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings_array.dot(similarity) / numpy.ratings_array([numpy.abs(similarity).sum(axis=1)])
    return pred

# game_rating_array = numpy.array(train_data_ptable.T)
train_data_array = numpy.array(train_data_ptable)
user_prediction = predict(train_data_array, user_similarity, type='user')

#item_prediction = predict(game_rating_array, item_similarity, type='item')

#*********************************************************************************************
# TAKE userID, gameID; RETURN predicted rating
# should change array to dictionary with userID as keys?
''' 
MATHJAX
This is the formula appliec in predict()
\[ \hat{x}_{k,m} = \bar{x}_{k} + \frac{\sum\limits_{u_a} sim_u(u_k, u_a) (x_{a,m} - \bar{x}_{u_a})}{\sum\limits_{u_a}|sim_u(u_k, u_a)|} \]
'''
def rating_prediction(userID, gameID):
    # Ensure userID has not already rated game
    assert(all_data_ptable[userID][gameID] == 0)
    # create an array from userID
    userRatingArray = numpy.array(all_data_ptable[userID])
    gameRatingArray = numpy.array(all_data_ptable.T[gameID])
    
rating_prediction(3, 5038)








#**********************************************************************************************
# EVALUATION, RMSE

# from sklearn.metrics import mean_squared_error
# from math import sqrt
# def rmse(prediction, ground_truth):
#     prediction = prediction[ground_truth.nonzero()].flatten()
#     ground_truth = ground_truth[ground_truth.nonzero()].flatten()
#     return sqrt(mean_squared_error(prediction, ground_truth))
# 
# test_data_array = numpy.array(test_data_ptable)
# print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_array)))
# print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_array)))

#*********************************************************************************************
# TEST KMEANS CLUSTERING
# MAYBE cluster user_similarity?

# from sklearn.cluster import KMeans
# kmeans = KMeans()
# kmeans.fit(train_data_array)
# print(kmeans.predict([test_data_array[0]]))
# print(kmeans.labels_)
# centroids = kmeans.cluster_centers_

# import matplotlib.pyplot as plt_
# plt.scatter(centroids[:,0], centroids[:,1])
# plt.scatter(test_data_array[:,0], test_data_array[:,1])
# plt.show()
#*********************************************************************************************
# SPARSITY
'''
# note Python 2.something, update it, make sure functions operate the same
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print 'Sparsity: {:4.2f}%'.format(sparsity)
'''