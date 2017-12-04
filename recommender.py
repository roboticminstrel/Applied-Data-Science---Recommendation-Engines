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

# TRAIN / TEST DATA SPLIT random_state=1
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
''' 
MATHJAX
This is the formula applied in derive_prediction_matrix()
\[ \hat{x}_{k,m} = \bar{x}_{k} + \frac{\sum\limits_{u_a} sim_u(u_k, u_a) (x_{a,m} - \bar{x}_{u_a})}{\sum\limits_{u_a}|sim_u(u_k, u_a)|} \]
'''
def derive_prediction_matrix(ratings_array, similarity, type='user'):
    if type == 'user':
        # Normalize user ratings
        mean_user_rating = ratings_array.mean(axis=1)
        ratings_diff = (ratings_array - mean_user_rating[:, numpy.newaxis])
        pred = mean_user_rating[:, numpy.newaxis] + similarity.dot(ratings_diff) / numpy.array([numpy.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings_array.dot(similarity) / numpy.ratings_array([numpy.abs(similarity).sum(axis=1)])
    return pred

train_data_array = numpy.array(train_data_ptable)
user_prediction_matrix = derive_prediction_matrix(train_data_array, user_similarity, type='user')
# game_rating_array = numpy.array(train_data_ptable.T)
#item_prediction_matrix = derive_prediction_matrix(game_rating_array, item_similarity, type='item')

#*********************************************************************************************
# PREDICT rating GIVEN userID AND gameID
# Working... Needs to account for user not in DB but has a rating array?

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
# print('User-based CF RMSE: ' + str(rmse(user_prediction_matrix, test_data_array)))
# print('Item-based CF RMSE: ' + str(rmse(item_prediction_matrix, test_data_array)))

#*********************************************************************************************
# SPARSITY of all ratings

# non_zero_ratings = 0.0
# total_rating_opportunities = float(len(numpy.array(all_data_ptable).flatten()))
#
# for k in numpy.array(all_data_ptable).flatten():
#     if int(k) > 0:
#         non_zero_ratings += 1
#         
# sparsity = non_zero_ratings / total_rating_opportunities
# print(sparsity)

