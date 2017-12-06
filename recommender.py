import numpy
import pandas

header = ['userID', 'gameID', 'rating']
df = pandas.read_csv('boardgame-elite-users.csv', names=header)

# TRAIN / TEST DATA SPLIT
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=1)

# CREATE PIVOT TABLES
all_data_ptable = pandas.pivot_table(df, index='userID', columns='gameID', values='rating', fill_value=0)
train_data_ptable = pandas.pivot_table(train_data, index='userID', columns='gameID', values='rating', fill_value=0)
test_data_ptable = pandas.pivot_table(test_data, index='userID', columns='gameID', values='rating', fill_value=0)

# FIND USERS AND ITEMS THAT ARE SIMILAR
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_ptable, metric='cosine')
item_similarity = pairwise_distances(train_data_ptable.T, metric='cosine')

# PREDICTION MATRIX
''' 
MATHJAX
This is the formula applied below in derive_prediction_matrix()
\[ \hat{x}_{k,m} = \bar{x}_{k} + \frac{\sum\limits_{u_a} sim_u(u_k, u_a) (x_{a,m} - \bar{x}_{u_a})}{\sum\limits_{u_a}|sim_u(u_k, u_a)|} \]
'''
def derive_prediction_matrix(ratings_array, similarity, type='user'):
    if type == 'user':
        # Normalize user ratings
        mean_user_rating = ratings_array.mean(axis=1)
        ratings_diff = (ratings_array - mean_user_rating[:, numpy.newaxis])
        # Generate predictions
        pred = mean_user_rating[:, numpy.newaxis] + similarity.dot(ratings_diff) / numpy.array([numpy.abs(similarity).sum(axis=1)]).T
    # For item similarity, not currently implemented
    elif type == 'item':
        pred = ratings_array.dot(similarity) / numpy.ratings_array([numpy.abs(similarity).sum(axis=1)])
    return pred

user_prediction_matrix = derive_prediction_matrix(numpy.array(train_data_ptable), user_similarity, type='user')
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