# make estimate_rating_prediction() account for users not in train_data
# use cross-validation
# do something (implement correctly or get rid of) with item-similarity stuff
# what is the type of data received when receiving a new userID with ratings array?
# functional? what are states I am tracking?
# user mean ... divided by standard deviation
# dive-into.info survey, hit 'up'
# for each row of ratings matrix, find each nonzero, divide by, (in derive_prediction_matrix)
# khan academy ... random variables
# add back in mean user rating
import numpy
import pandas

# header = ['userID', 'gameID', 'rating']
df = pandas.read_csv('inputs/boardgame-elite-users.csv').rename(columns = {'Compiled from boardgamegeek.com by Matt Borthwick':'userID'})

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
# item_similarity = pairwise_distances(train_data_ptable.T, metric='cosine')

# PREDICTION MATRIX
def derive_prediction_matrix(ratings_array, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings_array.mean(axis=1)
        # subtract mean rating from each rating
        ratings_diff = (ratings_array - mean_user_rating[:, numpy.newaxis])
        # Generate predictions
        # should divide by the commonly rated gameIDs, pairwise mult
        pred = mean_user_rating[:, numpy.newaxis] + similarity.dot(ratings_diff) / numpy.count_nonzero(numpy.array(train_data_ptable).T, axis = 1)
    # For item similarity, not currently implemented
        pred = pred + mean_user_rating[:, numpy.newaxis]
        # Weighted sevens hack
        pred = (pred + numpy.full(pred.shape, 7)) / 2
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

# new formula to apply in derive_prediction_matrix
# print(numpy.count_nonzero(numpy.array(train_data_ptable).T, axis = 1)[0])

# OLD formula below (without index)
# print(numpy.array([numpy.abs(user_similarity).sum(axis=1)]).T[0])

user_prediction_matrix = derive_prediction_matrix(numpy.array(train_data_ptable), user_similarity, type='user')

# game_rating_array = numpy.array(train_data_ptable.T)
#item_prediction_matrix = derive_prediction_matrix(game_rating_array, item_similarity, type='item')

#*********************************************************************************************
# PREDICT rating GIVEN userID AND gameID
# gameID must be in data
# if userID is not in data, need an array of ratings for that user
def estimate_rating_prediction(userID, gameID):
    # if userID not in train_data, call functions to 'retrain'
    if userID not in train_data_ptable.index.values:
        # WORKING <===
        pass
    for i, user in enumerate(train_data_ptable.index.values):
        if user == userID:
            for j, game in enumerate(train_data_ptable.columns):
                if game == gameID:
                    return user_prediction_matrix[i][j]
assert(type(estimate_rating_prediction(272, 118)) == numpy.float64)

# Takes an int userID and returns a list of ints (gameIDs)
def get_all_suggestions(userID):
    rating_game_tuples = []
    for gameID in train_data_ptable.columns:
        rating = estimate_rating_prediction(userID, gameID)
        rating_game_tuples.append((rating, gameID))
    # return gameID list of highest 8 predicted ratings
    return [x[1] for x in sorted(rating_game_tuples, reverse = True)[:9]]
assert(type(get_all_suggestions(272) == list))

#**********************************************************************************************
# EVALUATION, RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print('User-based CF RMSE: ' + str(rmse(user_prediction_matrix, numpy.array(test_data_ptable))))
all_sevens_matrix = numpy.full((199, 402), 7)
print('All sevens rating system is RMSE ' + str(rmse(all_sevens_matrix, numpy.array(test_data_ptable))))
weighted_sevens_matrix = (all_sevens_matrix + user_prediction_matrix) / 2
print('Weighted sevens rating system is RMSE ' + str(rmse(weighted_sevens_matrix, numpy.array(test_data_ptable))))
# print('Item-based CF RMSE: ' + str(rmse(item_prediction_matrix, test_data_array)))