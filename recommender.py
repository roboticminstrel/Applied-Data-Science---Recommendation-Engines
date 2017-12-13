import numpy
import pandas

# header = ['userID', 'gameID', 'rating']
df = pandas.read_csv('inputs/boardgame-frequent-users.csv').rename(columns = {'Compiled from boardgamegeek.com by Matt Borthwick':'userID'})

# TRAIN / TEST DATA SPLIT
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=1)

# CREATE PIVOT TABLES
# all_data_ptable = pandas.pivot_table(df, index='userID', columns='gameID', values='rating', fill_value=0)
train_data_ptable = pandas.pivot_table(train_data, index='userID', columns='gameID', values='rating', fill_value=0)
test_data_ptable = pandas.pivot_table(test_data, index='userID', columns='gameID', values='rating', fill_value=0)

# FIND USERS AND ITEMS THAT ARE SIMILAR
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_ptable, metric='cosine')
# item_similarity = pairwise_distances(train_data_ptable.T, metric='cosine')

# PREDICTION MATRIX
def derive_prediction_matrix(ratings_array, similarity):
    mean_user_rating = ratings_array.mean(axis=1)
    # subtract mean rating from each rating
    ratings_diff = (ratings_array - mean_user_rating[:, numpy.newaxis])
    # Generate predictions
    pred = mean_user_rating[:, numpy.newaxis] + similarity.dot(ratings_diff) / numpy.count_nonzero(numpy.array(train_data_ptable).T, axis = 1)
# For item similarity, not currently implemented
    pred = pred + mean_user_rating[:, numpy.newaxis]
    # Weighted sevens hack
    pred = ((((pred + numpy.full(pred.shape, 7)) / 2) + numpy.full(pred.shape, 7)) / 2)
    return pred

user_prediction_matrix = derive_prediction_matrix(numpy.array(train_data_ptable), user_similarity)

# def derive_item_item_prediction_matrix(ratings_array, similarity):
#     pred = ratings_array.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
#     return pred

# game_rating_array = numpy.array(train_data_ptable.T)
# item_item_prediction_matrix = derive_item_item_prediction_matrix(game_rating_array, item_similarity)

#*********************************************************************************************
# PREDICT rating GIVEN userID AND gameID
# add back in mean user rating
def estimate_rating_prediction(userID, gameID):
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
        if train_data_ptable.loc[(userID, gameID)] < .1:
            rating = estimate_rating_prediction(userID, gameID)
            rating_game_tuples.append((rating, gameID))
    # return gameID list of highest 8 predicted ratings
    return [x[1] for x in sorted(rating_game_tuples, reverse = True)[:9]]
assert(sum([train_data_ptable.loc[(272, x)] for x in get_all_suggestions(272)]) == 0)
#**********************************************************************************************
# EVALUATION, RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print('User-based CF RMSE: ' + str(rmse(user_prediction_matrix, numpy.array(test_data_ptable))))
all_sevens_matrix = numpy.full(user_prediction_matrix.shape, 7)
print('All sevens rating system is RMSE ' + str(rmse(all_sevens_matrix, numpy.array(test_data_ptable))))