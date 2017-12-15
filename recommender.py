import numpy
import pandas

# header = ['userID', 'gameID', 'rating']
df = pandas.read_csv('inputs/boardgame-elite-users.csv').rename(columns = {'Compiled from boardgamegeek.com by Matt Borthwick':'userID'})

# TRAIN / TEST DATA SPLIT
# from sklearn.model_selection import train_test_split
# train_data, test_data = train_test_split(df, test_size=0.2, random_state=1)

# CREATE PIVOT TABLE AND NUMPY ARRAY OF RATINGS
all_data_ptable = pandas.pivot_table(df, index='userID', columns='gameID', values='rating', fill_value=0)
ratings_array = numpy.array(all_data_ptable)

# FIND USERS THAT ARE SIMILAR
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(ratings_array, metric='cosine')

# Takes a 2d array (199, 402) and a 2d array (199, 199), returns a 2d array (199, 402) of the predicted rating
# for each userID-gameID pair
def derive_prediction_matrix(ratings_arr, user_simil):
    mean_user_rating = ratings_arr.mean(axis=1)
    ratings_minus_mean = (ratings_arr - mean_user_rating[:, numpy.newaxis])
    pred = mean_user_rating[:, numpy.newaxis] + user_simil.dot(ratings_minus_mean) / numpy.count_nonzero(numpy.array(ratings_arr).T, axis = 1)
    return pred + mean_user_rating[:, numpy.newaxis]

user_prediction_matrix = derive_prediction_matrix(ratings_array, user_similarity)

# Helper functions, round to a precision
def round_to(n, precision):
    correction = 0.5 if n >= 0 else -0.5
    return int( n/precision+correction ) * precision
def round_to_p5(n):
    return round_to(n, 0.5)
# Takes a float, ceilings it at 10.0, returns float
def max_ten(some_float):
    return 10.0 if some_float > 10 else some_float

# Takes 2 ints, returns a float
def predict_rating(userID, gameID):
    user_location = numpy.where(all_data_ptable.index.values == userID)
    game_location = numpy.where(all_data_ptable.columns == gameID)
    return max_ten(round_to_p5(user_prediction_matrix[user_location, game_location]))

# Takes an int userID and returns a list of 8 ints (gameIDs)
def get_top_suggestions(userID):
    rating_game_tuples = []
    for gameID in all_data_ptable.columns:
        if all_data_ptable.loc[(userID, gameID)] < .1:
            rating = predict_rating(userID, gameID)
            rating_game_tuples.append((rating, gameID))
    # return gameID list of highest 8 predicted ratings
    return [x[1] for x in sorted(rating_game_tuples, reverse = True)[:9]]
#**********************************************************************************************
# TESTS / PRINTS / ASSERTS
print([predict_rating(388, x) for x in [9216, 39463, 46, 35677, 17226, 25613]])
print(get_top_suggestions(388))
# print([predict_rating(272, x) for x in train_data_ptable.columns])
# compare the predictions with actual ratings
# print(list(zip([predict_rating(272, x) for x in train_data_ptable.columns], list(train_data_ptable.loc[(272,)]))))
#**********************************************************************************************
# EVALUATION, RMSE
from math import sqrt
def rmse(predictions, targets):
    return numpy.sqrt(((predictions - targets) ** 2).mean())
print('RMSE with all sevens is ', rmse(numpy.full(all_data_ptable.shape, 7), numpy.array(all_data_ptable)))
print('RMSE with pred matrix is ', rmse(user_prediction_matrix, numpy.array(all_data_ptable)))