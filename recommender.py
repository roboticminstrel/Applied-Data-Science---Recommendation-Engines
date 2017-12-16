# Load external data file into "df"
# Ensure proper header/index/column labeling as "userID", "gameID", "rating"
# Call predict_rating(userID, gameID)
# That's it! You can return the top eight suggested gameID with get_top_suggestions(userID)
# Data objects are "df", "all_data_ptable", "ratings_array", "user_similarity", "user_prediction_matrix"
# Data objects stay the same. New values are returned by all functions.
import numpy
import pandas

# header = ['userID', 'gameID', 'rating']
df = pandas.read_csv('inputs/withheld-ratings.csv').rename(columns = {'Compiled from boardgamegeek.com by Matt Borthwick':'userID'})

all_data_ptable = pandas.pivot_table(df, index='userID', columns='gameID', values='rating', fill_value=0)
ratings_array = numpy.array(all_data_ptable)

# For Testing
df2 = pandas.read_csv('inputs/boardgame-elite-users.csv').rename(columns = {'Compiled from boardgamegeek.com by Matt Borthwick':'userID'})
all_data_ptable2 = pandas.pivot_table(df2, index='userID', columns='gameID', values='rating', fill_value=0)
ratings_array2 = numpy.array(all_data_ptable2)

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(ratings_array, metric='cosine')

# Takes a 2d array (199, 402) and a 2d array (199, 199), returns a 2d array (199, 402) of the predicted rating
# for each userID-gameID pair
def derive_prediction_matrix(ratings_arr, user_simil):
    mean_user_rating = ratings_arr.mean(axis=1)
    ratings_minus_mean = (ratings_arr - mean_user_rating[:, numpy.newaxis])
    pred = mean_user_rating[:, numpy.newaxis] + user_simil.dot(ratings_minus_mean) / numpy.count_nonzero(numpy.array(ratings_arr).T, axis = 1)
    # Add back mean_user_rating
    pred = pred + mean_user_rating[:, numpy.newaxis]
    return pred

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

# Takes 2 ints, returns a float rounded to nearest half (1.0, 1.5, 2.0)
def predict_rating(userID, gameID):
    user_location = numpy.where(all_data_ptable.index.values == userID)
    game_location = numpy.where(all_data_ptable.columns == gameID)
    return max_ten(round_to_p5(user_prediction_matrix[user_location, game_location]))

# Takes an int and returns a list of 8 ints (gameIDs)
def get_top_suggestions(userID):
    rating_game_tuples = []
    for gameID in all_data_ptable.columns:
        if all_data_ptable.loc[(userID, gameID)] < .1:
            rating = predict_rating(userID, gameID)
            rating_game_tuples.append((rating, gameID))
    # return gameID list of highest 8 predicted ratings
    return [x[1] for x in sorted(rating_game_tuples, reverse = True)[:9]]
    
# Return a matrix with both predictions for unrated gameIDs AND ratings of gameIDs already rated
def correct_for_existing_ratings(pred_matrix, ratings_arr):
    new_pred = []
    for i, vec in enumerate(ratings_arr):
        for j, val in enumerate(ratings_arr[i]):
            if ratings_arr[i][j] > 0:
                new_pred.append(ratings_arr[i][j])
            else:
                new_pred.append(pred_matrix[i][j])
    new_pred = numpy.array(new_pred)
    new_pred.shape = pred_matrix.shape
    return new_pred



print(get_top_suggestions(388))
print(predict_rating(388, 39463))
print(predict_rating(388, 35677))
print(predict_rating(388, 17226))
print(predict_rating(388, 25613))
print(predict_rating(388, 96848))
print(predict_rating(430, 31260))
print(predict_rating(430, 25613))
print(predict_rating(430, 161936))
print(predict_rating(430, 188))
print(predict_rating(3080, 36932))
print(predict_rating(3080, 9209))
print(predict_rating(3080, 161936))
print(predict_rating(3080, 90137))

#**********************************************************************************************
# TESTS / PRINTS / ASSERTS
# print(list(zip([predict_rating(272, x) for x in train_data_ptable.columns], list(train_data_ptable.loc[(272,)]))))
#**********************************************************************************************
# EVALUATION, RMSE
from math import sqrt
def rmse(predictions, targets):
    return numpy.sqrt(((predictions - targets) ** 2).mean())
print('RMSE with all sevens is ', rmse(numpy.full(ratings_array.shape, 7), ratings_array2))
print('RMSE with pred matrix is ', rmse(user_prediction_matrix, ratings_array2))
print('RMSE with corrected pred matrix is ', rmse(correct_for_existing_ratings(user_prediction_matrix, ratings_array), ratings_array2))