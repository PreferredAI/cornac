# import cornac
# from cornac.eval_methods import RatioSplit
# from cornac.models import MF, PMF, BPR
# from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP

# # load the built-in MovieLens 100K and split the data based on ratio
# ml_100k = cornac.datasets.movielens.load_feedback()
# rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)

# # initialize models, here we are comparing: Biased MF, PMF, and BPR
# mf = MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123)
# pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)
# bpr = BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123)
# models = [mf, pmf, bpr]

# # define metrics to evaluate the models
# metrics = [MAE(), RMSE(), Precision(k=10), Recall(k=10), NDCG(k=10), AUC(), MAP()]

# # put it together in an experiment, voilà!
# cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()




import numpy as np
import cornac
from cornac.models import GlobalLocalKernel
from cornac.eval_methods import RatioSplit
from cornac.metrics import MAE, RMSE

# Load the MovieLens 100K dataset
ml_100k = cornac.datasets.movielens.load_feedback()

# Split the data
rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)

# Extract user, item, rating tuples
train_users, train_items, train_ratings = rs.train_set.uir_tuple
test_users, test_items, test_ratings = rs.test_set.uir_tuple

# Get the total number of users and items
n_u = rs.total_users
n_m = rs.total_items

# Prepare rating matrices in (n_m, n_u) format
train_r = np.zeros((n_m, n_u), dtype='float32')
test_r = np.zeros((n_m, n_u), dtype='float32')

# Populate the train and test matrices
train_r[train_items, train_users] = train_ratings
test_r[test_items, test_users] = test_ratings

train_m = (train_r > 1e-12).astype('float32')
test_m = (test_r > 1e-12).astype('float32')

print('data matrix loaded')
print('num of users: {}'.format(n_u))
print('num of movies: {}'.format(n_m))
print('num of training ratings: {}'.format(len(train_ratings)))
print('num of test ratings: {}'.format(len(test_ratings)))

# Initialize your model
my_model = GlobalLocalKernel()

# Provide the model with pre-processed train data
# my_model._train_mat = train_r     # Store original train matrix for scoring, if needed by score()
# my_model.train_r_local = train_r  # For pre-training phase if needed by the model

# Define some basic metrics
metrics = [MAE(), RMSE()]

# Run the experiment
cornac.Experiment(eval_method=rs, models=[my_model], metrics=metrics , user_based=True).run()
