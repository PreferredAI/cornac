import os
from datetime import datetime
import numpy as np
import cornac
from cornac.models import GlobalLocalKernel
from cornac.eval_methods import RatioSplit
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP
from cornac.models import MF, PMF, BPR

# Function to find the next available log file name
def get_next_log_file(base_name="experiment_log", ext=".txt"):
    counter = 1
    while os.path.exists(f"{base_name}_{counter}{ext}"):
        counter += 1
    return f"{base_name}_{counter}{ext}"

# Function to log results
def log_results(log_file, test_results, model_instance):
    with open(log_file, "a") as f:
        f.write("\n" + "=" * 40 + "\n")
        f.write(f"Experiment conducted on: {datetime.now()}\n")
        f.write("\nHyperparameters:\n")
        for attr, value in vars(model_instance).items():
            f.write(f"{attr}: {value}\n")
        f.write("\nTest Results:\n")
        f.write(test_results)
        f.write("\n" + "=" * 40 + "\n")

# Load the MovieLens 100K dataset
ml_100k = cornac.datasets.movielens.load_feedback()

# Take only a subset of the data, e.g., first 5000 interactions for quicker tests
# ml_100k = ml_100k[:500]

# Split the data
rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)

# Get the total number of users and items in the subset
n_u = rs.total_users
n_m = rs.total_items

print('Data matrix loaded')
print('Number of users: {}'.format(n_u))
print('Number of movies: {}'.format(n_m))
print('Number of training ratings: {}'.format(len(rs.train_set.uir_tuple[2])))
print('Number of test ratings: {}'.format(len(rs.test_set.uir_tuple[2])))

# Initialize your model
my_model = GlobalLocalKernel(
    # Example hyperparameters
    n_hid=10, 
    n_dim=2, 
    max_epoch_p=500, 
    max_epoch_f=500,
    lr_p=0.1,
    lr_f=0.01, 
    verbose=False
)

mf = MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123)
pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)
bpr = BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123)

# Define some basic metrics
metrics = [MAE(), RMSE(), Precision(k=10), Recall(k=10), NDCG(k=10), AUC(), MAP()]

# Redirect Cornac output to capture experiment results
from io import StringIO
import sys

# Get the next available log file name
log_file = get_next_log_file()
sys.stdout = StringIO()  # Redirect stdout to capture results

# Run the experiment on the smaller subset
cornac.Experiment(eval_method=rs, models=[my_model, mf, pmf, bpr], metrics=metrics, user_based=True).run()

# Retrieve experiment results
experiment_results = sys.stdout.getvalue()
sys.stdout = sys.__stdout__  # Restore stdout to original state

# Print the results to the console
print(experiment_results)

# Log results to file
log_results(log_file, experiment_results, my_model)

print(f"Experiment results and hyperparameters saved to {log_file}")
