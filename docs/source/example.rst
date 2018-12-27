First example
==============

This example will show you how to run your very first experiment using Cornac. It consists in training and evaluating the Probabilistic Matrix Factorization (PMF) recommender model.
::

	# Importing required modules from Cornac.
	from cornac.models import PMF
	from cornac import Experiment
	from cornac.eval_strategies import RatioSplit
	from cornac.datasets import MovieLens100K
	from cornac import metrics 
	
	
	# Load the MovieLens 100K dataset
	ml_100k = MovieLens100K.load_data()
	
	# Instantiate an evaluation strategy.
	ratio_split = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, exclude_unknowns=False)

	# Instantiate a PMF recommender model.
	pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lamda=0.001)

	# Instantiate evaluation metrics.
	mae = metrics.MAE()
	rmse = metrics.RMSE()
	rec_20 = metrics.Recall(k=20)
	pre_20 = metrics.Precision(k=20)

	# Instantiate and then run an experiment.
	exp = Experiment(eval_strategy=ratio_split, 
			 models=[pmf], 
			 metrics=[mae, rmse, rec_20, pre_20], 
			 user_based=True)
	exp.run()
	
	# Get average results.
	exp.avg_results