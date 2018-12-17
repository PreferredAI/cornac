First example
==============

This example will show you how to run your very first experiment using Cornac. It consists in training and evaluating the Probabilistic Matrix Factorization (PMF) recommender model.
::

	#Importing required modules from Cornac.
	from cornac.models import PMF
	from cornac.experiment import Experiment
	from cornac.evaluation_strategies import Split
	from cornac import metrics 
	
	#Importing some additional useful modules.
	from scipy.io import loadmat
	
	#Loading and preparing the Amazon office data,
	#Available in the GitHub repository, inside folder 'data/'. 
	office= loadmat("path to office.mat")
	mat_office = office['mat']

	#Instantiate a pfm recommender model.
	#Please refer to the documentation for details on parameter settings.
	rec_pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lamda=0.001, init_params={'U':None,'V':None})

	#Instantiate an evaluation strategy.
	es_split = Split(data = mat_office, prop_test=0.2, prop_validation=0.0, good_rating=4)

	#Instantiate evaluation metrics.
	rec = metrics.Recall(m=20)
	pre = metrics.Precision(m=20)
	mae = metrics.MAE()
	rmse = metrics.RMSE()

	#Instantiate and then run an experiment.
	res_pmf = Experiment(es_split, [rec_pmf], metrics=[mae,rmse,pre,rec])
	res_pmf.run()

	#Get average results.
	res_pmf.average_result