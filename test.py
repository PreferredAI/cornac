#Importing required modules from Cornac.
from cornac.models import Coe
from cornac.experiment import Experiment
from cornac.evaluation_strategies import Split
from cornac import metrics 

#Importing some additional useful modules.
from scipy.io import loadmat

#Loading and preparing the Amazon office data,
#Available in the GitHub repository, inside folder 'data/'. 
office= loadmat("./data/office.mat")
mat_office = office['mat']

#Instantiate a pfm recommender model.
#Please refer to the documentation for details on parameter settings.
rec_ibpr = Ibpr(k=20, max_iter=100, learning_rate = 0.001, lamda = 0.05, batch_size = 200, init_params={'U':None,'V':None})

#Instantiate an evaluation strategy.
es_split = Split(data = mat_office, prop_test=0.2, prop_validation=0.0, good_rating=4)

#Instantiate evaluation metrics.
ndcg = metrics.Ndcg(m=20)

#Instantiate and then run an exp  eriment.
res_ibpr = Experiment(es_split, [rec_ibpr], metrics=[ndcg])
res_ibpr.run_()

#Get average results.
print(res_ibpr.res_avg)