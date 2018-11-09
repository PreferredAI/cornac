
from cornac.models import Vbpr
from cornac.models import Bpr
from cornac.experiment import Experiment
from cornac.evaluation_strategies import Split
from cornac import metrics
from cornac.utils import util_data
import numpy
import pickle

#Importing some additional useful modules.
from scipy.io import loadmat

#Loading and preparing the Amazon office data,
#Available in the GitHub repository, inside folder 'data/'.
office= loadmat(r"data\tradesy_top50.mat")
mat_office = office['mat']
rawdata = util_data.Dataset(mat_office)
validData,validUsers,validItems = rawdata.index_trans()

pickle_in = open("data\image.pickle", "rb")
imgs = pickle.load(pickle_in)

imgInfo = numpy.zeros([len(validItems), 4096])
for i, item in enumerate(validItems):
       imgInfo[i,:]=imgs.get(str(int(item)))

rec_vbpr = Vbpr(k=10, d=10, max_iter=100, aux_info=imgInfo, learning_rate=0.001, lamda=0.001, init_params={'U':None,'V':None,'E':None,'Ue':None}, batch_size=100)
rec_bpr = Bpr(k=10, max_iter=100, learning_rate=0.001, lamda=0.001, init_params={'U':None,'V':None},batch_size=100)

#Instantiate an evaluation strategy.
es_split = Split(data= validData, prop_test=0.2, prop_validation=0.0, good_rating=1)

#Instantiate evaluation metrics.
rec = metrics.Recall(m=20)
pre = metrics.Precision(m=20)
mae = metrics.Mae()
rmse = metrics.Rmse()

#Instantiate and then run an experiment.
res_pmf = Experiment(es_split, [rec_vbpr], metrics=[mae,rmse,pre,rec])
res_pmf.run_()


#Get average results.
print(res_pmf.res_avg)