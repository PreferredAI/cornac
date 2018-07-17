# -*- coding: utf-8 -*-
"""
@author: asalah
"""

import gzip
import numpy as np 

class Dataset:

    def __init__(self,data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[0]
        self.index = None
        pass


    @property
    def data(self):
        #print("I use the property")
        return self._data

    #in this version we do not shuffle the original data (only the ids)
    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            print('Shafling the data')
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self.index = idx
            #self._data = self.data[idx]  # get list of `num` random samples

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[self.index][start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            #self._data = self.data[idx0]  # get list of `num` random samples
            idex_rest = self.index[start:self._num_examples]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            data_new_part =  self._data[idx0][start:end] 
            #data_new_part =  self._data[start:end]
            index_new = idx0[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate((idex_rest, index_new)) 
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            #alose return the ids
            return self._data[self.index][start:end], self.index[start:end]
            #return self._data[start:end], self.index[start:end]

#dataset = Dataset(np.arange(10, 21))
#for i in range(5):
#    print(dataset.next_batch(2))
        
    
def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]    
 
    
def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(28 * 28 * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    #data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, 28*28)
    return data    
    
    
    
    
    
    
    
    