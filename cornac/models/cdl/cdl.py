# -*- coding: utf-8 -*-

"""
@author: Trieu Thi Ly Ly 
"""

import numpy as np 
import tensorflow as tf 
import numpy as np
from ...utils.data_utils import Dataset

# Stacked Denoising Autoencoder
def sdae(text_information_corrupted, autoencoder_structure, W, b, batch_size, cdl_keep_prob):
	L = len(autoencoder_structure)
	for i in range(L - 1):
		# The encoder
		if i <= int(L / 2) -1:
			if i == 0:
				# The first layer
				h_value = tf.nn.sigmoid(tf.add(tf.matmul(text_information_corrupted,W[i]), b[i]))
			else:
				h_value = tf.nn.sigmoid(tf.add(tf.matmul(h_value,W[i]), b[i]))
		# The decoder
		elif i > int(L / 2) -1:
			h_value = tf.nn.sigmoid(tf.add(tf.matmul(h_value,W[i]), b[i]))
		# The dropout for all the layers except the final one
		if i < (L - 2):
			h_value = tf.nn.dropout(h_value, cdl_keep_prob)
		# The encoder output value
		if i == int(L / 2) -1:
			encoded_value = h_value

	sdae_value = h_value
		
	return sdae_value, encoded_value


""" Generate an user-by-item rating matrix R and confidence C
	if user i has/ likes item j , R[i,j] = 1 and C[i,j] = a
	otherwise, R[i,j] = 0 and C[i,j] = b
"""

def formatData(X,n_users, n_items,a,b):
	X = X.tocoo()
	row = X.row # user
	col = X.col # item
	R = np.zeros((n_users, n_items))
	C = np.ones((n_users, n_items)) * b
	for i in range(X.data.shape[0]):
		user = row[i]
		item = col[i]
		
		R[int(user), int(item)] = 1
		C[int(user), int(item)] = a

	return R, C


# Collaborative Deep Learning
def cdl(X, text_information, autoencoder_structure, k = 50 , lambda_u = 0.01, lambda_v = 0.01,lambda_w = 0.01, lambda_n = 0.01, a = 1, b = 0.01, autoencoder_corruption = 0.3, n_epochs=100, learning_rate=0.001, keep_prob = 0.9,batch_size = 100, init_params=None):

	n_users = len(set(X.indices))
	n_items = text_information.shape[0]
	n_vocabularies = text_information.shape[1]
	autoencoder_structure = [n_vocabularies] + autoencoder_structure + [k] + autoencoder_structure + [n_vocabularies]

	R, C = formatData(X,n_users, n_items, a, b) 

	#Initial user factors
	if init_params['U'] is None:
		U = tf.Variable(tf.random_normal([n_users, k]) ,name="user_factor", dtype=tf.float32)
	else:
		U = init_params['U']

	#Initial item factors
	if init_params['V'] is None:
		V = tf.Variable(tf.random_normal([n_items, k]), name="item_factor", dtype=tf.float32)
	else:
		V = init_params['V']

	cdl_mask_corruption = tf.placeholder(dtype=tf.float32, shape=[None, n_vocabularies], name="cdl_mask_corruption")
	cdl_text_information = tf.placeholder(dtype=tf.float32, shape=[None, n_vocabularies], name="cdl_text_information")
	cdl_rating_R = tf.placeholder(dtype=tf.float32, shape=[n_users,None], name="cdl_rating_R")
	cdl_C = tf.placeholder(dtype=tf.float32, shape=[n_users,None], name="cdl_C")
	cdl_keep_prob = tf.placeholder(dtype=tf.float32)
	
	cdl_batch = tf.placeholder(dtype=tf.int32)
	V_batch = tf.gather(V, cdl_batch)
	
	# The Weight and Bias for each layer in SDAE
	L = len(autoencoder_structure)
	W, b = dict(), dict()
	for i in range(L - 1):
		W[i] = tf.Variable(tf.random_normal([autoencoder_structure[i],autoencoder_structure[i+1]]))
		b[i] = tf.Variable(tf.random_normal([autoencoder_structure[i+1]]))

	text_information_corrupted = cdl_text_information + cdl_mask_corruption 

	sdae_value, encoded_value = sdae(text_information_corrupted, autoencoder_structure, W, b, batch_size, cdl_keep_prob)
	mask_corruption_np = np.random.binomial(1, 1-autoencoder_corruption,
											(n_items, n_vocabularies))
											
	# Generate loss function
	loss_w_b = tf.constant(0, dtype=tf.float32)
	for i in range(L - 1):
		loss_w_b = tf.add(loss_w_b, tf.add(tf.nn.l2_loss(W[i]), tf.nn.l2_loss(b[i])))

	loss_1 = lambda_u * tf.nn.l2_loss(U) + lambda_w * loss_w_b
	loss_2 = lambda_v * tf.nn.l2_loss(V_batch - encoded_value)
	loss_3 = lambda_n * tf.nn.l2_loss(sdae_value - cdl_text_information)
	loss_4 = tf.reduce_sum(tf.multiply(cdl_C,
		tf.square(cdl_rating_R - tf.matmul(U, V_batch, transpose_b=True))))
	
	loss = loss_1 + loss_2 + loss_3 + loss_4

	# Generate optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	# optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for _ in range(n_epochs):
			Data = Dataset(np.random.permutation(n_items))
			num_steps = int(Data.data.shape[0] / batch_size)
			for i in range(1, num_steps + 1):
				batch, _ = Data.next_batch(batch_size)
				_, _loss = sess.run(
					[optimizer, loss],
					feed_dict = {
						cdl_mask_corruption: mask_corruption_np[batch,:],
						cdl_text_information: text_information[batch,:],
						cdl_rating_R: R[:, batch],
						cdl_C: C[:, batch],
						cdl_keep_prob: keep_prob,
						cdl_batch: batch
					}
				)
 
		U_out, V_out = sess.run([U, V])
		res = {'U': U_out, 'V': V_out}
		
	return res