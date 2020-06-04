import pandas as pd
import numpy as np
from scipy import sparse
from sklearn import metrics
import copy
import itertools
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook
from scipy import spatial
import logging
import os
import sys
from pandas.api.types import CategoricalDtype


class MFDataClean(object):

	def __init__(self, df, user_col, item_col, item_name, interaction_col):
		self.raw_df = df

		self.user_col = user_col
		self.item_col = item_col
		self.item_name = item_name
		self.interaction_col = interaction_col

		self.raw_df[user_col] = self.raw_df[user_col].astype(int)
		self.raw_df[item_col] = self.raw_df[item_col].astype(int)
		self.raw_df[item_name] = self.raw_df[item_name].astype(str)
		self.raw_df[interaction_col] = self.raw_df[interaction_col].astype(float)

		self.clean_data = self.subset(
			self.raw_df,
			self.user_col,
			self.item_col,
			self.item_name,
			self.interaction_col
		)

		self.clean_data = self.drop_missing(
			self.clean_data,
			self.user_col,
			self.item_col,
			self.interaction_col
		)

		self.item_lookup = self.create_item_lookup(
			self.clean_data,
			self.item_col,
			self.item_name
		)

		self.clean_data = self.clean_data.drop(
			labels=self.item_name, axis=1)  # Remove col of item name from interaction dataset.

	def subset(self, df, *cols):
		subset = df.loc[:,cols]
		return subset

	def drop_missing(self, df, *cols):
		clean_df = df.copy()
		for col in cols:
			clean_df = clean_df.loc[pd.isnull(clean_df[col]) == False]
		return clean_df

	def create_item_lookup(self, df, item_col, item_name):
		item_lookup = df[[item_col, item_name]].drop_duplicates().reset_index(
			drop=True)  # Only get unique item/description pairs
		return item_lookup


def threshold_interact(df, user_col, item_col, uid_min, iid_min):
	"""
	REF: https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/

	Return dataframe which every user has a least interacted with items more than iid_min required,
	and item has a least interacted with users more than uid_min required.

	@params
		df: dataframe of columns (user, item, interaction_num)
		user_col (str): user column name
		item_col (str): item column name
		uid_min (int): minimal items user needs to interact with
		iid_min (int): minimal users item needs to interact with

	@return
		df
	"""
	n_users = df[user_col].unique().shape[0]
	n_items = df[item_col].unique().shape[0]
	sparsity = float(df.shape[0]) / float(n_users * n_items) * 100
	print('Starting likes info')
	print('Number of users: {}'.format(n_users))
	print('Number of items: {}'.format(n_items))
	print('Sparsity: {:4.3f}%'.format(sparsity))

	done = False
	while not done:
		starting_shape = df.shape[0]

		# Drop off users who have interaction with items fewer than iid_min.
		iid_counts = df.groupby(user_col)[item_col].count()
		df = df[~df[user_col].isin(iid_counts[
									   iid_counts < uid_min].index.tolist())]

		# Drop off items which have interaction with users fewer than uid_min.
		uid_counts = df.groupby(item_col)[user_col].count()
		df = df[~df[item_col].isin(uid_counts[
									   uid_counts < iid_min].index.tolist())]
		ending_shape = df.shape[0]
		if starting_shape == ending_shape:
			done = True
		elif (df[user_col].unique().shape[0]) < 10 or (df[item_col].unique().shape[0]) < 10:
			done = True  # too few user and items left, STOP!!

	if (df[user_col].unique().shape[0]) >= 10 or (df[item_col].unique().shape[0]) >= 10:
		assert (df.groupby(user_col)[item_col].count().min() >= uid_min)
		assert (df.groupby(item_col)[user_col].count().min() >= iid_min)

	n_users = df[user_col].unique().shape[0]
	n_items = df[item_col].unique().shape[0]
	sparsity = float(df.shape[0]) / float(n_users * n_items) * 100
	print('Ending likes info')
	print('Number of users: {}'.format(n_users))
	print('Number of items: {}'.format(n_items))
	print('Sparsity: {:4.3f}%'.format(sparsity))
	return df


def df2interact_mat(df, user_col, item_col, interact_col):
	"""
	Convert (user, item, interact) dataframe to sparse interaction matrix.

	REF: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html

	***** csr_matrix((data, ij), [shape=(M, N)]) *****
	where data and ij satisfy the relationship a[ij[0, k], ij[1, k]] = data[k]

	:param df: dataframe of (user, item, interact)
	:param user_col: (string) user_id column name
	:param item_col: (string) item_id column name
	:param interact_col: (string) interaction column name

	:return:
		interaction_sparse: (csr) interaction matrix
		users: (numpy array) user_id
		items: (numpy array) item_id

	"""
	# Extract unique ids.
	users = np.array(list(np.sort(
		df[user_col].unique())))  # Get our unique users

	items = np.array(list(
		df[item_col].unique()))  # Get our unique items that were interacted with

	interactions = list(
		df[interact_col])  # All of our interactions

	# Re-code user and item index for sparse matrix.
	rows = df[user_col].astype(CategoricalDtype(
		categories=users)).cat.codes  # Re-code user index starting from 0 for sparse matrix.

	cols = df[item_col].astype(CategoricalDtype(
		categories=items)).cat.codes

	# Construct sparse matrix.
	# Get the associated column indices
	interaction_sparse = sparse.csr_matrix(
		(interactions, (rows, cols)), shape=(len(users), len(items))
	)

	return interaction_sparse, users, items


def train_test_split(ratings, split_count, fraction=None, dir_path=None, output_file_long=None):
	"""
	Split recommendation data into train and test sets,
	where non-overlapping non-zero entries are split into train and test sets.

	REF: https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/

	Params
	------
	ratings : scipy.sparse matrix
		Interactions between users and items.
	split_count : int
		Number of user-item-interactions per user to move
		from training to test set.
	fractions : float
		Fraction of users to split off some of their
		interactions into test set. If None, then all
		users are considered.
	dir_path : string
		directory where dataframe to be saved to
	output_file_long : string
		file name

	Return
		train (csr)
		test (csr)
		user_index: (list) user indices on whom some interactions are masked for testing set
	"""
	# Note: likely not the fastest way to do things below.
	train = ratings.copy().tocoo()
	train.eliminate_zeros()
	test = sparse.lil_matrix(train.shape)

	if fraction:
		try:
			user_index = np.random.choice(
				np.where(np.bincount(train.row) >= split_count * 2)[0],
				replace=False,
				size=np.int32(np.floor(fraction * train.shape[0]))).tolist()
			logging.info('length of user_index: {}'.format(len(user_index)))
		except BaseException as e:
			logging.info('Exception: {}'.format(e))
			logging.info(
				'Not enough users with >= {}, '
				'interactions for fraction of {}'.format(
					2 * split_count, fraction)
			)

			logging.info('Stop building models and exit program because not enough users for test...')
			empty_df = pd.DataFrame(columns=['company', 'model', 'member_id', 'product_id', 'title', 'score'])
			empty_df.to_csv(os.path.join(dir_path, output_file_long), index=False)
			sys.exit(0)
	else:
		user_index = range(train.shape[0])

	train = train.tolil()

	for user in user_index:
		test_ratings = np.random.choice(
			ratings.getrow(user).indices, size=split_count,
			replace=False)  # indices for nonzero cols
		train[user, test_ratings] = 0.
		# These are just 1.0 right now
		test[user, test_ratings] = ratings[user, test_ratings]
	logging.info('Length of test non-zero rows: {}'.format(len(np.unique(test.nonzero()[0]))))
	# Test and training are truly disjoint
	assert (train.multiply(test).nnz == 0)
	return train.tocsr(), test.tocsr(), user_index


def precision_at_k(model, true_ratings, k=5, user_index=None):
	"""
	Compute the percentage of top k items predicted by model are included in non-zero entries,
	comparing with ratings matrix.

	REF: https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/

	:param model: (lightFM model)
	:param true_ratings: (csr) interaction matrix
	:param k: (int) top k preferable items
	:param user_index: (list) index of matrix-split users
	:return:
		precisions: (float) mean top k precision for all users in user_index
	"""
	if not user_index:
		user_index = range(true_ratings.shape[0])
	precisions = []
	for user in user_index:
		# In case of large dataset, compute predictions row-by-row like below
		predictions = np.array(
			model.predict([user], list(range(true_ratings.shape[1]))))  # preference prediction for all items
		top_k = np.argsort(-predictions)[:k]  # top k item index
		labels = true_ratings.getrow(user).indices  # non-zero index
		precision = float(len(set(top_k) & set(labels))) / float(k)
		precisions.append(precision)
	return np.mean(precisions)


def auc_mean(model, train, test=None, user_index=None, mode='train'):
	"""
	Compute mean accuracy.
	In train mode, only entries in train are used, while in test mode,
	entries excluding non-zero train entries are used.

	:param model: (lightFM model)
	:param train: (csr)
	:param test: (csr)
	:param user_index: (list) index of matrix-split users
	:param mode: (string) train/test mode
	:return:
		auc: (float) mean of accuracy for all users in user_index
	"""
	if not user_index:
		user_index = range(train.shape[0])
	auc = []
	for user in user_index:
		preds = np.array(model.predict([user], list(range(train.shape[1]))))
		if mode == 'train':
			col_indices = list(range(train.shape[1]))  # entries from train matrix
			actual = train[user, col_indices].toarray()[0]  # list of list
			actual[actual > 0] = 1
			pred = preds[col_indices]
		elif mode == 'test':
			col_indices = set(range(train.shape[1])).difference(
				set(train[user].nonzero()[1]))  # entries excluding train matrix
			col_indices = list(col_indices)
			actual = test[user, col_indices].toarray()[0]  # list of list
			actual[actual > 0] = 1
			pred = preds[col_indices]
		fpr, tpr, thresholds = metrics.roc_curve(actual, pred)
		auc.append(metrics.auc(fpr, tpr))

	return np.mean(auc)


def print_log(row, header=False, spacing=12):
	"""
	Helper function to print log per line.

	:param row: (list) list of metric values
	:param header: (boolean) indicator to print header
	:param spacing: (int) spacing
	:return:
		No return.
	"""
	top = ''
	middle = ''
	bottom = ''
	for r in row:
		top += '+{}'.format('-' * spacing)
		if isinstance(r, str):
			middle += '| {0:^{1}} '.format(r, spacing - 2)
		elif isinstance(r, int):
			middle += '| {0:^{1}} '.format(r, spacing - 2)
		elif isinstance(r, float):
			middle += '| {0:^{1}.5f} '.format(r, spacing - 2)
		bottom += '+{}'.format('=' * spacing)
	top += '+'
	middle += '|'
	bottom += '+'
	if header:
		print(top)
		print(middle)
		print(bottom)
	else:
		print(middle)
		print(top)


def learning_curve_predict_k(model, train, all_data, epochs=range(2, 40, 2), k=5, user_index=None):
	"""
	Helper function to depict the improvement as training iteration goes on at top k precision.

	:param model: (lightFM model)
	:param train: (csr)
	:param all_data: (csr) original interaction matrix
	:param epochs: (iterator)
	:param k: (int) top k for precision coverage
	:param user_index: (list) index of matrix-split users
	:return:
		model: (lightFM model)
		train_precision: (list)
		all_data_precision: (list)
	"""
	if not user_index:
		user_index = range(train.shape[0])
	prev_epoch = 0
	train_precision = []
	all_data_precision = []

	headers = ['epochs', 'p@k train', 'p@k all_data']
	print_log(headers, header=True)

	for epoch in epochs:
		iterations = epoch - prev_epoch  # num of iteration to train
		model.fit_partial(train, epochs=iterations)
		train_precision.append(precision_at_k(model, train, k, user_index))  # p@k on train set only
		all_data_precision.append(precision_at_k(model, all_data, k, user_index))  # p@k on all data
		row = [epoch, train_precision[-1], all_data_precision[-1]]
		print_log(row)
		prev_epoch = epoch
	return model, train_precision, all_data_precision


def learning_curve_auc(model, train, test, epochs=range(2, 40, 2), user_index=None):
	"""
	Helper function to depict the improvement as training iteration goes on at accuracy.

	:param model: (lightFM model)
	:param train: (csr)
	:param test: (csr)
	:param epochs: (iterator)
	:param user_index: (list) index of matrix-split users
	:return:
		model: (lightFM model)
		train_auc: (list)
		test_auc: (list)
	"""
	if not user_index:
		user_index = range(train.shape[0])
	prev_epoch = 0
	train_auc = []
	test_auc = []

	headers = ['epochs', 'auc train', 'auc test']
	print_log(headers, header=True)

	for epoch in epochs:
		iterations = epoch - prev_epoch  # num of iteration to train
		model.fit_partial(train, epochs=iterations)
		train_auc.append(
			auc_mean(model, train, user_index=user_index, mode='train')
		)
		test_auc.append(
			auc_mean(model, train, test, user_index=user_index, mode='test')
		)
		row = [epoch, train_auc[-1], test_auc[-1]]
		print_log(row)
		prev_epoch = epoch
	return model, train_auc, test_auc


def grid_search_learning_curve_predict_k(
		base_model,
		train,
		all_data,
		param_grid,
		user_index=None,
		patk=5,
		epochs=range(2, 40, 2)
	):
	"""
	Inspired (stolen) from sklearn gridsearch
	https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py

	REF: https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/

	:param base_model: (lightFM model)
	:param train: (csr)
	:param all_data: (csr)
	:param param_grid: (dict)
	:param user_index: (list)
	:param patk: (int) top k
	:param epochs: (iterator)
	:return:
		curves: (list of dict) list of each learning curve
	"""
	curves = []
	keys, values = zip(*param_grid.items())
	# list combination of values in each parameter, such as (20, 0.01) (20, 0.05) etc.
	for val in itertools.product(*values):
		params = dict(zip(keys, val))
		this_model = copy.deepcopy(base_model)
		print_line = []
		for k, v in params.items():
			setattr(this_model, k, v)  # set up model's hyper-parameter
			print_line.append((k, v))

		print(' | '.join('{}: {}'.format(k, v) for (k, v) in print_line))
		_, train_patk, all_data_patk = learning_curve_predict_k(
			this_model, train, all_data, epochs, k=patk, user_index=user_index)
		curves.append({
			'params': params,
			'metric': {
				'train': train_patk,
				'all_data': all_data_patk
			}
		})
	return curves


def grid_search_learning_curve_auc(
		base_model,
		train,
		test,
		param_grid,
		user_index=None,
		epochs=range(2, 40, 2)
	):
	"""
	Inspired (stolen) from sklearn gridsearch
	https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py

	:param base_model: (lightFM model)
	:param train: (csr)
	:param test: (csr)
	:param param_grid: (dict)
	:param user_index: (list)
	:param epochs: (iterator)
	:return:
		curves: (list of dict) list of each learning curve
	"""
	curves = []
	keys, values = zip(*param_grid.items())
	# list combination of values in each parameter, such as (20, 0.01) (20, 0.05) etc.
	for val in itertools.product(*values):
		params = dict(zip(keys, val))
		this_model = copy.deepcopy(base_model)
		print_line = []
		for k, v in params.items():
			setattr(this_model, k, v)  # set up model's hyper-parameter
			print_line.append((k, v))

		print(' | '.join('{}: {}'.format(k, v) for (k, v) in print_line))
		_, train_auc, test_auc = learning_curve_auc(
									this_model,
									train,
									test,
									epochs,
									user_index=user_index
									)
		curves.append({
			'params': params,
			'metric': {
				'train': train_auc,
				'test': test_auc
			}
		})
	return curves


def refit_model(base_model, best_params, interact_matrix, best_iters):
	"""
	re-train a model.

	:param base_model: (lightFM model)
	:param best_params: (dict) best hyper-parameters
	:param interact_matrix: (csr)
	:param best_iters: (int) best num of interations
	:return:
		model: (lightFM model)
	"""
	model = copy.deepcopy(base_model)
	for k, v in best_params.items():
		setattr(model, k, v)

	model.fit(interact_matrix, epochs=best_iters)
	return model


def get_items_interact(
		customer_id,
		true_mf,
		users,
		items,
		item_lookup
	):
	"""
	REF: https://jessesw.com/Rec-System/

	This just tells me which items have been already interacted with by a specific user in the interaction matrix.

	@parameters:
		customer_id: (int) Input the customer's id number that you want to see prior purchases of at least once
		true_mf: (csr) The initial ratings matrix used
		users: (numpy array) The array of customers used in the ratings matrix
		items: (numpy array) The array of products used in the ratings matrix
		item_lookup: (dataframe) A simple pandas dataframe of the unique product ID/product descriptions available

	@returns:
		A list of item IDs and item descriptions for a particular customer
		that were already interacted with in the interaction matrix
	"""
	cust_ind = np.where(users == customer_id)[0][
		0]  # Returns the index row of our customer id
	purchased_ind = true_mf[cust_ind, :].nonzero()[
		1]  # Get column indices of purchased items
	prod_codes = items[
		purchased_ind]  # Get the stock codes for our purchased items
	return item_lookup.loc[item_lookup.product_id.isin(prod_codes)]


def top_interact(customer_id, users, interaction_sparse, item_lookup, top_num=3, item_name='title'):
	"""
	Return the top interact items for customer.

	:param customer_id: (int)
	:param users: (list)
	:param interaction_sparse: (csr)
	:param item_lookup: (list)
	:param top_num: (int) top number of interacted product from customer
	:param item_name: (str) column for product name
	:return:
		 (customer_id, top_interact_items): (tuple)
	"""
	# Find out the top three items with most interactions.
	cust_ind = np.where(users == customer_id)[0][0]
	interaction_nd = np.squeeze(interaction_sparse[cust_ind, :].toarray(), axis=0)  # convert to ndarray for `squeeze`
	sort_item_idx = np.argsort(interaction_nd)[::-1]
	item_idx_value = interaction_nd[sort_item_idx]
	top_item_idx = sort_item_idx[item_idx_value > 0][:top_num]

	top_interact_items = []
	for idx in top_item_idx:
		top_interact_items.append(item_lookup.loc[idx][item_name])

	# Fill in `None` if top_interact_items is less than top_num
	if top_num - len(top_interact_items) > 0:
		fill_num = top_num - len(top_interact_items)
		fill_none = [None for _ in range(fill_num)]
		top_interact_items.extend(fill_none)

	return (customer_id, top_interact_items)


def rec_items(
		customer_id,
		exclude_mf,
		user_vecs,
		item_vecs,
		users,
		items,
		item_lookup,
		num_items=10,
		item_name='title',
		item_col='product_id'
	):
	"""
	REF: https://jessesw.com/Rec-System/

	This function will return the top recommended items to our users

	@parameters:
		customer_id: (int) Input the customer's id number that you want to get recommendations for
		exclude_mf: (csr) The interaction matrix you use to exclude already purchased/interacted items
		user_vecs - the user vectors from your fitted matrix factorization
		item_vecs - the item vectors from your fitted matrix factorization
		users: (numpy array) an array of the customer's ID numbers that make up the rows of your ratings matrix
			(in order of matrix)
		items: (numpy array) an array of the products that make up the columns of your ratings matrix
			(in order of matrix)
		item_lookup - A simple pandas dataframe of the unique product ID/product descriptions available
		num_items: (int) The number of items you want to recommend in order of best recommendations. Default is 10.
		item_name: (str) column for product name
		item_col: (str) column for product id

	@returns:
		The top n recommendations chosen based on the user/item vectors for items never interacted with
	"""

	cust_ind = np.where(users == customer_id)[0][
		0]  # Returns the index row of our customer id
	pref_vec = exclude_mf[cust_ind, :].toarray(
	)  # Get the ratings from the ratings matrix
	pref_vec = pref_vec.reshape(-1) + 1  # Add 1 to everything, so that items not purchased yet become equal to 1
	pref_vec[pref_vec > 1] = 0  # Make everything already purchased zero
	rec_vector = user_vecs[cust_ind, :].dot(
		item_vecs.T)  # Get dot product of user vector and all item vectors

	# Scale this recommendation vector between 0 and 1
	min_max = MinMaxScaler()
	rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]
	recommend_vector = pref_vec * rec_vector_scaled
	# Items already purchased have their recommendation multiplied by zero
	product_idx = np.argsort(
		recommend_vector
	)[::-1][:num_items]  # Sort the indices of the items into order
	# of best recommendations
	rec_list = []  # start empty list to store items
	for index in product_idx:
		code = items[index]
		rec_list.append([
			customer_id,
			code,
			item_lookup[item_name].loc[item_lookup[item_col] == code].iloc[0],
			recommend_vector[index]
		])
	# Append our descriptions to the list
	customer_ids = [item[0] for item in rec_list]
	codes = [item[1] for item in rec_list]
	descriptions = [item[2] for item in rec_list]
	scores = [item[3] for item in rec_list]
	final_frame = pd.DataFrame({
		'member_id':customer_ids,
		'product_id': codes,
		'title': descriptions,
		'score': scores
	})  # Create a dataframe
	return final_frame


def random_rec(
		customer_id,
		cust_product_click,
		group_data,
		item_freq,
		item_lookup,
		num_reco=10,
		user_col='member_id',
		item_col='product_id',
		item_name='title'

	):
	"""
	Randomly sample recommendation list, excluding the items they have clicked/viewed.
	The sampling distribution of items follows the frequency of items being viewed by users.

	:param customer_id: (int)
	:param cust_product_click: (series) index of customer with number of product they've clicked
	:param group_data: (df) group of (user, item, interaction)
	:param item_freq: (series) index of item with its frequency
	:param item_lookup: (list)
	:param num_reco: (int) number of recommendations
	:param user_col: (str)
	:param item_col: (str)
	:param item_name: (str)
	:return:
		rec_df: (df) Recommendation list for a customer
	"""
	num_clicked = cust_product_click.loc[customer_id]
	clicked_items = list(group_data[group_data[user_col] == customer_id][item_col])
	num_total = num_clicked + num_reco
	num_item = item_freq.shape[0]
	if num_total > num_item:
		recommend_items = [x for x in item_freq.index.values if x not in clicked_items]
		fill_num = num_reco - len(recommend_items)
		fill_items = np.random.choice(a=item_freq.index.values, size=fill_num, replace=False, p=item_freq)
		recommend_items.extend(fill_items)
		ranked_items = item_freq[item_freq.index.isin(recommend_items)].sort_values().index.values[:num_reco]
	else:
		sample_items = np.random.choice(a=item_freq.index.values, size=num_total, replace=False, p=item_freq)
		recommend_items = [x for x in sample_items if x not in clicked_items]
		ranked_items = item_freq[item_freq.index.isin(recommend_items)].sort_values().index.values[:num_reco]

	title_lookup = item_lookup.set_index(item_col)
	rank_title = list(title_lookup.loc[ranked_items][item_name])

	rec_df = pd.DataFrame({
		'member_id': [customer_id for _ in range(len(ranked_items))],
		'product_id': ranked_items,
		'title': rank_title,
		'rank': [x for x in range(len(ranked_items))]})

	return rec_df


def get_similar_items(
		item_id,
		items,
		item_vecs,
		item_lookup,
		item_col='product_id',
		item_name='title'
	):
	"""
	REF: https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe

	Get most similar items. The first most similar item will always be itself.

	:param item_id: (int) product_id
	:param items: (numpy array) array of product_id
	:param item_vecs: (numpy array)
	:param item_lookup: (df)
	:param item_col: (str)
	:param item_name: (str)
	:return:
		similar_df: (df)
	"""
	# Get the item row
	item_row_index = np.where(items == item_id)[0][0]
	item_vec = item_vecs[item_row_index].T

	# Calculate the similarity score
	# and select the top 10 most similar.
	scores = item_vecs.dot(item_vec)
	top_10_indices = np.argsort(scores)[::-1][:10]

	similar_item_ids = []
	similar_items = []
	item_scores = []

	# Get and print the actual item names and scores
	for idx in top_10_indices:
		similar_item_ids.append(item_lookup[item_col].loc[item_lookup[item_col] == items[idx]].iloc[0])
		similar_items.append(item_lookup[item_name].loc[item_lookup[item_col] == items[idx]].iloc[0])
		item_scores.append(scores[idx])

	similar_df = pd.DataFrame({
		'product_id': similar_item_ids,
		'title': similar_items,
		'score': item_scores})
	return similar_df


def item_similarity(items, item_interaction_sparse):
	"""
	REF: https://medium.com/@wwwbbb8510/comparison-of-user-based-and-item-based-collaborative-filtering-f58a1c8a3f1d
	REF: http://kevingo.logdown.com/posts/245762-item-based-collaborative-filtering-recommendation-method

	Return symmetric matrix for item similarity based on cosine similarity.

	:param items: (numpy array) item_ids
	:param item_interaction_sparse: (csr) (row: item, col: user) sparse matrix
	:return:
		item_similar: (numpy array)
	"""
	# Identity matrix.
	item_num = len(items)
	item_similar = np.identity(item_num)

	# Fill in the blank.
	for i in tqdm_notebook(range(item_num)):
		for j in range(item_num):
			if j > i:
				indices_i = set(item_interaction_sparse[i, :].nonzero()[1])
				indices_j = set(item_interaction_sparse[j, :].nonzero()[1])
				# Non-zero entries at both items.
				indices_intersect = list(indices_i.intersection(indices_j))
				if len(indices_intersect) == 0:
					# print('{},{} items no intersected values...'.format(i,j))
					item_similar[i, j] = 0  # if no interacted values, return 0
					item_similar[j, i] = 0
				else:
					data_i = item_interaction_sparse[i, indices_intersect].toarray()[0]
					data_j = item_interaction_sparse[j, indices_intersect].toarray()[0]
					sim_score = 1 - spatial.distance.cosine(data_i, data_j)
					item_similar[i, j] = sim_score
					item_similar[j, i] = sim_score
	return item_similar


def fill_unrate(user_interaction_sparse, users, items, item_similar):
	"""
	REF: https://medium.com/@wwwbbb8510/comparison-of-user-based-and-item-based-collaborative-filtering-f58a1c8a3f1d
	REF: http://kevingo.logdown.com/posts/245762-item-based-collaborative-filtering-recommendation-method

	Fill in (row: user, col: item) interaction matrix with predicted values on missing ratings.

	:param user_interaction_sparse: (csr)
	:param users: (numpy array) user_ids
	:param items: (numpy array) item_ids
	:param item_similar: (numpy array) item similarity matrix
	:return:
		user_interaction_spare_filled: (csr) filled with predicted rating
	"""
	user_interaction_sparse_filled = copy.deepcopy(user_interaction_sparse).toarray()  # to ndarray for efficiency

	users_num = len(users)
	items_num = len(items)
	item_indices_set = set(list(range(items_num)))

	for member_index in tqdm_notebook(range(users_num)):
		zero_indices = list(
			item_indices_set.difference(
				set(user_interaction_sparse[member_index, :].nonzero()[1])))
		for unrate_index in zero_indices:
			rated_indices = user_interaction_sparse[member_index, :].nonzero()[1]
			S = item_similar[unrate_index, rated_indices]
			abs_S = list(map(abs, S))
			R = user_interaction_sparse[member_index, rated_indices].toarray()[0]
			numerator = np.sum(np.multiply(R, S))
			denominator = np.sum(abs_S)
			if denominator == 0:
				user_interaction_sparse_filled[
					member_index, unrate_index] = numerator / 1.
			else:
				user_interaction_sparse_filled[
					member_index, unrate_index] = numerator / denominator
	return sparse.csr_matrix(user_interaction_sparse_filled)


def item_based_predict_at_k(filled_ratings, true_ratings, k=5, user_index=None):
	"""
	Compute the percentage of top k items predicted by item-based matrix are actually included in
	original non-zero ratings entries.

	REF: https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/

	:param filled_ratings: (csr) item-based matrix filled with predicted values
	:param true_ratings: (csr) interaction matrix
	:param k: (int) top k preferable items
	:param user_index: (list) index of matrix-split users
	:return:
		precisions: (float) mean top k precision for all users in user_index
	"""
	if not user_index:
		user_index = range(true_ratings.shape[0])
	precisions = []
	for user in user_index:
		# In case of large dataset, compute predictions row-by-row like below
		predictions = filled_ratings[user, :].toarray()[0]  # preference prediction for all items
		top_k = np.argsort(-predictions)[:k]  # top k item index
		labels = true_ratings.getrow(user).indices  # non-zero index
		precision = float(len(set(top_k) & set(labels))) / float(k)
		precisions.append(precision)
	return np.mean(precisions)


def item_based_auc_mean(filled_ratings, train, test=None, user_index=None, mode='train'):
	"""
	Compute mean accuracy.
	In train mode, only entries in train are used, while in test mode,
	entries excluding non-zero train entries are used.

	:param filled_ratings: (csr) item-based matrix filled with predicted values
	:param train: (csr)
	:param test: (csr)
	:param user_index: (list) index of matrix-split users
	:param mode: (string) train/test mode
	:return:
		auc: (float) mean of accuracy for all users in user_index
	"""
	if not user_index:
		user_index = range(train.shape[0])
	auc = []
	for user in user_index:
		preds = filled_ratings[user, :].toarray()[0]  # preference prediction for all items
		if mode == 'train':
			col_indices = list(range(train.shape[1]))  # entries from train matrix
			actual = train[user, col_indices].toarray()[0]  # list of list
			actual[actual > 0] = 1
			pred = preds[col_indices]
		elif mode == 'test':
			col_indices = set(range(train.shape[1])).difference(
				set(train.getrow(user).indices))  # entries excluding train matrix
			col_indices = list(col_indices)
			actual = test[user, col_indices].toarray()[0]  # list of list
			actual[actual > 0] = 1
			pred = preds[col_indices]
		fpr, tpr, thresholds = metrics.roc_curve(actual, pred)
		auc.append(metrics.auc(fpr, tpr))

	return np.mean(auc)


def item_based_rec_items(
		customer_id,
		exclude_mf,
		filled_ratings,
		users,
		items,
		item_lookup,
		num_items=10,
		item_col='product_id',
		item_name='title'
	):
	"""
	REF: https://jessesw.com/Rec-System/

	This function will return the top recommended items to our users

	@parameters:
		customer_id: (int) Input the customer's id number that you want to get recommendations for
		exclude_mf: (csr) The interaction matrix you use to exclude already purchased/interacted items
		filled_ratings: (csr) item-based matrix filled with predicted values
		users: (numpy array) an array of the customer's ID numbers that make up the rows of your ratings matrix
			(in order of matrix)
		items: (numpy array) an array of the products that make up the columns of your ratings matrix
			(in order of matrix)
		item_lookup - A simple pandas dataframe of the unique product ID/product descriptions available
		num_items: (int) The number of items you want to recommend in order of best recommendations. Default is 10.
		item_col: (str)
		item_name: (str)

	@returns:
		The top n recommendations chosen based on the user/item vectors for items never interacted with
	"""

	cust_ind = np.where(users == customer_id)[0][
		0]  # Returns the index row of our customer id
	pref_vec = exclude_mf[cust_ind, :].toarray(
	)  # Get the ratings from the ratings matrix
	pref_vec = pref_vec.reshape(-1) + 1  # Add 1 to everything, so that items not purchased yet become equal to 1
	pref_vec[pref_vec > 1] = 0  # Make everything already purchased zero
	rec_vector = filled_ratings[cust_ind, :].toarray()

	# Scale this recommendation vector between 0 and 1
	min_max = MinMaxScaler()
	rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]
	# Items already purchased have their recommendation multiplied by zero
	recommend_vector = pref_vec * rec_vector_scaled
	product_idx = np.argsort(
		recommend_vector
	)[::-1][:num_items]  # Sort the indices of the items into order
	# of best recommendations
	rec_list = []  # start empty list to store items
	for index in product_idx:
		code = items[index]
		rec_list.append([
			code, item_lookup[item_name].loc[item_lookup[item_col] == code].iloc[0]
		])
	# Append our descriptions to the list
	codes = [item[0] for item in rec_list]
	descriptions = [item[1] for item in rec_list]
	final_frame = pd.DataFrame({
		'product_id': codes,
		'title': descriptions
	})  # Create a dataframe
	return final_frame[['product_id', 'title']]  # Switch order of columns around
