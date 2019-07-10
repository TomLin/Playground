import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from IPython.display import display
import numpy as np
import pandas as pd


def plot_yaxis_with_break(
		df,
		lower_ylim=(0,2300000),
		upper_ylim=(2800000, 3900000),
		title='Weekly Sales over Store'):
	"""
	Plot y axis with breaks when there exist outliers with very large y axis value.

	:param df: dataframe
	:param lower_ylim (tuple): y axis range on the lower subplot
	:param upper_ylim (tuple): y axis range on the upper subplot -> y axis break for outliers
	:param title (string): chart title
	:return:
		subplots
	"""

	# line plot
	n = 1 # ratio of upper yaxis
	m = 3 # ratio of lower yaxis
	sns.set_style('whitegrid')
	fig, ax = plt.subplots(2, 1, figsize=(20, 10), sharex=True, gridspec_kw={'height_ratios': [n, m]})
	ax = ax.ravel()
	_ = df.plot(kind='line', legend=False, ax=ax[0], colormap='tab20c')
	_ = ax[0].ticklabel_format(axis='y', style='plain')  # supress scientific notation in plot
	_ = ax[0].set_ylim(upper_ylim) # limit for upper yaxis
	_ = df.plot(kind='line', legend=False, ax=ax[1], colormap='tab20c')
	_ = ax[1].ticklabel_format(axis='y', style='plain')  # supress scientific notation in plot
	_ = ax[1].set_ylim(lower_ylim) # limit for lower yaxis

	ax[0].spines['bottom'].set_visible(False)
	ax[1].spines['top'].set_visible(False)
	ax[0].xaxis.tick_top()
	ax[0].tick_params(labeltop='off')  # don't put tick labels at the top
	ax[1].xaxis.tick_bottom()

	d = .015  # how big to make the diagonal lines in axes coordinates
	# arguments to pass to plot, just so we don't keep repeating them
	kwargs = dict(transform=ax[0].transAxes, color='k', clip_on=False)
	_ = ax[0].plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
	_ = ax[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
	kwargs.update(transform=ax[1].transAxes)  # switch to the bottom axes
	_ = ax[1].plot((-d, +d), (1 - d / 3, 1 + d / 3), **kwargs)  # bottom-left diagonal
	_ = ax[1].plot((1 - d, 1 + d), (1 - d / 3, 1 + d / 3), **kwargs)  # bottom-right diagonal

	_ = ax[1].set_ylabel('weekly sales', fontsize='x-large')
	_ = ax[1].set_xlabel('date', fontsize='x-large')
	_ = ax[0].set_title('{}'.format(title), fontsize='xx-large')

	plt.show()


def plot_zoom(df, zoom=(-0.3, 0.3)):
	"""
	Plot two subplots, the upper one shows full range of y axis,
	the lower one shows the zoomed in y axis specified in the argument.

	:param df: dataframe
	:param zoom (tuple): the zoomed in y axis with specified upper band and lower band
	:return:
		subplots
	"""
	sns.set_style('whitegrid')
	fig, ax = plt.subplots(2, 1, figsize=(30, 15), sharex=True)
	ax = ax.ravel()
	_ = df.plot(kind='line', legend=False, ax=ax[0], colormap='tab20c')
	_ = ax[0].set_title('Grow Rate', fontsize='xx-large')

	_ = df.plot(kind='line', legend=False, ax=ax[1], colormap='tab20c')
	_ = ax[1].set_ylim(zoom)
	_ = ax[1].set_title('Grow Rate (Zoom in)', fontsize='xx-large')
	_ = ax[1].set_xlabel('date', fontsize='x-large')

	plt.show()


def consecutive_series(df, col, day_width = 7):
	"""
	Create a time series with complete, consecutive weekday.
	The min and max of date is set based on the min and max date from the given df.

	:param df: dataframe
	:param col (string): column name of the time series
	:param day_width (int): the period interval of the time series
	:return:
		dataframe of the time series
	"""

	min_date = df[col].min()
	max_date = df[col].max()

	days_series = list()
	days_series.append(min_date)

	temp_date = min_date
	while temp_date < max_date:
		temp_date = temp_date + timedelta(days=day_width)
		days_series.append(temp_date)

	days_series = pd.DataFrame({'date': days_series})
	display(days_series.head()) # print the preview of result

	return days_series

def cross_join_key_series(df, key, series):
	"""
	Cross join the key from df to the series.

	:param df: dataframe
	:param key (string): column name
	:param series (datafra	me): the series to cross join to
	:return:
		cross-joined dataframe
	"""

	unique_key = df[key].unique()
	num_key = len(unique_key) # total number of unique key

	final_df =pd.DataFrame()

	for i in range(num_key):
		temp_df = series
		temp_df[key] = unique_key[i]
		final_df = final_df.append(temp_df)

	display(final_df.head()) # print the preview of result

	return final_df


def create_lag_sales(df, groupby_col, shift_col, shift_num=1):
	"""
	Create shift column feature.

	:param df: dataframe
	:param groupby_col (list of groupby columns):
	:param shift_col (string): the column being shifted
	:param shift_num (int): the times shifted
	:return:
		single column dataframe of the shifted column
	"""

	col_previous = df.groupby(groupby_col)[shift_col].transform(lambda x: x.shift(shift_num))
	col_previous.rename('{}_lag_{}'.format(shift_col,shift_num), inplace=True)
	display(col_previous.head()) # preview of the result

	return col_previous










