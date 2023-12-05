
from __future__ import division, print_function
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import gc
import pickle
import warnings
from itertools import product
from functools import reduce

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import lightgbm as lgb

from ..preprocess.preprocess import bcolors, color_print, initializer, nullTreatment, outlierTreatment, logTransform, normalization
# from ..preprocess.preprocess import environment
from ..select_feature.select_feature import selectFeature, fit_LogisticRegression
from ..fit_model.fit_model import fit_classifier, leaderboard
from ..fit_model.performance import performance

import time


class pipeline:

	def __init__(self, filename = 'automl_result.pkl', models = ['rf','xgb','lgb','mlp']):
		self.models_ = models
		self.filename_ = filename

	def fit(self, train, valid):

		start = time.time()

		### PREPROCESS
		color_print("== Preprocessing ==", "OKGREEN", bBold=True)

		# set environment
		# environment()

		# initialize
		train = initializer(train)
		valid = initializer(valid)
		color_print("     Complete : Change column names in lower case", "OKBLUE", bBold=True)

		# Null value treatment
		pre_null = nullTreatment()
		pre_null.fit(train)
		train = pre_null.transform(train)
		valid = pre_null.transform(valid)
		color_print("     Complete : Fill Null data with median", "OKBLUE", bBold=True)

		# OutlierTreatment
		pre_outlier = outlierTreatment()
		pre_outlier.fit(train)
		train = pre_outlier.transform(train)
		valid = pre_outlier.transform(valid)
		color_print("     Complete : Outlier treatment", "OKBLUE", bBold=True)

		# Log transform
		pre_log = logTransform()
		pre_log.fit(train)
		train = pre_log.transform(train)
		valid = pre_log.transform(valid)
		color_print("     Complete : Log transform with skewed features", "OKBLUE", bBold=True)

		# Normalization
		pre_norm = normalization()
		pre_norm.fit(train)
		train, _ = pre_norm.transform(train)
		valid, _ = pre_norm.transform(valid)
		color_print("     Complete : Normalization", "OKBLUE", bBold=True)

		# FEATURE SELECTION
		color_print("== Feature Selection ==", "OKGREEN", bBold=True)
		f_select = selectFeature()
		f_select.fit_IV(train)
		f_select.fit_corr(train)
		f_select.fit_feature_selection(train)
		color_print("     Complete : Removed features with low IV values", "OKBLUE", bBold=True)
		color_print("     Complete : Removed feature with high correlation", "OKBLUE", bBold=True)

		# fit logistic regression and select features

		selected_features = f_select.selected_features_
		fit_logit = fit_LogisticRegression()
		fit_logit.fit_binning(train[selected_features], train.target)
		fit_logit.fit(train[selected_features], train.target, n_features=30)
		color_print("     Complete : Fit Logistic Regression for feature selection", "OKBLUE", bBold=True)
		selected_features = fit_logit.selected_features_

		# visualize features

		f_select.vis_lift(selected_features = selected_features)
		color_print("     Complete : Visualize (Lift) ", "OKBLUE", bBold=True)
		f_select.vis_hist(train, selected_features = selected_features)
		color_print("     Complete : Visualize (Hist) ", "OKBLUE", bBold=True)

		# fit models

		color_print("== Model Fitting ==", "OKGREEN", bBold=True)
		fit_leaderboard = leaderboard()
		list_model = self.models_
		fit_leaderboard.fit(train[selected_features], valid[selected_features], train.target, valid.target, model = list_model)

		end = time.time()
		exetime = end - start
		exetime_hrs = (end - start) / 60

		self.nullTreatment_ = pre_null
		self.outlierTreatment_ = pre_outlier
		self.logTransform_ = pre_log
		self.normalization_ = pre_norm
		self.featureSelection_ = f_select
		self.logit_ = fit_logit
		self.leaderboard_ = fit_leaderboard

		filename = self.filename_
		with open(filename, 'wb') as f:
			pickle.dump(self, f)
		
		color_print("\n== Finished, Spent {:.1f} Mins==\n".format(exetime_hrs), "OKGREEN", bBold=True)
		return self


