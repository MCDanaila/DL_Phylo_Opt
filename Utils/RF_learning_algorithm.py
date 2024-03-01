import math
import os
import numpy as np
import pandas as pd
from .defs_PhyAI import FEATURES, FEATURES_PRUNE, FEATURES_RGFT, GROUP_ID, KFOLD, LABEL, N_ESTIMATORS, OPT_TYPE, PHYML_TREE_FILENAME, SUBTREE2, SUMMARY_PER_DS, TREES_PER_DS, SEP
from ete3 import Tree
from collections import OrderedDict
from .tree_functions import get_total_branch_lengths

from sklearn.ensemble import RandomForestRegressor
from statistics import mean, median


def score_rank(df_by_ds, sortby, locatein, random, scale_score):
	'''
	find the best tree in 'sortby' (e.g., predicted as the best) foreach dataset and locate its rank in 'locatein' (e.g., y_test)
	'''
	print(df_by_ds.columns)
	best_pred_ix = df_by_ds[sortby].idxmax()    # changed min to max!
	print("best_pred_ix === ", best_pred_ix)
	if random:
		best_pred_ix = np.random.choice(df_by_ds[sortby].index, 1, replace=False)[0]
	temp_df = df_by_ds.sort_values(by=locatein, ascending=False).reset_index()   # changed ascending to False
	print('temp_df ====== ', temp_df)
	best_pred_rank = min(temp_df.index[temp_df["index"] == best_pred_ix].tolist())
	best_pred_rank += 1  # convert from pythonic index to position
	
	if scale_score:
		best_pred_rank /= len(df_by_ds[sortby].index)   # scale the rank according to rankmax
		best_pred_rank *= 100


	return best_pred_rank


def ds_scores(df, move_type, random, scale_score):
	rank_pred_by_ds, rank_test_by_ds = {}, {}

	label = LABEL.format(move_type)
	sp_corrs = []
	grouped_df_by_ds = df.groupby(FEATURES[GROUP_ID], sort=False)
	for group_id, df_by_ds in grouped_df_by_ds:
		rank_pred_by_ds[group_id] = score_rank(df_by_ds, "pred", label, random, scale_score)
		rank_test_by_ds[group_id] = score_rank(df_by_ds, label, "pred", random, scale_score)
		
		temp_df = df_by_ds[[label, "pred"]]
		sp_corr = temp_df.corr(method='spearman').ix[1,0]
		if sp_corr:
			sp_corrs.append(sp_corr)
		else:
			sp_corrs.append(None)
	
	return rank_pred_by_ds, rank_test_by_ds, sp_corrs


def split_features_label(df, move_type, features):
	attributes_df = df[features].reset_index(drop=True)
	label_df = df[LABEL.format(move_type)].reset_index(drop=True)

	x = np.array(attributes_df)
	y = np.array(label_df).ravel()

	return x, y


def apply_RFR(df_test, df_train, move_type, features):
	X_train, y_train = split_features_label(df_train, move_type, features)
	X_test, y_test = split_features_label(df_test, move_type, features)

	regressor = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_features=0.33,  oob_score=True).fit(X_train, y_train) # 0.33=nfeatures/3. this is like in R (instead of default=n_features)
	y_pred = regressor.predict(X_test)
	oob = regressor.oob_score_
	f_imp = regressor.feature_importances_

	return y_pred, oob, f_imp


def truncate(df):
	df = df.dropna()
	groups_ids = df[FEATURES[GROUP_ID]].unique()
	kfold = len(groups_ids) if KFOLD=="LOO" else KFOLD
	assert len(groups_ids) >= kfold
	ndel = len(groups_ids) % kfold
	if ndel != 0:   # i removed datasets from the end, and not randomly. from some reason..
		for group_id in groups_ids[:-ndel-1:-1]:
			df = df[df[FEATURES[GROUP_ID]] != group_id]

	groups_ids = df[FEATURES[GROUP_ID]].unique()
	new_length = len(groups_ids) - ndel
	test_batch_size = int(new_length / kfold)
	print(kfold, ndel, groups_ids, new_length, test_batch_size )
	return df.reset_index(drop=True), groups_ids, test_batch_size


def cross_validation_RF(df, move_type, features, trans=False, validation_set=False, random=False, scale_score=False):
	df, groups_ids, test_batch_size = truncate(df)
	res_dict = {}
	oobs, f_imps, = [], []
	my_y_pred, imps = np.full(len(df), np.nan), np.full(len(df), np.nan)
	
	if not validation_set:
		if test_batch_size != 0:
			for low_i in groups_ids[::test_batch_size]:
				low_i, = np.where(groups_ids == low_i)
				low_i = int(low_i)
				up_i = low_i + test_batch_size
		
				test_ixs = groups_ids[low_i:up_i]
				train_ixs = np.setdiff1d(groups_ids, test_ixs)
				df_test = df.loc[df[FEATURES[GROUP_ID]].isin(test_ixs)]
				df_train = df.loc[df[FEATURES[GROUP_ID]].isin(train_ixs)]
		
				y_pred, oob, f_imp = apply_RFR(df_test, df_train, move_type, features)
		
				oobs.append(oob)
				f_imps.append(f_imp)
				my_y_pred[df_test.index.values] = y_pred       # sort the predictions into a vector sorted according to the respective dataset
			
		df["pred"] = my_y_pred
	
	else:     # namely if validation set strategy, and not cross validation
		df_train = df
		df_test = pd.read_csv(VALSET_FEATURES_LABEL)
		df_test = fit_transformation(df_test, move_type, trans).dropna()
		y_pred, oob, f_imp = apply_RFR(df_test, df_train, move_type, features)
		
		oobs.append(oob)
		f_imps.append(f_imp)
		df_test["pred"] = y_pred  # the predictions vec is the same lengths of test set
		df = df_test
	
	print(df["pred"])
	
	rank_pred_by_ds, rank_test_by_ds, corrs = ds_scores(df, move_type, random, scale_score)

	# averaged over cv iterations
	res_dict['oob'] = sum(oobs) / len(oobs)
	res_dict['f_importance'] = sum(f_imps) / len(f_imps)
	# foreach dataset (namely arrays are of lengths len(sampled_datasets)
	res_dict["rank_first_pred"] = rank_pred_by_ds
	res_dict["rank_first_true"] = rank_test_by_ds
	res_dict["spearman_corr"] = corrs
	
	
	return res_dict, df


def fit_transformation(df, move_type, trans=False):
	groups_ids = df[FEATURES[GROUP_ID]].unique()
	for group_id in groups_ids:
		scaling_factor = df[df[FEATURES[GROUP_ID]] == group_id]["orig_ds_ll"].iloc[0]
		df.loc[df[FEATURES[GROUP_ID]] == group_id, LABEL.format(move_type)] /= -scaling_factor    # todo: make sure I run it with minus/abs to preserve order. also change 'ascending' to True in 'get_cumsun_preds' function
	
	if trans:
		df[LABEL.format(move_type)] = np.exp2(df[LABEL.format(move_type)]+1)
	
	return df


def parse_relevant_summaries_for_learning(datapath, outpath, move_type):
	columns = ["prune_name", "rgft_name", "orig_ds_ll", "ll", "time", "d_ll_prune"]
	columns += FEATURES_PRUNE if move_type == "prune" else FEATURES_RGFT
	df = pd.DataFrame(columns=columns)
	for i,relpath in enumerate(os.listdir(datapath)):
		if os.path.isdir(os.path.join(datapath, relpath)):
			#print(i, relpath)
			ds_path = datapath + relpath + SEP
			'''user_tree_file = os.path.join(ds_path, tree_file.format(relpath))
			with open(user_tree_file) as fpr:
				ds_tbl = get_total_branch_lengths(fpr.read().strip()) '''
			summary_per_ds = SUMMARY_PER_DS.format(ds_path, move_type)
			if os.path.exists(summary_per_ds) and FEATURES["bl"] in pd.read_csv(summary_per_ds).columns:
				df_ds = pd.read_csv(summary_per_ds, index_col='iteration')
				#df_ds.insert(1, "path", datapath)
				df_ds[FEATURES[GROUP_ID]] = str(i)
				#!added in simulate_SPR df_ds[FEATURES["group_tbl"]] = ds_tbl
				#print(df_ds)
				df = pd.concat([df, df_ds], join='inner', axis=0)
	df.to_csv(outpath, index_label='iteration')
	
	
def print_and_index_results(df_datasets, res_dict, features):
	
	#### score 1 ####
	spearman_corrs = res_dict['spearman_corr']
	df_datasets['corr'] = spearman_corrs
	print("\nsapearman corr:\n" + "mean:", mean([e for e in spearman_corrs if not math.isnan(e)]), ", median:",median(spearman_corrs))
	
	#### score 2 + 3 ####
	res_vec1 = np.asarray(list(res_dict['rank_first_pred'].values())) if type(res_dict['rank_first_pred']) is dict else res_dict['rank_first_pred']
	res_vec2 = np.asarray(list(res_dict['rank_first_true'].values()))  if type(res_dict['rank_first_true']) is dict else res_dict['rank_first_true']
	df_datasets['best_predicted_ranking'] = res_vec1
	df_datasets['best_empirically_ranking'] = res_vec2
	print("\nbest predicted rank in true:\n" + "mean:",np.mean(res_vec1), ", median:", np.median(res_vec1))
	print("\nbest true rank in pred :\n" + "mean:",np.mean(res_vec2), ", median:", np.median(res_vec2))
	
	mean_importances = res_dict['f_importance']   # index in first row only (score foreach run and not foreach dataset)
	for i, f in enumerate(features):
		colname = "imp_" + f
		df_datasets.loc[0, colname] = mean_importances[i]

	#### additional information ####
	df_datasets.loc[0, 'oob'] = res_dict['oob']   # index in first row only (score foreach run and not foreach dataset)
	print("oob:", res_dict['oob'])
	print("ndatasets: ", len(res_vec1))
	

	return df_datasets


def sort_features(res_dict, features):
	feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(features, res_dict['f_importance'])]
	feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)  # most important first
	sorted_importances = [importance[1] for importance in feature_importances]
	sorted_features = [importance[0] for importance in feature_importances]
	
	return sorted_importances, sorted_features
	
	
def extract_scores_dict(res_dict, df_with_scores):
	res_dict['rank_first_pred'], res_dict["rank_first_true"] = df_with_scores['best_predicted_ranking'].values, df_with_scores['best_empirically_ranking'].values
	res_dict['spearman_corr'], res_dict['%neighbors'], res_dict['oob'] = df_with_scores['corr'].values, df_with_scores['required_evaluations_0.95'].values, df_with_scores.loc[0, 'oob']
	res_dict['f_importance'] = df_with_scores.loc[0, df_with_scores.columns[pd.Series(df_with_scores.columns).str.startswith('imp_')]].values

	return res_dict
