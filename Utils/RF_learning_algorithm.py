import math
import os
import numpy as np
import pandas as pd
from .defs_PhyAI import FEATURES, GROUP_ID, KFOLD, LABEL, N_ESTIMATORS, OPT_TYPE, PHYML_TREE_FILENAME, SUBTREE2, SUMMARY_PER_DS, TREES_PER_DS
from ete3 import Tree
from collections import OrderedDict
from .tree_functions import get_total_branch_lengths

from sklearn.ensemble import RandomForestRegressor
from statistics import mean, median


def score_rank(df_by_ds, sortby, locatein, random, scale_score):
	'''
	find the best tree in 'sortby' (e.g., predicted as the best) foreach dataset and locate its rank in 'locatein' (e.g., y_test)
	'''

	best_pred_ix = df_by_ds[sortby].idxmax()    # changed min to max!
	if random:
		best_pred_ix = np.random.choice(df_by_ds[sortby].index, 1, replace=False)[0]
	temp_df = df_by_ds.sort_values(by=locatein, ascending=False).reset_index()   # changed ascending to False
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

	return df.reset_index(drop=True), groups_ids, test_batch_size


def cross_validation_RF(df, move_type, features, trans=False, validation_set=False, random=False, scale_score=False):
	df, groups_ids, test_batch_size = truncate(df)
	res_dict = {}
	oobs, f_imps, = [], []
	my_y_pred, imps = np.full(len(df), np.nan), np.full(len(df), np.nan)
	
	if not validation_set:
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


def parse_relevant_summaries_for_learning(datapath, outpath, move_type, step_number, tree_type='bionj'):
	# TO DO rivisitare tutti i datapath in modo da avere una situazione uniforme ovunque
	print(datapath)
	print(outpath)
	for i,relpath in enumerate(os.listdir(datapath)):
		if i ==0:
			ds_path_init = datapath+relpath
			print(i, relpath)
			cols = list(pd.read_csv(SUMMARY_PER_DS.format(ds_path_init, move_type)))
			cols.insert(1, "path")
			cols.extend([FEATURES[GROUP_ID], FEATURES["group_tbl"]])
			df = pd.DataFrame(index=np.arange(0), columns=cols)
			
		ds_path = datapath + relpath
		ds_tbl = get_total_branch_lengths(ds_path + PHYML_TREE_FILENAME.format(tree_type))
		summary_per_ds = SUMMARY_PER_DS.format(ds_path, move_type)
		print(summary_per_ds)
		
		if os.path.exists(summary_per_ds) and FEATURES["bl"] in pd.read_csv(summary_per_ds).columns:
			df_ds = pd.read_csv(summary_per_ds)
			df_ds.insert(1, "path", ds_path)
			df_ds[FEATURES[GROUP_ID]] = str(i)
			df_ds[FEATURES["group_tbl"]] = ds_tbl
			df = pd.concat([df, df_ds], ignore_index=True)
	
	df.to_csv(outpath)
	
	
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

	parser = argparse.ArgumentParser(description='arrange data for learning and implement learning algo')
	parser.add_argument('--validation_set', '-val', default=False, action='store_true') # whether to use validation set INSTEAD of cross validation
	parser.add_argument('--transform_target', '-trans', default=False, action='store_true')
	args = parser.parse_args()

	move_type, st = "merged", "1"
	
	df_path = PROJECT_PATH + LEARNING_DATA.format("all_moves", st)
	df_prune_features = PROJECT_PATH + LEARNING_DATA.format("all_moves_prune", st)
	df_rgft_features = PROJECT_PATH + LEARNING_DATA.format("all_moves_rgft", st)

	if not os.path.exists(df_path):
		parse_relevant_summaries_for_learning(DATA_PATH, df_prune_features, "prune", st,)
		parse_relevant_summaries_for_learning(DATA_PATH, df_rgft_features, "rgft", st)
		shared_cols = FEATURES_SHARED + ["path","prune_name","rgft_name","orig_ds_ll", "ll"]
		complete_df = pd.read_csv(df_prune_features, dtype=types_dict).merge(pd.read_csv(df_rgft_features, dtype=types_dict),on=shared_cols, left_index=True, right_index=True, suffixes=('_prune', '_rgft'))
		complete_df = complete_df.rename(columns={FEATURES[f]: FEATURES[f] + "_rgft" for f in FEATURES_RGFT_ONLY})
		complete_df[LABEL.format(move_type)] = complete_df[LABEL.format("prune")]
		complete_df.to_csv(df_path)

	df_learning = pd.read_csv(df_path, dtype=types_dict)
	df_learning = fit_transformation(df_learning, move_type, trans=args.transform_target)
	
	features = FEATURES_PRUNE if move_type == "prune" else FEATURES_RGFT if move_type == "rgft" else FEATURES_MERGED
	features.remove(FEATURES[GROUP_ID])
	
	########################
	
	suf = "_{}_validation_set".format(st) if args.validation_set else "_{}".format(st)
	iftrans = "" if not args.transform_target else "_ytransformed"
	suf += iftrans
	csv_with_scores = PROJECT_PATH + SCORES_PER_DS.format(str(len(features))+ suf)
	csv_with_preds = PROJECT_PATH + DATA_WITH_PREDS.format(str(len(features)) + suf)
	if not os.path.exists(csv_with_scores) or args.validation_set:
		print("*@*@*@* scores for step{} with {} features are not available, thus applying learning".format(suf, len(features)))
		res_dict, df_out = cross_validation_RF(df_learning, move_type, features, trans=args.transform_target ,validation_set=args.validation_set)
		df_out.to_csv(csv_with_preds)

		df_datasets =  pd.DataFrame(columns=["init"])
	else:
		df_datasets = pd.read_csv(csv_with_scores)
		res_dict = extract_scores_dict({}, df_datasets)
	df_datasets = print_and_index_results(df_datasets, res_dict, features)
	df_datasets.to_csv(csv_with_scores)