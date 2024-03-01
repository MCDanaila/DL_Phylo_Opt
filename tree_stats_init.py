import csv
import os
from Bio.Align import AlignInfo
from Bio import AlignIO
import pandas as pd
import numpy as np
from Utils.RF_learning_algorithm import parse_relevant_summaries_for_learning

from Tools.PhyML import parse_phyml_stats_file, run_phyml
from Tools.RaxML import call_raxml_mem
from Utils import tree_functions, collect_features
from Utils.defs_PhyAI import FEATURES, FEATURES_SHARED, ROOTLIKE_NAME, TEST_DATA_PATH, SEP, DEFAULT_MODEL, SUBTREE1, SUBTREE2, LEARNING_DATA, MSA_PHYLIP_FILENAME, TREES_PER_DS, SUMMARY_PER_DS, GROUP_ID, FEATURES_PRUNE, FEATURES_RGFT, types_dict, LABEL

######################## VALIDATION #######################

def validate_input(msa_file, user_tree_file):
	"""
	:param msa_file: the path to an MSA file, one of biopython's formats
	:param user_tree_file: (optional) the path to a user tree file, if fixed tree was desired
	:return: a biopython object of the msa and an ete3 object of the tree if exists
	"""
	# identify format and retrieve all MSAs
	for aln_format in ["clustal", "emboss", "fasta", "fasta-m10", "ig", "maf", "mauve", "nexus", "phylip-relaxed", "phylip-sequential", "stockholm"]:
		try:
			msa_obj = AlignIO.read(msa_file, format=aln_format)
			print("INFO - The MSA file is format: " + aln_format)
			break
		except Exception:
			msa_obj = None
	if msa_obj is None:
		print("ERROR - Error occured: the input file is not a valid alignmnet in a supported format.\n"
					 "Please verify that all sequences are at the same length and that the input format is correct.")
	# validate MSA characters
	msa_info = AlignInfo.SummaryInfo(msa_obj)
	aln_letters = msa_info._get_all_letters()
	for let in aln_letters:
		if not (let.lower() in "acgt-"):
			print("WARNING - There are characters that are not nucleotides or gaps in your input MSA.")
			break
	# validate tree file in Newick format and suits the msa
	tree_obj = None
	if user_tree_file:
		try:
			with open(user_tree_file) as fpr:
				tree_obj = tree_functions.get_newick_tree(fpr.read().strip())
		except:
			print("ERROR - Tree file is invalid. Please verify that it's in Newick format.")
		# assert that the tree matches the corresponding MSA
		leaves = sorted([node.name for node in tree_obj.get_leaves()])
		seq_names = sorted([rec.id for rec in msa_obj])
		if len(leaves) != len(seq_names) or (not all(x == y for x,y  in zip(seq_names,leaves))):
			print("ERROR - The tips of the tree and the MSA sequences names do not match")

	return msa_obj

def simulate_SPR(current_wd, stats_file, orig_tree_obj, orig_msa_file):
	outpath_trees = TREES_PER_DS.format(current_wd)
	outpath_prune = SUMMARY_PER_DS.format(current_wd, 'prune')
	outpath_rgft = SUMMARY_PER_DS.format(current_wd, 'rgft')
	newick_steps_list = []
	#df_newick_steps = pd.DataFrame(columns=['iteration', 'prune_name', 'rgft_name', 'newick'])
	with open(outpath_trees, "w", newline='') as fpw:
			csvwriter = csv.writer(fpw)
			csvwriter.writerow(['iteration', 'prune_name', 'rgft_name', 'newick'])
	# first, copy msa file to memory and save it:
	msa_rampath = "/Volumes/RAMDisk/tmp" + current_wd.split(SEP)[-2] #  to be on the safe side (even though other processes shouldn't be able to access it)
	with open(orig_msa_file) as fpr:
		msa_str = fpr.read()
	try:  
		with open(msa_rampath, "w") as fpw:
			fpw.write(msa_str)  # don't write the msa string to a variable (or write and release it)
		msa_str = ''
  
		print("INFO - RUN: parse_phyml_stats_output ")
		params_dict = (parse_phyml_stats_file(stats_file))
		#keep pinv and alpha 
		freq, rates, pinv, alpha = [params_dict["fA"], params_dict["fC"], params_dict["fG"], params_dict["fT"]], [params_dict["subAC"], params_dict["subAG"], params_dict["subAT"], params_dict["subCG"],params_dict["subCT"], params_dict["subGT"]], params_dict["pInv"], params_dict["gamma"]
		df = pd.DataFrame(columns=['orig_ds_ll','orig_ds_tbl','prune_name','rgft_name','time','ll'])
		orig_ds_ll = float(params_dict["logL"])
		total_bl = None
		root_children = orig_tree_obj.get_tree_root().get_children()
	
		for i, prune_node in enumerate(orig_tree_obj.iter_descendants("levelorder")):
			if prune_node in root_children:
				#print("SKIP TREE prune_node: ", prune_node.get_ascii(attributes=["name", "dist"]))
				continue
			prune_name = prune_node.name
			#x = prune_node.get_ascii(attributes=["name", "dist"])
			nname, subtree1, subtree2 = tree_functions.prune_branch(orig_tree_obj, prune_name) # subtree1 is the pruned
			#newick_steps_list.append((str(i)+",0", prune_name, SUBTREE1, subtree1.write(format=1)))
			#newick_steps_list.append((str(i)+",1", prune_name, SUBTREE2, subtree2.write(format=1)))
			with open(outpath_trees, "a", newline='') as fpa:
				csvwriter = csv.writer(fpa)
				csvwriter.writerow([str(i)+",0", prune_name, SUBTREE1, subtree1.write(format=1)])
				csvwriter.writerow([str(i)+",1", prune_name, SUBTREE2, subtree2.write(format=1)])
			for j, rgft_node in enumerate(subtree2.iter_descendants("levelorder")): # traversing over subtree2 capture cases (1) and (3)
				# skip the ROOT node when regraft 
				ind = str(i) + "," + str(j)
				rgft_name = rgft_node.name
				y = rgft_node.get_ascii(attributes=["name", "dist"])
				if nname == rgft_name: # captures case (2)
					continue
				rearr_tree, preserve = tree_functions.regraft_branch(subtree2, rgft_node, subtree1, rgft_name, nname)
				neighbor_tree_str = rearr_tree.write(format=1, format_root_node=True)

				### save tree to file by using "append"
				#newick_steps_list.append((ind, prune_name, rgft_name, neighbor_tree_str))
				with open(outpath_trees, "a", newline='') as fpa:
					csvwriter = csv.writer(fpa)
					csvwriter.writerow([ind, prune_name, rgft_name, neighbor_tree_str])
				if total_bl == None :
					total_bl = tree_functions.get_total_branch_lengths(neighbor_tree_str)
				
				ll_rearr, rtime = call_raxml_mem(neighbor_tree_str, msa_rampath, rates, pinv, alpha, freq)
				#print(f">> Total branch lenght: {total_bl}")
				#print(f">> PIP Likelihood: {ll_rearr}" )
				df.loc[ind, "prune_name"], df.loc[ind, "rgft_name"] = prune_name, rgft_name
				df.loc[ind, "prune_name"], df.loc[ind, "rgft_name"] = prune_name, rgft_name
				df.loc[ind, "time"] = rtime
				df.loc[ind, "ll"] = ll_rearr
		df["orig_ds_ll"] = orig_ds_ll
		df[FEATURES["group_tbl"]] = total_bl
		#pd.DataFrame(data=newick_steps_list, columns=['iteration', 'prune_name', 'rgft_name', 'newick']).to_csv(outpath_trees, index=False)
		df.to_csv(outpath_prune.format("prune"), index_label='iteration')
		df.to_csv(outpath_rgft.format("rgft"), index_label='iteration')
		#df.to_csv(outpath_prune.format("prune"), index=False)
		#df.to_csv(outpath_rgft.format("rgft"), index=False)
	except Exception as e:
		print('ERROR - Could not complete the all_SPR function on dataset:', current_wd, '\nError message:')
		print(e)
		exit()
	finally:
		os.remove(msa_rampath)
	return outpath_trees, outpath_prune, outpath_rgft

def start(ds_path):
	"""
	:param ds_path: the path to the folder where the dasets are stored e.g. path/to/test_data
	"""
	dirList = [ name for name in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, name)) ]
	print(f'--------- TOTAL DIR: {len(dirList)}')
	counter = 0
	for folder_name in dirList:
		CWD = ds_path + folder_name + SEP
		if os.path.exists(TREES_PER_DS.format(CWD)) and os.path.exists(SUMMARY_PER_DS.format(CWD, 'prune')) and os.path.exists(SUMMARY_PER_DS.format(CWD, 'rgft')): 
			counter += 1
			print(f'--------- SKIPPED {folder_name} -> {counter}/{len(dirList)}')
			continue
		orig_msa_file = os.path.join(CWD, MSA_PHYLIP_FILENAME)
		run_id = DEFAULT_MODEL
		print(f"INFO - RUN: PhyML in {CWD=} ")
		stats_file, tree_file = run_phyml(orig_msa_file, DEFAULT_MODEL, run_id=run_id)
		print(f"INFO - Files created: \n {stats_file=} \n {tree_file=} ")
		validate_input(orig_msa_file, tree_file)

		orig_tree_obj = tree_functions.get_phylo_tree(tree_file, orig_msa_file)
		orig_tree_obj.get_tree_root().name = ROOTLIKE_NAME
		#print(orig_tree_obj.get_ascii(attributes=["name", "dist"])) 

		outpath_trees, outpath_prune, outpath_rgft = simulate_SPR(CWD, stats_file, orig_tree_obj, orig_msa_file)
		print(f"INFO - SPR moves simulated on dataset {CWD=}")
		print(f"INFO - Files created: \n {outpath_trees=} \n {outpath_prune=} \n {outpath_rgft=}")
	
		collect_features.collect_features(CWD, tree_file, outpath_prune, outpath_rgft)
		counter += 1
		print(f'--------- DONE {counter}/{len(dirList)}')


def start_parallel(ds_path):
	print(f'--------- RUN: {ds_path}') 
	CWD = ds_path + SEP
	#if os.path.exists(TREES_PER_DS.format(CWD)) and os.path.exists(SUMMARY_PER_DS.format(CWD, 'prune')) and os.path.exists(SUMMARY_PER_DS.format(CWD, 'rgft')): return
	orig_msa_file = os.path.join(CWD, MSA_PHYLIP_FILENAME)
	run_id = DEFAULT_MODEL
	print(f"INFO - RUN: PhyML in {CWD=} ")
	stats_file, tree_file = PhyML.run_phyml(orig_msa_file, DEFAULT_MODEL, run_id=run_id)
	print(f"INFO - Files created: \n {stats_file=} \n {tree_file=} ")
	validate_input(orig_msa_file, tree_file)

	orig_tree_obj = tree_functions.get_phylo_tree(tree_file, orig_msa_file)
	orig_tree_obj.get_tree_root().name = ROOTLIKE_NAME
	#print(orig_tree_obj.get_ascii(attributes=["name", "dist"])) 

	outpath_trees, outpath_prune, outpath_rgft = simulate_SPR(CWD, stats_file, orig_tree_obj, orig_msa_file)
	print(f"INFO - SPR moves simulated on dataset {CWD=}")
	print(f"INFO - Files created: \n {outpath_trees=} \n {outpath_prune=} \n {outpath_rgft=}")

	collect_features.collect_features(CWD, tree_file, outpath_prune, outpath_rgft)


def start_single(cwd_path):
	"""
	:param cwd_path: the path to the folder where the msa is stored e.g. path/to/test_data/5
	"""
	orig_msa_file = os.path.join(cwd_path, MSA_PHYLIP_FILENAME)
	run_id = DEFAULT_MODEL	
	print(f"INFO - RUN: PhyML in {cwd_path=} ")
	stats_file, tree_file = PhyML.run_phyml(orig_msa_file, DEFAULT_MODEL, run_id=run_id)
	print(f"INFO - Files created: \n {stats_file=} \n {tree_file=} ")
	validate_input(orig_msa_file, tree_file)

	orig_tree_obj = tree_functions.get_phylo_tree(tree_file, orig_msa_file)
	orig_tree_obj.get_tree_root().name = ROOTLIKE_NAME
	print(orig_tree_obj.get_ascii(attributes=["name", "dist"])) 

	outpath_trees, outpath_prune, outpath_rgft = simulate_SPR(cwd_path, stats_file, orig_tree_obj, orig_msa_file)
	print(f"INFO - SPR moves simulated on dataset {cwd_path=}")
	print(f"INFO - Files created: \n {outpath_trees=} \n {outpath_prune=} \n {outpath_rgft=}")
	
	collect_features.collect_features(cwd_path, tree_file, outpath_prune, outpath_rgft)


def parse_and_group_features(ds_path):
	"""
		:param ds_path: the path to the folder where the dasets are stored e.g. path/to/test_data
	"""
	df_path = ds_path + LEARNING_DATA.format("all_moves")
	#print(df_path)
	df_prune_features = ds_path + LEARNING_DATA.format("all_moves_prune")
	#print(df_prune_features)
	df_rgft_features = ds_path + LEARNING_DATA.format("all_moves_rgft")

	parse_relevant_summaries_for_learning(ds_path, df_prune_features, "prune")
	parse_relevant_summaries_for_learning(ds_path, df_rgft_features, "rgft")

	shared_cols = FEATURES_SHARED + ["iteration", "prune_name","rgft_name","orig_ds_ll"]
	
	df_prune_moves = pd.read_csv(df_prune_features, dtype=types_dict, index_col='iteration')
	print(df_prune_moves.columns)

	df_rgft_moves = pd.read_csv(df_rgft_features, dtype=types_dict, index_col='iteration')
	print(df_rgft_moves.columns)

	complete_df = pd.merge(df_prune_moves, df_rgft_moves, how='outer', on=shared_cols, suffixes=('_prune', '_rgft'))

	complete_df[LABEL.format("merged")] = complete_df[LABEL.format("prune")]
	complete_df.to_csv(df_path, index_label='iteration')

	return df_path