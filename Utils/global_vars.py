SEP = "/"

######## CHANGE ME ########
PROJECT_PATH = "/Users/mihaid/Coding-Projects/thesis/DL_Phylo_Opt"
##########################


#/Users/mihaid/Coding-Projects/thesis/DL_Phylo_Opt/data
DATA_PATH = SEP.join([PROJECT_PATH, "data", ""])
#/Users/mihaid/Coding-Projects/thesis/DL_Phylo_Opt/data/test_data
TEST_DATA_PATH = DATA_PATH + "test_data" + SEP
#/Users/mihaid/Coding-Projects/thesis/DL_Phylo_Opt/data/training_data
TEST_DATA_PATH = DATA_PATH + "test_data" + SEP


###############################################################################
######################## global vars for common strings #######################
###############################################################################
DEFAULT_MODEL = "GTR+I+G"
#DEFAULT_MODEL = "GTR"
OPT_TYPE = "br"
PHYLIP_FORMAT = "iphylip"
REARRANGEMENTS_NAME = "rearrangement"
SUBTREE1 = "subtree1"
SUBTREE2 = "subtree2"
ROOTLIKE_NAME = "ROOT_LIKE"
GROUP_ID = 'group_id'
KFOLD = 10
N_ESTIMATORS = 70


MSA_PHYLIP_FILENAME = "real_msa.phy"
PHYML_STATS_FILENAME = MSA_PHYLIP_FILENAME + "_phyml_stats_{0}.txt"
PHYML_TREE_FILENAME = MSA_PHYLIP_FILENAME + "_phyml_tree_{0}.txt"
SUMMARY_PER_DS = "{}ds_summary_SPR_{}.csv"
TREES_PER_DS = "{}newicks_step.csv"
LEARNING_DATA = "learning_{}.csv"
DATA_WITH_PREDS = "with_preds_merged_{}.csv"
SCORES_PER_DS = "scores_per_ds_{}.csv"
###############################################################################
