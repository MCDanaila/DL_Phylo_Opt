from tree_stats_init import start, parse_and_group_features
# ds_path = './data/validation_data/'
# ds_path = './data/additional_training_data/'
# ds_path = './data/training_data/'
# ds_path = './data/test_data/'
ds_path = '/Users/mihaid/Coding-Projects/thesis/others/harness_ML_phy-tree-search/validation_data/'
# ds_path = '/Users/mihaid/Coding-Projects/thesis/others/harness_ML_phy-tree-search/training_data/'

start(ds_path)
ds_path = parse_and_group_features(ds_path)
print(ds_path)
