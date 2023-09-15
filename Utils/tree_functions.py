import os, itertools, shutil
import numpy as np
from ete3 import Tree, PhyloTree
from .utils import compute_entropy, lists_diff
from .defs_PhyAI import PHYLIP_FORMAT

################### TREE OPERATIONS ######################

def get_phylo_tree(tree_file, msa_file):
	try:
		orig_tree_obj = PhyloTree(newick=tree_file, alignment=msa_file, alg_format=PHYLIP_FORMAT, format=1)
		# resolve triplets  of nodes		
		orig_tree_obj.resolve_polytomy(recursive=True)
		add_internal_names(tree_file, orig_tree_obj)
	except:
		orig_tree_obj = PhyloTree(newick=tree_file, alignment=msa_file, alg_format=PHYLIP_FORMAT, format=3)
	return orig_tree_obj

def add_internal_names(tree_file, orig_tree_obj):
	#shutil.copy(tree_file, "tree_copy.txt")
	for i, node in enumerate(orig_tree_obj.traverse()):
		if not node.is_leaf():
			node.name = "N{}".format(i)
	orig_tree_obj.write(format=1, outfile=tree_file)   # runover the orig file with no internal nodes names

def get_newick_tree(tree):
	"""
	:param tree: newick tree string or txt file containing one tree
	:return:	tree: a string of the tree in ete3.Tree format
	"""
	if type(tree) == str:
		if os.path.exists(tree):
			with open(tree, 'r') as tree_fpr:
				tree = tree_fpr.read().strip()
		tree = Tree(tree, format=1)
	return tree

def prune_branch(orig_tree_obj, prune_name):
	t_cp_p = orig_tree_obj.copy()  				# the original tree is needed for each iteration
	prune_node_cp = t_cp_p & prune_name     # locate the node in the copied subtree
	assert prune_node_cp.up					# pointer to parent’s node

	nname = prune_node_cp.up.name
	prune_loc = prune_node_cp
	prune_loc.detach()  # pruning: prune_node_cp is now the subtree we detached. t_cp_p is the one that was left behind
	t_cp_p.search_nodes(name=nname)[0].delete(preserve_branch_length=True)  # delete the specific node (without its childs) since after pruning this branch should not be divided

	return nname, prune_node_cp, t_cp_p

def regraft_branch(t_cp_p, rgft_node, prune_node_cp, rgft_name, nname, preserve=False):
	new_branch_length = rgft_node.dist /2
	t_temp = PhyloTree()  			   # for concatenation of both subtrees ahead, to avoid polytomy
	t_temp.add_child(prune_node_cp)
	t_curr = t_cp_p.copy()
	rgft_node_cp = t_curr & rgft_name  # locate the node in the copied subtree

	rgft_loc = rgft_node_cp.up
	rgft_node_cp.detach()
	t_temp.add_child(rgft_node_cp, dist=new_branch_length)
	t_temp.name = nname
	rgft_loc.add_child(t_temp, dist=new_branch_length)  # regrafting
	if nname == "ROOT_LIKE":  # (4)
		t_temp.delete()
		preserve = True  # preserve the name of the root node, as this is a REAL node in this case

	return t_curr, preserve

def reroot_tree(tree, outgroup_name):
	tree = get_newick_tree(tree)
	tree.set_outgroup(tree & outgroup_name)
	return tree

################### FEATURE EXTRACTION ###################

def get_frac_of_cherries(tree):
	"""
	McKenzie, Andy, and Mike Steel. "Distributions of cherries for two models of trees."
	 Mathematical biosciences 164.1 (2000): 81-92.
	:param tree:
	:return:
	"""
	tree = get_newick_tree(tree)
	tree_root = tree.get_tree_root()
	leaves = list(tree_root.iter_leaves())
	cherries_cnt = 0
	for leaf1, leaf2 in itertools.combinations(leaves, 2):
		if leaf1.up is leaf2.up:
			cherries_cnt += 1
	return 2*cherries_cnt/len(leaves)

def get_leaves_branches(tree):
	"""
	:param tree:
	:return: a list of pendant edges lengths, i.e., the brnches that lead to leaves
	"""
	tree = get_newick_tree(tree)
	tree_root = tree.get_tree_root()
	leaves_bl = []
	for node in tree_root.iter_leaves():
		leaves_bl.append(node.dist)
	return leaves_bl

def get_stemminess_indexes(tree):
	"""
		:param tree: ete3 tree; not rerooting!
		:return: cumulative stemminess index Fiala and Sokal 1985
				 noncumulative stemminess index Rohlf 1990
		formula cumulative stemminess: https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1558-5646.1985.tb00398.x
		formula noncumulative stemminess: https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1558-5646.1990.tb03855.x
		"""
	subtree_blsum_dict = {}
	nodes_height_dict = {}
	stem85_index_lst = []
	stem90_index_lst = []
	for node in tree.traverse(strategy="postorder"):
		if node.is_leaf():
			subtree_blsum_dict[node] = 0
			nodes_height_dict[node] = 0
		elif node.is_root():
			continue
		else:
			subtree_blsum_dict[node] = subtree_blsum_dict[node.children[0]] + subtree_blsum_dict[node.children[1]] + \
									   node.children[0].dist + node.children[1].dist
			nodes_height_dict[node] = max(nodes_height_dict[node.children[0]] + node.children[0].dist,
										  nodes_height_dict[node.children[1]] + node.children[1].dist)
			stem85_index_lst.append(node.dist/(subtree_blsum_dict[node] + node.dist))
			stem90_index_lst.append(node.dist/(nodes_height_dict[node]) + node.dist)

	return np.mean(stem85_index_lst), np.mean(stem90_index_lst)

def get_branch_lengths(tree):
	"""
	:param tree: Tree node or tree file or newick tree string;
	:return: total branch lengths
	"""
	# TBL
	tree = get_newick_tree(tree)
	tree_root = tree.get_tree_root()
	branches = []
	for node in tree_root.iter_descendants(): # the root dist is 1.0, we don't want it
		branches.append(node.dist)
	return branches

def get_total_branch_lengths(tree):
	"""
	:param tree: Tree node or tree file or newick tree string;
	:return: total branch lengths
	"""
	branches = get_branch_lengths(tree)
	return sum(branches)

def get_branch_lengths_estimates(tree):
	"""
	:param tree: Tree node or tree file or newick tree string;
	:return:
	"""
	# TBL
	branches = get_branch_lengths(tree)
	entropy = compute_entropy(branches)

	return max(branches), min(branches), np.mean(branches), np.std(branches), entropy

def get_diameters_estimates(tree_filepath, actual_bl=True):
	"""
	if not actual_bl - function changes the tree! send only filepath
	:param tree_filepath: tree file or newick tree string;
	:param actual_bl: True to sum actual dists, False for num of branches
	:return: min, max, mean, and std of tree diameters
	"""
	# tree = copy.deepcopy(get_newick_tree(tree)) # do not deepcopy! when trees are large it exceeds recursion depth
	if not actual_bl:
		assert isinstance(tree_filepath, str)
	tree = get_newick_tree(tree_filepath)
	tree_root = tree.get_tree_root()
	if not actual_bl:
		for node in tree_root.iter_descendants():
			node.dist = 1.0
	tree_diams = []
	leaves = list(tree_root.iter_leaves())
	for leaf1, leaf2 in itertools.combinations(leaves, 2):
		tree_diams.append(leaf1.get_distance(leaf2))
	entropy = compute_entropy(tree_diams)

	return max(tree_diams), min(tree_diams), np.mean(tree_diams), np.std(tree_diams), entropy

def get_internal_and_external_leaves_relative_to_subroot(tree_root, subroot):
	all_leaves = tree_root.get_leaves()
	subtree_leaves = subroot.get_leaves()
	other_leaves = lists_diff(all_leaves, subtree_leaves)
	return subtree_leaves, other_leaves

def get_largest_branch(tree):
	tree = get_newick_tree(tree)
	tree_nodes = list(tree.traverse("levelorder"))
	max_bl_node = max(tree_nodes, key=lambda node: node.dist)
	return max_bl_node

def dist_between_nodes(t, node1):
	nleaves_between, tbl_between = {},{}
	for node2 in t.get_descendants("levelorder")[::-1]:
		if not node2.name:
			node2.name = "Nnew"
		nname2 = node2.name
		if node1.name == nname2:
			continue

		nleaves_between[nname2] = node1.get_distance(node2, topology_only=True)+1   # +1 to convert between nodes count to edges
		tbl_between[nname2] = node1.get_distance(node2, topology_only=False)

	return nleaves_between, tbl_between