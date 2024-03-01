# DL_Phylo_Opt
Master thesis: "Phylogenetic tree search heuristic optimisation using deep learning methods"



1. select dataset ad aligment
2. 	validate input
3. 	call PhyML to generate starting tree and initial stats
	
	3.1 calculate all SPR moves from the starting tree
	
	3.2 for each new tree from the SPR move, calculate the likelihood RaxML or PIP parse and store the stats
	
4. collect features
