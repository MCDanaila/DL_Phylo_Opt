# DL_Phylo_Opt
Master thesis: "Phylogenetic tree search heuristic optimisation using deep learning methods"

## Data availability ([from original paper](https://www.nature.com/articles/s41467-021-22073-8#data-availability))
The datasets contained within the empirical set have been deposited in Open Source Framework (OSF) with the identifier [DOI 10.17605/OSF.IO/B8AQJ51](https://osf.io/b8aqj/)
<br>
These datasets were assembled from the following databases: TreeBase (https://treebase.org/treebase-web/urlAPI.html); Selectome (https://selectome.org/); protDB (https://protdb.org/); PloiDB (https://doi.org/10.3732/ajb.1500424); PANDIT (https://www.ebi.ac.uk/research/goldman/software/pandit); OrthoMaM (https://orthomam.mbb.cnrs.fr/).

## Code availability ([from original paper](https://www.nature.com/articles/s41467-021-22073-8#code-availability))
The code that supports the findings of this study was written in Python version 3.6 and has been deposited in Open Source Framework (OSF) with the identifier [DOI 10.17605/OSF.IO/B8AQJ51](https://osf.io/b8aqj/). Computation of likelihoods and parameter estimates were executed using the following application versions: PhyML 3.031, RAxML-NG 0.9.048.

## Project Structure

<b>tools:</b> scripts to handle external tools PhyML and RaxML

<b>utils:</b> utility scripts adapted from original project
- <b>collect_features.py:</b> features extraction and organizazion in csv
- <b>defs_PhyAI.py:</b> definition of phyton shared constant variables
- <b>RF_learning_algorithm.py:</b> original RandomForestRegressor algorith and related methods to score the predictions
- <b>tree_functions.py:</b> methods for tree operation and tree based feature extraction
- <b>utils.py:</b> math auxiliary functions


<b>runanalysis.py:</b> script to start the pipeline from command line. Uncomment the path to the folder you want to run. 
```bash
python3 runanalysis.py
```
<b>tree_stats_init.py:</b> methods to start first step of the pipeline

<b>DL_DynamicRegressionModel.ipynb:</b> notebook to train the DynamicRegressionModel

<b>DL_Optimization_DynamicRegressionModel.ipynb:</b> notebook for the hyperparametrization of the DynamicRegressionModel

<b>DL_RandomForestRegressor.ipynb:</b> notebook to train a general RandomForestRegressor

<b>DL_RNN_Model.ipynb:</b> notebook to train a RNN Model

<b>DL_stats_utility.ipynb:</b> notebook with different auxiliary methods to manage folders and file in the data folders needed to avoid and handle specific errors after analysis

<b>original_run.ipynb:</b> notebook to run original pipeline

<b>original_test.ipynb:</b> notebook to run original pipeline splitted in different blocks for test


## Data collection 

1. select dataset ad aligment
2. validate input
3. call PhyML to generate starting tree and initial stats
	
	- 3.1 calculate all SPR moves from the starting tree
	- 3.2 for each new tree from the SPR move, calculate the likelihood RaxML or PIP parse and store the stats
	
4. collect features in a big CSV file from all the data


## Models training

Models are trained using the original folder division, but it is better to train models by grouping all data together and let the `sklearn` library methods handle the split into training, testing and validation. This, as discussed in the thesis, will ensure greater homogeneity of the data in the different folds of the training process.
