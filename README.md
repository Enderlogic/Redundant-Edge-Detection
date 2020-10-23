# Introduction
This is an implementation of the Spurious Edge Detection (SED) algorithm which aims to eliminate the spurious edges learned by structure learning algorithms in the presence of measurement error.

There are two embedded structure learning algorithms (HC, PC-Stable) in this code. If you want to test other algorithms, please save the learned graph in the folder "learned graph" with format as the files in it.

The sturcture of a BN is stored in an object of dictionary. The key of the dictionary is the name of variable in BN and the corresponding value is a dictionary with two lists ("par" and "nei") consisted by the key's parents and non-parent-child neighbors respectively. In other words, the "par" list contains all the parents of the key node and the "nei" list contains all the nodes that connect to the key node via an undirected edge. If a BN is a DAG, there should be no nodes in any "nei" list.
  
# Requirements
- Python 3.5+
- pandas
- scipy
- numba

# Contents
- folders
    - true graph: the ground truth graph
    - learned graph: the graphs learned by structure learning algorithms
    - modified graph: the graphs modified by RED
    - result: the evaluation results of learned graph and corrected graph
    - lib: a folder with necessary scripts including:
        - accessory.py: some auxiliary functions used in other scripts
        - evaluation.py: script for evaluation scores (the F1 and SHD score)
        - score.py: some score computing functions used in other scripts
        - hc.py: implementation of the hill-climbing algorithm
        - pc_stable.py: implementation of the PC-Stable algorithm
        - sed.py: implementation of the SED algorithm
- SED_demo.py: a demo sript
