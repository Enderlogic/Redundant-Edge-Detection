# Introduction
This is an implementation of the Redundant Edge Detection (RED) algorithm which aims to eliminate the redundant edges learned by structure learning algorithms in the presence of measurement error.

There are two embedded structure learning algorithms (HC, PC-Stable) in this code. If you want to test other algorithms, please save the learned graph in the folder "learned graph" with format as the files in it.

The sturcture of a BN is stored in an object of dictionary. The key of the dictionary is the name of variable in BN and the corresponding value is a dictionary with two lists ("par" and "nei") consisted by the key's parents and neighbors respectively. The parent of a variable will only appear in the "par" list but not in the "nei" list. 
# Requirements
- Python 3.5+
- pandas
- scipy
- numba

# Contents
- folders
    - true graph: the ground truth graph
    - learned graph: the graphs learned by structure learning algorithms
    - corrected graph: the graphs corrected by RED
    - result: the evaluation results of learned graph and corrected graph
    - lib: a folder with necessary scripts including:
        - accessory.py: some auxiliary functions used in other scripts
        - evaluation.py: the F1 evaluation
        - score.py: some score computing functions used in other scripts
        - hc.py: implementation of the hill-climbing algorithm
        - pc_stable.py: implementation of the PC-Stable algorithm
        - red.py: implementation of the RED algorithm
- RED_demo.py: a demo sript
