import pandas as pd
from lib.evaluation import compare, shd
from lib.accessory import data_preprocessing, cpdag
import json
from lib.pc_stable import pc_stable
from lib.hc import hc
from lib.sed import sed
from pathlib import Path
import os
import csv

dataset_repository = ['asia', 'alarm', 'child', 'insurance', 'mildew', 'water', 'hailfinder']
datasize_repository = [100, 500, 1000, 5000, 10000, 50000, 100000]
noise_repository = ['E0', 'E5', 'E10']
algorithm_repository = ['hc', 'pc.stable', 'h2pc', 'mmhc', 'gobnilp']

dataset_list = ['alarm']
datasize_list = [100, 500, 1000, 5000, 10000, 50000, 100000]
noise_list = ['E0', 'E5', 'E10']
algorithm_list = ['hc', 'h2pc', 'pc.stable', 'gobnilp']

result_columns = ['dataset', 'datasize', 'noise', 'algorithm', 'f1', 'shd']
result = []
for dataset in dataset_list:
    # load true graph
    with open('true graph/' + dataset + '.json') as json_file:
        true_dag = json.load(json_file)
    true_cpdag = cpdag(true_dag)
    for datasize in datasize_list:
        for noise in noise_list:
            if noise == 'N':
                noise_in_file = 'Error-free'
            else:
                noise_in_file = 'Noisy'
            # load data
            data_file = 'data/' + dataset + '/' + noise + '/' + dataset + str(datasize) + noise + '.csv'
            if os.path.exists(data_file):
                data_training = pd.read_csv(data_file)
                data, counts, arities, arities_wm, varnames = data_preprocessing(data_training)
                for alg in algorithm_list:
                    learned_dag_file = 'learned graph/' + alg + '/' + dataset + '/' + noise + '/' + dataset + str(datasize) + noise + '.json'
                    if os.path.exists(learned_dag_file):
                        with open(learned_dag_file) as json_file:
                            learned_dag = json.load(json_file)
                        # evaluate learned graph
                        result.append({'dataset': dataset, 'datasize': datasize, 'noise': noise_in_file, 'algorithm': alg, 'f1': compare(true_cpdag, cpdag(learned_dag))['f1'], 'shd': shd(true_cpdag, cpdag(learned_dag))})
                        print(result[-1])
                        # run RED algorithm
                        modified_cpdag = sed(learned_dag, data, counts, arities, varnames, score_function = 'bic')
                        # generate and save modified graph
                        modified_folder = 'modified graph/' + alg + '/' + dataset + '/' + noise
                        Path(modified_folder).mkdir(parents=True, exist_ok=True)
                        with open(modified_folder + '/' + dataset + str(datasize) + noise + '_modified.json', 'w') as outfile:
                            json.dump(modified_cpdag, outfile)
                        # or load modified graph from existing file
                        # with open('modified graph/' + alg + '/' + dataset + '/' + noise + '/' + dataset + str(datasize) + noise + '_corrected.json') as json_file:
                        #     corrected_cpdag = json.load(json_file)
                        # evaluate modified graph
                        result.append({'dataset': dataset, 'datasize': datasize, 'noise': noise_in_file, 'algorithm': alg + '+red', 'f1': compare(true_cpdag, cpdag(modified_cpdag))['f1'], 'shd': shd(true_cpdag, cpdag(modified_cpdag))})
                        print(result[-1])
with open('result/results.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=result_columns)
    writer.writeheader()
    for row in result:
        writer.writerow(row)