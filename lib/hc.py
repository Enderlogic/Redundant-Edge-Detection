import numpy as np
from scipy.special import gammaln
from copy import deepcopy
import lib.score as score
from time import time


def score_diff(gra1, gra2, data, counts, arities, varnames, score_function):
    if score_function == 'bdeu':
        sco_diff = score_bdeu_diff(gra1, gra2, data, counts, arities, varnames)
    elif score_function == 'bic':
        sco_diff = score_bic_diff(gra1, gra2, data, counts, arities, varnames)
    else:
        raise Exception('Score function is undefined.')
    return sco_diff


# compare the bdeu score between gra1 and gra2, a positive value means bdeu(gra2) > bdeu(gra1)
def score_bdeu_diff(gra1, gra2, data, counts, arities, varnames):
    iss = 1
    bdeu_diff = 0
    for tar in data:
        if set(gra1[tar]) != set(gra2[tar]):
            q1 = 1
            q2 = 1
            r = len(data[tar].unique())

            if gra1[tar]:
                N_jk = {k: v[tar].value_counts().to_dict() for k, v in data.groupby(gra1[tar])}

                for par_var in gra1[tar]:
                    q1 = q1 * len(data[par_var].unique())
                alp1_j = iss / q1
                alp1_jk = iss / q1 / r

                for key_par in N_jk:
                    for key_tar in N_jk[key_par]:
                        bdeu_diff = bdeu_diff - gammaln(alp1_jk + N_jk[key_par][key_tar]) + gammaln(alp1_jk)
                    bdeu_diff = bdeu_diff - gammaln(alp1_j) + gammaln(alp1_j + sum(N_jk[key_par].values()))
            else:
                alp1_j = iss
                alp1_jk = iss / r
                N_k = data[tar].value_counts().to_dict()
                bdeu_diff = bdeu_diff - gammaln(alp1_j) + gammaln(alp1_j + len(data))

                for key_tar in N_k:
                    bdeu_diff = bdeu_diff - gammaln(alp1_jk + N_k[key_tar]) + gammaln(alp1_jk)
            if gra2[tar]:
                N_jk = {k: v[tar].value_counts().to_dict() for k, v in data.groupby(gra2[tar])}

                for par_var in gra2[tar]:
                    q2 = q2 * len(data[par_var].unique())

                alp2_j = iss / q2
                alp2_jk = iss / q2 / r

                for key_par in N_jk:
                    for key_tar in N_jk[key_par]:
                        bdeu_diff = bdeu_diff + gammaln(alp2_jk + N_jk[key_par][key_tar]) - gammaln(alp2_jk)
                    bdeu_diff = bdeu_diff + gammaln(alp2_j) - gammaln(alp2_j + sum(N_jk[key_par].values()))
            else:
                alp2_j = iss
                alp2_jk = iss / r
                N_k = data[tar].value_counts().to_dict()

                bdeu_diff = bdeu_diff + gammaln(alp2_j) - gammaln(alp2_j + len(data))

                for key_tar in N_k:
                    bdeu_diff = bdeu_diff + gammaln(alp2_jk + N_k[key_tar]) - gammaln(alp2_jk)
    return bdeu_diff


# compare the bic score between gra1 and gra2, a positive value means bdeu(gra2) > bdeu(gra1)
def score_bic_diff(gra1, gra2, data, counts, arities, varnames):
    bic_diff = 0
    for tar in gra1:
        if set(gra1[tar]['par']) != set(gra2[tar]['par']):
            cols = [varnames.index(tar)]
            for var in gra1[tar]['par']:
                cols.append(varnames.index(var))
            cols = np.asarray(cols)
            bic_diff -= score.bic_counter(data, counts, arities, cols)

            cols = [varnames.index(tar)]
            for var in gra2[tar]['par']:
                cols.append(varnames.index(var))
            cols = np.asarray(cols)
            bic_diff += score.bic_counter(data, counts, arities, cols)
    return bic_diff


def hc(data, counts, arities, varnames, score_function='bic', pc=None):
    # input:
    # data: training data
    # pc: PC set for variables
    # score_function: score function used in hill climbing algorithm
    # output:
    # gra: a dictionary containing variables with their parents

    ave_time = 0
    test_num = 0

    if pc is None:
        pc = {}
        for var in varnames:
            pc[var] = deepcopy(varnames)
            pc[var].remove(var)

    dag = {}
    dag_temp = {}
    for var in varnames:
        dag[var] = {}
        dag[var]['par'] = []
        dag[var]['nei'] = []
        dag_temp[var] = {}
        dag_temp[var]['par'] = []
        dag_temp[var]['nei'] = []

    diff = 1

    # attempt to find better graph until no difference could make
    while diff > 0:

        diff = 0
        edge_candidate = []
        dag_temp = deepcopy(dag)

        cyc_flag = False

        for tar in varnames:
            # attempt to add edges
            for pc_var in pc[tar]:
                underchecked = [tar]
                checked = []
                while underchecked:
                    if cyc_flag:
                        break
                    underchecked_copy = deepcopy(underchecked)
                    for gra_par in underchecked_copy:
                        if dag[gra_par]['par']:
                            if pc_var in dag[gra_par]['par']:
                                cyc_flag = True
                                break
                            else:
                                for key in dag[gra_par]['par']:
                                    if key not in checked:
                                        underchecked.append(key)
                        underchecked.remove(gra_par)
                        checked.append(gra_par)

                if cyc_flag:
                    cyc_flag = False
                else:
                    dag_temp[pc_var]['par'].append(tar)

                    # start = time()
                    score_diff_temp = score_diff(dag, dag_temp, data, counts, arities, varnames, score_function)
                    # end = time()
                    # test_num += 1
                    # ave_time = (ave_time * (test_num - 1) + end - start) / test_num

                    if score_diff_temp - diff > 1e-10:
                        diff = score_diff_temp
                        edge_candidate = [pc_var, tar, 'a']

                    dag_temp[pc_var]['par'].remove(tar)

            for par_var in dag[tar]['par']:
                # attempt to reverse edges
                dag_temp[par_var]['par'].append(tar)
                dag_temp[tar]['par'].remove(par_var)
                underchecked = [tar]
                checked = []
                while underchecked:
                    if cyc_flag:
                        break
                    underchecked_copy = deepcopy(underchecked)
                    for gra_par in underchecked_copy:
                        if dag_temp[gra_par]['par']:
                            if par_var in dag_temp[gra_par]['par']:
                                cyc_flag = True
                                break
                            else:
                                for key in dag_temp[gra_par]['par']:
                                    if key not in checked:
                                        underchecked.append(key)
                        underchecked.remove(gra_par)
                        checked.append(gra_par)

                if cyc_flag:
                    cyc_flag = False
                else:
                    # start = time()
                    score_diff_temp = score_diff(dag, dag_temp, data, counts, arities, varnames, score_function)
                    # end = time()
                    # test_num += 1
                    # ave_time = (ave_time * (test_num - 1) + end - start) / test_num
                    if score_diff_temp - diff > 1e-10:
                        diff = score_diff_temp
                        edge_candidate = [tar, par_var, 'r']

                dag_temp[par_var]['par'].remove(tar)

                # attempt to delete edges
                # start = time()
                score_diff_temp = score_diff(dag, dag_temp, data, counts, arities, varnames, score_function)
                # end = time()
                # test_num += 1
                # ave_time = (ave_time * (test_num - 1) + end - start) / test_num
                if score_diff_temp - diff > 1e-10:
                    diff = score_diff_temp
                    edge_candidate = [tar, par_var, 'd']

                dag_temp[tar]['par'].append(par_var)

        # print(diff)
        # print(edge_candidate)
        # print(ave_time)
        # print(sum([len(dag_temp[x]['par']) for x in dag_temp]))
        if edge_candidate:
            if edge_candidate[-1] == 'a':
                dag[edge_candidate[0]]['par'].append(edge_candidate[1])
                pc[edge_candidate[0]].remove(edge_candidate[1])
                pc[edge_candidate[1]].remove(edge_candidate[0])
            elif edge_candidate[-1] == 'r':
                dag[edge_candidate[1]]['par'].append(edge_candidate[0])
                dag[edge_candidate[0]]['par'].remove(edge_candidate[1])
            elif edge_candidate[-1] == 'd':
                dag[edge_candidate[0]]['par'].remove(edge_candidate[1])
                pc[edge_candidate[0]].append(edge_candidate[1])
                pc[edge_candidate[1]].append(edge_candidate[0])
    return dag
