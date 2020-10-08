import numpy as np
import itertools
from lib.accessory import orient_propagation, orient_v_structure, independence_test
from copy import deepcopy


def pc_stable(data, counts, arities, arities_wm, varnames, mod='normal', test_function='g-square', dof_type='ori', threshold=0.05):
    '''
    # input:
        :param data: input training data (unique version)
        :param counts: the counts of unique observation
        :param arities: the number of states for each variable
        :param arities_wm: the number of states without missing for each variable
        :param varnames: the name of variables
        :param mod: the method to deal with missing value (options: normal and test-delete)
        :param test_function: the type of test function, currently support 'g-square'
        :param dof: the manner to compute degree of freedom of G-test (options: ori, enp)
        :param threshold: threshold for CI test
    # output:
        :return dag: a dictionary containing variables with their parents
    '''

    # initialise pc set as full for all variables
    pc = {}
    sepset = {}

    for tar in varnames:
        pc[tar] = deepcopy(varnames)
        pc[tar].remove(tar)
        sepset[tar] = {}
    l = 0

    # find PC set for each variable
    while max({key: len(value) for key, value in pc.items()}.values()) > l:
        pc_copy = deepcopy(pc)
        for tar in varnames:
            for adj in pc_copy[tar]:
                if adj not in sepset[tar]:
                    pc_temp = deepcopy(pc_copy[tar])
                    pc_temp.remove(adj)
                    for con in itertools.combinations(pc_temp, l):
                        cols = np.asarray([varnames.index(tar), varnames.index(adj)] + [varnames.index(var) for var in con], dtype = 'uint32')
                        test_result = independence_test(data, counts, arities, cols, arities_wm, dof_type,
                                                        mod, threshold)[0]
                        if test_result:
                            pc[tar].remove(adj)
                            pc[adj].remove(tar)
                            sepset[tar][adj] = con
                            sepset[adj][tar] = con
                            break
            # print('remaining edges:', sum(len(lst) for lst in pc.values()) / 2)
        l += 1

    # orient v-structrues
    dag = orient_v_structure(data, counts, arities, varnames, pc, sepset, arities_wm, dof_type, mod, threshold)

    # orientation propagation
    dag = orient_propagation(dag)
    return dag