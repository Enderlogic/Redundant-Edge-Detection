from copy import deepcopy
import numpy as np
from scipy.stats.distributions import chi2
from numba import njit
import random


def skeleton(dag):
    ske = {}
    for var in dag:
        for nei in dag[var]['par'] + dag[var]['nei']:
            if var not in ske:
                ske[var] = [nei]
            elif nei not in ske[var]:
                ske[var].append(nei)
            if nei not in ske:
                ske[nei] = [var]
            elif var not in ske[nei]:
                ske[nei].append(var)
    return ske


# preprocess the input data to accelerate the speed of statistical tests
def data_preprocessing(data):
    data_number = np.zeros(data.shape)
    num = 0
    for _, row in data.iteritems():
        state = sorted(row.unique())
        if 'missing' in state:
            state.remove('missing')
            state.append('missing')
        for sta in state:
            data_number[row == sta, num] = state.index(sta)
        num = num + 1
    data_unique, counts = np.unique(np.array(data_number, dtype=np.uint32), axis=0, return_counts=True)
    arities = np.array([data[col].unique().shape[0] for col in data.columns], dtype=np.uint32)
    arities_wm = np.array([data[col].unique()[data[col].unique() != 'missing'].size for col in data.columns],
                          dtype=np.uint32)
    varnames = list(data.columns)

    return data_unique, counts, arities, arities_wm, varnames


# convert dag to cpdag
def cpdag(dag):
    # convert a DAG to a CPDAG
    vstruct = {}
    for var in dag:
        if len(dag[var]['par']) > 1:
            for par in dag[var]['par']:
                par_set = deepcopy(dag[var]['par'])
                par_set.remove(par)
                for par_oth in par_set:
                    if (par not in dag[par_oth]['par']) & (par not in dag[par_oth]['nei']) & (
                            par_oth not in dag[par]['par']):
                        vstruct_single = sorted([par, par_oth])
                        if var in vstruct:
                            if vstruct_single not in vstruct[var]:
                                vstruct[var].append(vstruct_single)
                        else:
                            vstruct[var] = [vstruct_single]
                for nei in dag[var]['nei']:
                    if v_check(dag, var, nei)[1] == 'both':
                        if (par not in dag[nei]['par']) & (par not in dag[nei]['nei']) & (nei not in dag[par]['par']):
                            vstruct_single = [par]
                            if var in vstruct:
                                if vstruct_single not in vstruct[var]:
                                    vstruct[var].append(vstruct_single)
                            else:
                                vstruct[var] = [vstruct_single]
    cpdag = {}
    for var in dag:
        cpdag[var] = {}
        cpdag[var]['par'] = []
        cpdag[var]['nei'] = deepcopy(dag[var]['nei'])
        if var in vstruct:
            for vstruct_single in vstruct[var]:
                cpdag[var]['par'] += vstruct_single
            cpdag[var]['par'] = list(dict.fromkeys(cpdag[var]['par']))

    for var in dag:
        nei_set = list(set(dag[var]['par'] + dag[var]['nei']) - set(cpdag[var]['par']))
        for nei in nei_set:
            if nei not in cpdag[var]['nei']:
                cpdag[var]['nei'].append(nei)
            if var not in cpdag[nei]['nei']:
                cpdag[nei]['nei'].append(var)
    cpdag = orient_propagation(cpdag)
    return cpdag


# orientation propagation
def orient_propagation(dag):
    while True:
        dag_check = deepcopy(dag)
        # check cycles
        for tar in dag:
            if len(dag_check[tar]['nei']):
                nei_set = deepcopy(dag_check[tar]['nei'])
                for nei in nei_set:
                    sin_flag, sin_direction = sin_path_check(dag, tar, nei)
                    v_flag, v_direction = v_check(dag_check, tar, nei)
                    if sin_flag:
                        if (not v_flag) | (sin_direction == v_direction):
                            if sin_direction[0] not in dag[sin_direction[1]]['par']:
                                dag[sin_direction[1]]['par'].append(sin_direction[0])
                            if sin_direction[0] in dag[sin_direction[1]]['nei']:
                                dag[sin_direction[1]]['nei'].remove(sin_direction[0])
                                dag[sin_direction[0]]['nei'].remove(sin_direction[1])
        # check v-structures
        for tar in dag:
            if len(dag_check[tar]['nei']):
                nei_set = deepcopy(dag_check[tar]['nei'])
                for nei in nei_set:
                    sin_flag, sin_direction = sin_path_check(dag, tar, nei)
                    v_flag, v_direction = v_check(dag_check, tar, nei)
                    if v_flag & (v_direction != 'both'):
                        if (not sin_flag) | (sin_direction == v_direction):
                            if v_direction[0] not in dag[v_direction[1]]['par']:
                                dag[v_direction[1]]['par'].append(v_direction[0])
                            if v_direction[0] in dag[v_direction[1]]['nei']:
                                dag[v_direction[1]]['nei'].remove(v_direction[0])
                                dag[v_direction[0]]['nei'].remove(v_direction[1])

        # check Meek's rule 3
        for tar in dag:
            if len(dag_check[tar]['nei']):
                nei_set = deepcopy(dag_check[tar]['nei'])
                for nei in nei_set:
                    if (not sin_path_check(dag_check, tar, nei)[0]) & (not v_check(dag_check, tar, nei)[0]):
                        common_var = list(set(dag[nei]['par']).intersection(dag[tar]['nei']))
                        if len(common_var) > 1:
                            flag = False
                            for var1 in common_var:
                                common_var_remain = deepcopy(common_var)
                                common_var_remain.remove(var1)
                                for var2 in common_var_remain:
                                    if (var1 not in (dag[var2]['par'] + dag[var2]['nei'])) & (
                                            var2 not in dag[var1]['par']):
                                        flag = True
                                        dag[tar]['nei'].remove(nei)
                                        dag[nei]['nei'].remove(tar)
                                        dag[nei]['par'].append(tar)
                                        break
                                if flag:
                                    break
        if dag_check == dag:
            break
    return dag


# random orient a CPDAG to a DAG
def random_orient(dag, no_v=None):
    undirected_fixed = []
    if no_v:
        if no_v[0] in dag[no_v[1]]['nei']:
            undirected_fixed.append(sorted([no_v[0], no_v[1]]))
        if no_v[0] in dag[no_v[2]]['nei']:
            undirected_fixed.append(sorted([no_v[0], no_v[2]]))
    undirected_edges = []
    for var in dag:
        for nei in dag[var]['nei']:
            edge = sorted([var, nei])
            if edge not in undirected_edges + undirected_fixed:
                undirected_edges.append(edge)
    random.shuffle(undirected_edges)
    undirected_edges = undirected_fixed + undirected_edges
    if no_v:
        undirected_fixed = [sorted([no_v[0], no_v[1]])]
        undirected_fixed.append(sorted([no_v[0], no_v[2]]))
    orient_state = []
    orient_history = []
    dag_temp = deepcopy(dag)
    index = 0
    while len(undirected_edges):
        edge = undirected_edges[index]
        sin_flag_temp, sin_direction_temp = sin_path_check(dag_temp, edge[0], edge[1])
        v_flag_temp, v_direction_temp = v_check(dag_temp, edge[0], edge[1])
        dag_temp[edge[0]]['nei'].remove(edge[1])
        dag_temp[edge[1]]['nei'].remove(edge[0])
        if (not sin_flag_temp) & (not v_flag_temp):
            if (edge in undirected_fixed) and (
                    (no_v[1] in dag_temp[no_v[0]]['par']) or (no_v[2] in dag_temp[no_v[0]]['par'])):
                if no_v[1] in edge:
                    dag_temp[no_v[1]]['par'].append(no_v[0])
                    if no_v[1] == edge[0]:
                        orient_history.append(0)
                    else:
                        orient_history.append(1)
                else:
                    dag_temp[no_v[2]]['par'].append(no_v[0])
                    if no_v[2] == edge[0]:
                        orient_history.append(0)
                    else:
                        orient_history.append(1)
                orient_state.append(1)
            else:
                orient_history.append(random.randint(0, 1))
                if orient_history[-1] == 0:
                    dag_temp[edge[0]]['par'].append(edge[1])
                else:
                    dag_temp[edge[1]]['par'].append(edge[0])
                orient_state.append(0)
            index += 1
        elif sin_flag_temp & (not v_flag_temp):
            dag_temp[sin_direction_temp[1]]['par'].append(sin_direction_temp[0])
            if sin_direction_temp[1] == edge[0]:
                orient_history.append(0)
            else:
                orient_history.append(1)
            orient_state.append(1)
            index += 1
        elif (not sin_flag_temp) & v_flag_temp & (v_direction_temp != 'both'):
            dag_temp[v_direction_temp[1]]['par'].append(v_direction_temp[0])
            if v_direction_temp[1] == edge[0]:
                orient_history.append(0)
            else:
                orient_history.append(1)
            orient_state.append(1)
            index += 1
        elif sin_flag_temp & v_flag_temp & (v_direction_temp == sin_direction_temp):
            dag_temp[v_direction_temp[1]]['par'].append(v_direction_temp[0])
            if v_direction_temp[1] == edge[0]:
                orient_history.append(0)
            else:
                orient_history.append(1)
            orient_state.append(1)
            index += 1
        else:
            sin_flag, sin_direction = sin_path_check(dag, edge[0], edge[1])
            v_flag, v_direction = v_check(dag, edge[0], edge[1])
            if (sin_flag and v_flag) or (v_direction == 'both'):
                if (edge in undirected_fixed) and (
                        (no_v[1] in dag_temp[no_v[0]]['par']) or (no_v[2] in dag_temp[no_v[0]]['par'])):
                    if no_v[1] in edge:
                        dag_temp[no_v[1]]['par'].append(no_v[0])
                        if no_v[1] == edge[0]:
                            orient_history.append(0)
                        else:
                            orient_history.append(1)
                    else:
                        dag_temp[no_v[2]]['par'].append(no_v[0])
                        if no_v[2] == edge[0]:
                            orient_history.append(0)
                        else:
                            orient_history.append(1)
                    orient_state.append(1)
                else:
                    orient_history.append(random.randint(0, 1))
                    if orient_history[-1] == 0:
                        dag_temp[edge[0]]['par'].append(edge[1])
                    else:
                        dag_temp[edge[1]]['par'].append(edge[0])
                    orient_state.append(0)
                index += 1
            else:
                if 0 in orient_state[::-1]:
                    last = len(orient_state) - 1 - orient_state[::-1].index(0)
                    dag_temp = deepcopy(dag)
                    orient_history_temp = []
                    for i in range(last):
                        edge = undirected_edges[i]
                        dag_temp[edge[0]]['nei'].remove(edge[1])
                        dag_temp[edge[1]]['nei'].remove(edge[0])
                        if orient_history[i] == 0:
                            dag_temp[edge[0]]['par'].append(edge[1])
                        else:
                            dag_temp[edge[1]]['par'].append(edge[0])
                        orient_history_temp.append(orient_history[i])
                    edge = undirected_edges[last]
                    dag_temp[edge[0]]['nei'].remove(edge[1])
                    dag_temp[edge[1]]['nei'].remove(edge[0])
                    if orient_history[last] == 0:
                        dag_temp[edge[1]]['par'].append(edge[0])
                        orient_history_temp.append(1)
                    else:
                        dag_temp[edge[0]]['par'].append(edge[1])
                        orient_history_temp.append(0)
                    index = last + 1
                    orient_state = orient_state[: last + 1]
                    orient_state[last] = 1
                    orient_history = deepcopy(orient_history_temp)
                else:
                    orient_history.append(random.randint(0, 1))
                    if orient_history[-1] == 0:
                        dag_temp[edge[0]]['par'].append(edge[1])
                    else:
                        dag_temp[edge[1]]['par'].append(edge[0])
                    orient_state.append(0)
                    index += 1

        if index == len(undirected_edges):
            break
    return dag_temp


# single direction path check
def sin_path_check(dag, var1, var2):
    sin_flag = False
    sin_direction = None
    # check single direction path var1 -> ... -> var2
    unchecked = deepcopy(dag[var2]['par'])
    checked = []
    while unchecked:
        if sin_flag:
            break
        unchecked_copy = deepcopy(unchecked)
        for dag_par in unchecked_copy:
            if var1 in dag[dag_par]['par']:
                sin_flag = True
                sin_direction = [var1, var2]
                break
            else:
                for key in dag[dag_par]['par']:
                    if key not in checked:
                        unchecked.append(key)
            unchecked.remove(dag_par)
            checked.append(dag_par)

    # check single direction path var2 -> ... -> var1
    if not sin_flag:
        unchecked = deepcopy(dag[var1]['par'])
        checked = []
        while unchecked:
            if sin_flag:
                break
            unchecked_copy = deepcopy(unchecked)
            for dag_par in unchecked_copy:
                if var2 in dag[dag_par]['par']:
                    sin_flag = True
                    sin_direction = [var2, var1]
                    break
                else:
                    for key in dag[dag_par]['par']:
                        if key not in checked:
                            unchecked.append(key)
                unchecked.remove(dag_par)
                checked.append(dag_par)
    return sin_flag, sin_direction


# v-structure check
def v_check(dag, var1, var2):
    v_flag1 = False
    v_flag2 = False
    if len(dag[var1]['par']):
        for par in dag[var1]['par']:
            if (var2 not in dag[par]['nei']) and (var2 not in dag[par]['par']) and (par not in dag[var2]['par']):
                v_flag1 = True
                break
        # dag_temp = deepcopy(dag)
        # if var1 in dag_temp[var2]['nei']:
        #     dag_temp[var2]['nei'].remove(var1)
        #     dag_temp[var1]['nei'].remove(var2)
        #     dag_temp[var1]['par'].append(var2)
        # if var1 in dag_temp[var2]['par']:
        #     dag_temp[var2]['par'].remove(var1)
        #     dag_temp[var1]['par'].append(var2)
        # if not v_flag1:
        #     for nei in dag[var1]['nei']:
        #         if (nei not in dag[var2]['nei'] + dag[var2]['par']) and (var2 not in dag[nei]['par']):
        #             v_flag1 = True
        #             for par in dag[var1]['par']:
        #                 if (nei not in dag[par]['par'] + dag[par]['nei']) and (par not in dag[nei]['par']):
        #                     v_flag1 = False
        #                     break
    if len(dag[var2]['par']):
        for par in dag[var2]['par']:
            if (var1 not in dag[par]['nei']) & (var1 not in dag[par]['par']) & (par not in dag[var1]['par']):
                v_flag2 = True
                break
        # dag_temp = deepcopy(dag)
        # if var1 in dag_temp[var2]['nei']:
        #     dag_temp[var2]['nei'].remove(var1)
        #     dag_temp[var1]['nei'].remove(var2)
        #     dag_temp[var2]['par'].append(var1)
        # if var2 in dag_temp[var1]['par']:
        #     dag_temp[var1]['par'].remove(var2)
        #     dag_temp[var2]['par'].append(var1)
        # if not v_flag2:
        #     for nei in dag[var2]['nei']:
        #         if (nei not in dag[var1]['nei'] + dag[var1]['par']) and (var1 not in dag[nei]['par']):
        #             v_flag2 = True
        #             for par in dag[var2]['par']:
        #                 if (nei not in dag[par]['par'] + dag[par]['nei']) and (par not in dag[nei]['par']):
        #                     v_flag2 = False
        #                     break
    if v_flag1 & (not v_flag2):
        return v_flag1, [var1, var2]
    elif (not v_flag1) & v_flag2:
        return v_flag2, [var2, var1]
    elif v_flag1 & v_flag2:
        return True, 'both'
    else:
        return False, None


# orient v-structrues
def orient_v_structure(data, counts, arities, varnames, pc, sepset, arities_wm=None, dof_type='ori', mod='normal',
                       threshold=0.05):
    dag = {}
    v_candidate = {}
    for var in pc:
        dag[var] = {}
        dag[var]['par'] = []
        dag[var]['nei'] = deepcopy(pc[var])
        if len(pc[var]) > 1:
            pc_copy = deepcopy(pc[var])
            for var1 in pc_copy:
                for var2 in pc_copy[pc_copy.index(var1) + 1:]:
                    if var2 in sepset[var1]:
                        if var not in sepset[var1][var2]:
                            cols = [varnames.index(var1), varnames.index(var2)] + \
                                   [varnames.index(node) for node in sepset[var1][var2]]
                            v_candidate[tuple([var, var1, var2])] = \
                                independence_test(data, counts, arities, cols, arities_wm, dof_type, mod, threshold)[1]
    while len(v_candidate):
        pair = min(v_candidate, key=v_candidate.get)
        if (pair[0] not in dag[pair[1]]['par']) & (pair[0] not in dag[pair[2]]['par']):
            if (sin_path_check(dag, pair[0], pair[1])[1] != [pair[0], pair[1]]) & \
                    (sin_path_check(dag, pair[0], pair[2])[1] != [pair[0], pair[2]]):
                if pair[1] in dag[pair[0]]['nei']:
                    dag[pair[0]]['par'].append(pair[1])
                    dag[pair[0]]['nei'].remove(pair[1])
                    dag[pair[1]]['nei'].remove(pair[0])
                if pair[2] in dag[pair[0]]['nei']:
                    dag[pair[0]]['par'].append(pair[2])
                    dag[pair[0]]['nei'].remove(pair[2])
                    dag[pair[2]]['nei'].remove(pair[0])
        del v_candidate[pair]
    return dag


# statistical test
def independence_test(data, counts, arities, cols, arities_wm=None, dof_type='ori', mod='normal', threshold=0.05):
    '''
    :param data: the unique datapoints as a 2-d array, each row is a datapoint, assumed unique
    :param counts: the count of how often each unique datapoint occurs in the original data
    :param arities: the arities of the variables (=columns) for the contingency table order must match that of `cols`.
    :param cols: the columns (=variables) for the marginal contingency table. columns must be ordered low to high
    :param threshold: the threshold for G-test

    :returns 1st element: test result
    :returns 2nd element: test score
    '''
    if arities_wm is None:
        arities_wm = arities
    G, dof = it_counter(data, counts, arities, cols, arities_wm, dof_type, mod)
    p = chi2.sf(G, dof)
    return p > threshold, p


@njit(fastmath=True)
def it_counter(data, counts, arities, cols, arities_wm, dof_type, mod):
    strides = np.empty(len(cols), dtype=np.uint32)
    idx = len(cols) - 1
    stride = 1
    while idx > -1:
        strides[idx] = stride
        if mod == 'normal':
            stride *= arities[cols[idx]]
        elif mod == 'test-delete':
            stride *= arities_wm[cols[idx]]
        else:
            raise Exception('Unexpected mode for dealing with noise.')
        idx -= 1
    N_ijk = np.zeros(stride)
    N_ik = np.zeros(stride)
    N_jk = np.zeros(stride)
    N_k = np.zeros(stride)
    for rowidx in range(data.shape[0]):
        idx_ijk = 0
        idx_ik = 0
        idx_jk = 0
        idx_k = 0
        skip = False
        for i in range(len(cols)):
            if (mod == 'test-delete') & (data[rowidx, cols[i]] == arities_wm[cols[i]]):
                skip = True
                break
            else:
                idx_ijk += data[rowidx, cols[i]] * strides[i]
                if i != 0:
                    idx_jk += data[rowidx, cols[i]] * strides[i]
                if i != 1:
                    idx_ik += data[rowidx, cols[i]] * strides[i]
                if (i != 0) & (i != 1):
                    idx_k += data[rowidx, cols[i]] * strides[i]
        if not skip:
            N_ijk[idx_ijk] += counts[rowidx]
            if mod == 'normal':
                for j in range(arities[cols[1]]):
                    N_ik[idx_ik + j * strides[1]] += counts[rowidx]
                for i in range(arities[cols[0]]):
                    N_jk[idx_jk + i * strides[0]] += counts[rowidx]
                for i in range(arities[cols[0]]):
                    for j in range(arities[cols[1]]):
                        N_k[idx_k + i * strides[0] + j * strides[1]] += counts[rowidx]
            elif mod == 'test-delete':
                for j in range(arities_wm[cols[1]]):
                    N_ik[idx_ik + j * strides[1]] += counts[rowidx]
                for i in range(arities_wm[cols[0]]):
                    N_jk[idx_jk + i * strides[0]] += counts[rowidx]
                for i in range(arities_wm[cols[0]]):
                    for j in range(arities_wm[cols[1]]):
                        N_k[idx_k + i * strides[0] + j * strides[1]] += counts[rowidx]
            else:
                raise Exception('Unexpected mode for dealing with misisng value.')
    G = 0
    for i in range(stride):
        if N_ijk[i] != 0:
            G += 2 * N_ijk[i] * np.log(N_ijk[i] * N_k[i] / N_ik[i] / N_jk[i])

    if mod == 'normal':
        if dof_type == 'ori':
            dof = max((arities[cols[0]] - 1) * (arities[cols[1]] - 1) * strides[1], 1)
        elif dof_type == 'enp':
            dof = max(np.sum(N_ijk != 0) - np.sum(N_ik != 0) / arities[cols[1]] - np.sum(N_jk != 0) / arities[cols[0]] + \
                      np.sum(N_k != 0) / arities[cols[0]] / arities[cols[1]], 1)
        else:
            raise Exception('Unexpected type of degree of freedom.')
    elif mod == 'test-delete':
        if dof_type == 'ori':
            dof = max((arities_wm[cols[0]] - 1) * (arities_wm[cols[1]] - 1) * strides[1], 1)
        elif dof_type == 'enp':
            dof = max(
                np.sum(N_ijk != 0) - np.sum(N_ik != 0) / arities_wm[cols[1]] - np.sum(N_jk != 0) / arities_wm[cols[0]] + \
                np.sum(N_k != 0) / arities_wm[cols[0]] / arities_wm[cols[1]], 1)
        else:
            raise Exception('Unexpected type of degree of freedom.')
    else:
        raise Exception('Unexpected mode for dealing with missing value.')
    return G, dof


# convert the dag to bnlearn format
def to_bnlearn(dag):
    output = ''
    for var in dag:
        output += '[' + var
        if dag[var]['par']:
            output += '|'
            for par in dag[var]['par']:
                output += par + ':'
            output = output[:-1]
        output += ']'
    return output
