import numpy as np
from numba import njit
from numba.typed import Dict
from numba.core import types
from copy import deepcopy


def score(dag, data, counts, arities, varnames, score_function):
    score = 0
    if score_function == 'bic':
        for tar in dag:
            cols = [varnames.index(tar)]
            for var in dag[tar]['par']:
                cols.append(varnames.index(var))
            cols = np.asarray(cols)
            score += bic_counter(data, counts, arities, cols)
    return score


@njit
def bic_counter(data, counts, arities, cols):
    strides = np.empty(len(cols), dtype=np.uint32)
    idx = len(cols) - 1
    stride = 1
    while idx > -1:
        strides[idx] = stride
        stride *= arities[cols[idx]]
        idx -= 1
    N_ijk = np.zeros(stride)
    N_ij = np.zeros(stride)
    for rowidx in range(data.shape[0]):
        idx_ijk = 0
        idx_ij = 0
        for i in range(len(cols)):
            idx_ijk += data[rowidx, cols[i]] * strides[i]
            if i != 0:
                idx_ij += data[rowidx, cols[i]] * strides[i]
        N_ijk[idx_ijk] += counts[rowidx]
        for i in range(arities[cols[0]]):
            N_ij[idx_ij + i * strides[0]] += counts[rowidx]
    bic = 0
    for i in range(stride):
        if N_ijk[i] != 0:
            bic += N_ijk[i] * np.log(N_ijk[i] / N_ij[i])

    bic -= 0.5 * np.log(np.sum(counts)) * (arities[cols[0]] - 1) * strides[0]
    return bic


@njit
def counter(data, counts, arities, cols):
    strides = np.empty(len(cols), dtype=np.uint32)
    idx = len(cols) - 1
    stride = 1
    while idx > -1:
        strides[idx] = stride
        stride *= arities[cols[idx]]
        idx -= 1
    N_ijk = np.zeros(stride)
    N_ij = np.zeros(stride)
    for rowidx in range(data.shape[0]):
        idx_ijk = 0
        idx_ij = 0
        for i in range(len(cols)):
            idx_ijk += data[rowidx, cols[i]] * strides[i]
            if i != 0:
                idx_ij += data[rowidx, cols[i]] * strides[i]
        N_ijk[idx_ijk] += counts[rowidx]
        for i in range(arities[cols[0]]):
            N_ij[idx_ij + i * strides[0]] += counts[rowidx]
    for i in range(stride):
        if N_ijk[i] != 0:
            N_ijk[i] /= N_ij[i]
    return N_ijk


@njit
def em_counter(data, counts, arities, prob_fitted, onetime_set, multime_set):
    loglik = 0
    prob_update = {}
    for varidx in range(len(arities)):
        if varidx in onetime_set:
            loglik += - 0.5 * np.log(counts.sum()) * (arities[varidx] - 1) * np.prod(arities[onetime_set[varidx][1:]])
        else:
            loglik += - 0.5 * np.log(counts.sum()) * (arities[varidx] - 1) * np.prod(arities[multime_set[varidx][1:]])
            prob_update[varidx] = np.zeros(len(prob_fitted[varidx]))

    for rowidx in range(data.shape[0]):
        prob_joint = 1
        for varidx in onetime_set:
            cols = onetime_set[varidx]
            strides = np.empty(len(cols), dtype=np.uint32)
            idx = len(cols) - 1
            stride = 1
            while idx > -1:
                strides[idx] = stride
                stride *= arities[cols[idx]]
                idx -= 1
            idx_ijk = 0
            for i in range(len(cols)):
                idx_ijk += data[rowidx, cols[i]] * strides[i]
            prob_joint *= prob_fitted[varidx][idx_ijk]

        prob_joint = np.ones(np.prod(arities[data.shape[1]:])) * prob_joint

        data_ite = np.zeros(len(arities) - data.shape[1], dtype=np.uint32)
        index = 0
        while True:
            data_temp = np.append(data[rowidx, :], data_ite)
            for varidx in multime_set:
                cols = multime_set[varidx]
                strides = np.empty(len(cols), dtype=np.uint32)
                idx = len(cols) - 1
                stride = 1
                while idx > -1:
                    strides[idx] = stride
                    stride *= arities[cols[idx]]
                    idx -= 1
                idx_ijk = 0
                for i in range(len(cols)):
                    idx_ijk += data_temp[cols[i]] * strides[i]
                prob_joint[index] *= prob_fitted[varidx][idx_ijk]
            index += 1
            data_ite[-1] += 1
            for i in range(len(data_ite) - 1):
                if data_ite[- (i + 1)] == arities[- (i + 1)]:
                    data_ite[- (i + 1)] = 0
                    data_ite[- (i + 2)] += 1
            if data_ite[0] == arities[data.shape[1]]:
                break
        loglik += counts[rowidx] * np.log(np.sum(prob_joint))

        prob_joint /= np.sum(prob_joint)
        data_ite = np.zeros(len(arities) - data.shape[1], dtype=np.uint32)
        index = 0
        while True:
            data_temp = np.append(data[rowidx, :], data_ite)
            for varidx in prob_update:
                cols = multime_set[varidx]
                strides = np.empty(len(cols), dtype=np.uint32)
                idx = len(cols) - 1
                stride = 1
                while idx > -1:
                    strides[idx] = stride
                    stride *= arities[cols[idx]]
                    idx -= 1
                idx_ijk = 0
                for i in range(len(cols)):
                    idx_ijk += data_temp[cols[i]] * strides[i]
                prob_update[varidx][idx_ijk] += prob_joint[index] * counts[rowidx]
            index += 1
            data_ite[-1] += 1
            for i in range(len(data_ite) - 1):
                if data_ite[- (i + 1)] == arities[- (i + 1)]:
                    data_ite[- (i + 1)] = 0
                    data_ite[- (i + 2)] += 1
            if data_ite[0] == arities[data.shape[1]]:
                break
    return loglik, prob_update


def em(dag, prob_fitted_input, data, counts, arities, varnames, noise_node, noise_list, score_function='bic'):
    varnames_append = deepcopy(varnames)
    for latent in noise_list:
        if latent != noise_node:
            varnames_append.append(latent + '_h')
            arities = np.append(arities, (arities[varnames.index(latent)]))
    varnames_append.append(noise_node + '_h')
    arities = np.append(arities, (arities[varnames.index(noise_node)]))
    parents = Dict.empty(key_type=types.uint32, value_type=types.int32[:], )
    onetime_set = []

    if prob_fitted_input is None:
        prob_fitted = [None] * len(varnames_append)
        for var in varnames:
            cols = [varnames.index(var)]
            for par in dag[var]['par']:
                cols.append(varnames.index(par))
            cols = np.asarray(cols)
            if var != noise_node:
                prob_fitted[varnames_append.index(var)] = counter(data, counts, arities, cols)
                if noise_node in dag[var]['par']:
                    cols[dag[var]['par'].index(noise_node) + 1] = varnames_append.index(noise_node + '_h')
                else:
                    onetime_set.append(varnames.index(var))
                parents[varnames_append.index(var)] = np.asarray(cols[1:], dtype='int32')
            else:
                prob_fitted[varnames_append.index(var + '_h')] = counter(data, counts, arities, cols)
                parents[varnames_append.index(var + '_h')] = np.asarray(cols[1:], dtype='int32')
                level = 0.1
                n = arities[varnames_append.index(var)]
                prob_fitted[varnames_append.index(var)] = level / (n - 1) * np.ones((n, n)) + (
                        1 - level * n / (n - 1)) * np.identity(n)
                prob_fitted[varnames_append.index(var)] = prob_fitted[varnames_append.index(var)].flatten()
                parents[varnames_append.index(var)] = np.asarray([varnames_append.index(var + '_h')], dtype='int32')
    else:
        prob_fitted = deepcopy(prob_fitted_input)
        for var in varnames:
            cols = [varnames.index(var)]
            for par in dag[var]['par']:
                cols.append(varnames.index(par))
            cols = np.asarray(cols)
            if (var != noise_node) and (var not in noise_list):
                if np.prod(arities[cols]) != prob_fitted[varnames.index(var)].size:
                    prob_fitted[varnames.index(var)] = counter(data, counts, arities, cols)
                for latent in noise_list:
                    if latent in dag[var]['par']:
                        cols[dag[var]['par'].index(latent) + 1] = varnames_append.index(latent + '_h')
                if noise_node in dag[var]['par']:
                    cols[dag[var]['par'].index(noise_node) + 1] = varnames_append.index(noise_node + '_h')
                if max(cols) < len(varnames):
                    onetime_set.append(varnames.index(var))
                parents[varnames_append.index(var)] = np.asarray(cols[1:], dtype='int32')
            else:
                if varnames_append.index(var + '_h') != len(prob_fitted):
                    if np.prod(arities[cols]) != prob_fitted[varnames_append.index(var + '_h')].size:
                        prob_fitted[varnames_append.index(var + '_h')] = counter(data, counts, arities, cols)
                else:
                    prob_fitted.append(prob_fitted[varnames.index(var)])
                    level = 0.1
                    n = arities[varnames.index(var)]
                    prob_fitted[varnames.index(var)] = level / (n - 1) * np.ones((n, n)) + (
                            1 - level * n / (n - 1)) * np.identity(n)
                    prob_fitted[varnames.index(var)] = prob_fitted[varnames.index(var)].flatten()
                for latent in noise_list:
                    if latent in dag[var]['par']:
                        cols[dag[var]['par'].index(latent) + 1] = varnames_append.index(latent + '_h')
                if noise_node in dag[var]['par']:
                    cols[dag[var]['par'].index(noise_node) + 1] = varnames_append.index(noise_node + '_h')
                parents[varnames_append.index(var + '_h')] = np.asarray(cols[1:], dtype='int32')
                parents[varnames.index(var)] = np.asarray([varnames_append.index(var + '_h')], dtype='int32')
    multime_set = np.asarray(list(set(list(range(len(varnames_append)))) - set(onetime_set)))
    onetime_set = np.asarray(onetime_set)
    loglik = float('-inf')
    while True:
        # start = time()
        loglik_update, prob_update = em_counter(data, counts, arities, prob_fitted, parents, onetime_set, multime_set)
        for varidx in prob_update:
            cols = [varidx]
            cols = cols + list(parents[varidx])
            if varidx != varnames.index(noise_node):
                prob_update_single = prob_update[varidx].reshape(arities[cols])
                prob_update_single /= prob_update_single.sum(0)
                prob_fitted[varidx] = prob_update_single.flatten()
            else:
                level = 1 - np.diag(prob_update[varidx].reshape((arities[varidx], arities[varidx]))).sum() / np.sum(
                    counts)
                n = arities[varidx]
                prob_fitted[varidx] = level / (n - 1) * np.ones((n, n)) + (1 - level * n / (n - 1)) * np.identity(n)
                prob_fitted[varidx] = prob_fitted[varidx].flatten()
        if loglik_update - loglik > 1e-03:
            loglik = loglik_update
        else:
            break
    return loglik, prob_fitted


def em_weaker(dag, data, counts, arities, varnames, noise_node, score_function='bic'):
    varnames_append = deepcopy(varnames)
    varnames_append.append(noise_node + '_h')
    arities = np.append(arities, (arities[varnames.index(noise_node)]))
    multime_list = []

    prob_fitted_list = []
    bic = 0
    for var in varnames:
        cols = [varnames.index(var)]
        for par in dag[var]['par']:
            cols.append(varnames.index(par))
        if var != noise_node:
            if noise_node in dag[var]['par']:
                prob_fitted_list.append(counter(data, counts, arities, cols))
                cols[dag[var]['par'].index(noise_node) + 1] = len(varnames)
                multime_list.append(cols)
            else:
                bic += bic_counter(data, counts, arities, cols)
        else:
            prob_fitted_list.append(counter(data, counts, arities, cols))
            multime_list.append([len(varnames)] + cols[1:])
            level = 0.1
            n = arities[varnames.index(var)]
            CPT = level / (n - 1) * np.ones((n, n)) + (1 - level * n / (n - 1)) * np.identity(n)
            prob_fitted_list.append(CPT.flatten())
            multime_list.append([varnames.index(var), len(varnames)])

    multime_set = - np.ones((len(multime_list), len(max(multime_list, key=len))), dtype='int32')
    prob_fitted = np.zeros((len(prob_fitted_list), len(max(prob_fitted_list, key=len))))
    cols_num = np.zeros(len(multime_list), dtype = 'uint32')
    for i in range(len(multime_list)):
        multime_set[i, : len(multime_list[i])] = multime_list[i]
        cols_num[i] = len(multime_list[i])
        prob_fitted[i, : len(prob_fitted_list[i])] = prob_fitted_list[i]
    score = float('-inf')
    while True:
        # start = time()
        score_update, prob_update = em_counter_one_hidden(data, counts, arities, prob_fitted, multime_set, cols_num)
        score_update += bic
        # end = time()
        # print(end - start)
        for varidx in range(len(multime_list)):
            cols = multime_list[varidx]
            # if cols[0] != varnames.index(noise_node):
            prob_update_single = prob_update[varidx, : np.prod(arities[cols])].reshape(arities[cols])
            prob_update_single /= prob_update_single.sum(0)
            prob_fitted[varidx, : np.prod(arities[cols])] = prob_update_single.flatten()
            # else:
            #     level = 1 - np.diag(prob_update[varidx, : np.prod(arities[cols])].reshape((arities[cols[0]], arities[cols[0]]))).sum() / np.sum(counts)
            #     n = arities[cols[0]]
            #     prob_update_single = level / (n - 1) * np.ones((n, n)) + (1 - level * n / (n - 1)) * np.identity(n)
            #     prob_fitted[varidx, : np.prod(arities[cols])] = prob_update_single.flatten()
        if score_update - score > abs(score_update * 1e-6):
            score = score_update
        else:
            break
    return score


@njit(fastmath = True)
def em_counter_one_hidden(data, counts, arities, prob_fitted, multime_set, cols_num):
    score = 0
    prob_update = np.zeros(prob_fitted.shape)
    arities_len = len(arities)

    for varidx in range(len(multime_set)):
        cols = multime_set[varidx, : cols_num[varidx]]
        score += (arities[cols[0]] - 1) * np.prod(arities[cols[1 :]])
    score *= - 0.5 * np.log(counts.sum())

    for rowidx in range(data.shape[0]):
        prob_joint = np.ones(arities[-1])
        for sta in range(arities[-1]):
            for varidx in range(len(multime_set)):
                cols = multime_set[varidx, : cols_num[varidx]]
                strides = np.empty(len(cols), dtype=np.uint32)
                idx = len(cols) - 1
                stride = 1
                while idx > -1:
                    strides[idx] = stride
                    stride *= arities[cols[idx]]
                    idx -= 1
                idx_ijk = 0
                for i in range(len(cols)):
                    if cols[i] != arities_len - 1:
                        idx_ijk += data[rowidx, cols[i]] * strides[i]
                    else:
                        idx_ijk += sta * strides[i]
                prob_joint[sta] *= prob_fitted[varidx, idx_ijk]
        score += counts[rowidx] * (np.log(np.sum(prob_joint)))

        prob_joint /= np.sum(prob_joint)
        for sta in range(arities[-1]):
            for varidx in range(len(multime_set)):
                cols = multime_set[varidx, : cols_num[varidx]]
                strides = np.empty(len(cols), dtype=np.uint32)
                idx = len(cols) - 1
                stride = 1
                while idx > -1:
                    strides[idx] = stride
                    stride *= arities[cols[idx]]
                    idx -= 1
                idx_ijk = 0
                for i in range(len(cols)):
                    if cols[i] != arities_len - 1:
                        idx_ijk += data[rowidx, cols[i]] * strides[i]
                    else:
                        idx_ijk += sta * strides[i]
                prob_update[varidx, idx_ijk] += prob_joint[sta] * counts[rowidx]
    return score, prob_update