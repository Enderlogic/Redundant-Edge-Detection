from copy import deepcopy
import lib.score as score
from lib.accessory import random_orient, cpdag, skeleton
import random
from time import time

def sed(dag, data, counts, arities, varnames, score_function='bic'):
    '''

    :param dag: the original learned graph
    :param data: input training data (unique version)
    :param counts: the counts of unique observation
    :param arities: the number of states for each variable
    :param varnames: the name of variables
    :param score_function: the type of score function (default = bic)
    :return: corrected CPDAG
    '''
    random.seed(990806)
    bic = score.score(random_orient(dag), data, counts, arities, varnames, score_function)
    tuple_list = {}
    dag_corrected = deepcopy(cpdag(dag))
    ske = skeleton(dag)
    for var in ske:
        for mid in ske[var]:
            for end in ske[mid]:
                if end in ske[var]:
                    edge = sorted([mid, end])
                    if var not in tuple_list:
                        tuple_list[var] = [edge]
                    elif edge not in tuple_list[var]:
                        tuple_list[var].append(edge)
    edge_dict = {}
    edge_list = []
    for var in tuple_list:
        for edge in tuple_list[var]:
            edge_list.append([var] + edge)
            dag_attempt = deepcopy(cpdag(dag))
            if (edge[0] not in dag_attempt[var]['par']) | (edge[1] not in dag_attempt[var]['par']):
                dag_attempt = random_orient(dag_attempt, no_v=[var] + edge)
                if edge[0] in dag_attempt[edge[1]]['par']:
                    dag_attempt[edge[1]]['par'].remove(edge[0])
                else:
                    dag_attempt[edge[0]]['par'].remove(edge[1])
                edge_dict[edge_list.index([var] + edge)] = score.em_weaker(dag_attempt, data, counts, arities, varnames, var) - bic
    if bool(edge_dict):
        while max(edge_dict.values()) > 0:
            bic_temp = bic + max(edge_dict.values())
            dag_temp = deepcopy(cpdag(dag))
            edge_id = max(edge_dict, key = lambda x: edge_dict[x])
            del edge_dict[edge_id]
            var = edge_list[edge_id][0]
            edge = edge_list[edge_id][1 : ]
            if edge[0] in dag_temp[edge[1]]['par']:
                dag_temp[edge[1]]['par'].remove(edge[0])
            elif edge[1] in dag_temp[edge[0]]['par']:
                dag_temp[edge[0]]['par'].remove(edge[1])
            else:
                dag_temp[edge[1]]['nei'].remove(edge[0])
                dag_temp[edge[0]]['nei'].remove(edge[1])
            if (edge[0] in dag_temp[var]['par']) and (edge[1] in dag_temp[var]['par']):
                dag_temp[var]['par'].remove(edge[0])
                dag_temp[var]['par'].remove(edge[1])
                dag_temp[var]['nei'].extend(edge)
                dag_temp[edge[0]]['nei'].append(var)
                dag_temp[edge[1]]['nei'].append(var)
            if edge[0] in dag_corrected[edge[1]]['par']:
                dag_corrected[edge[1]]['par'].remove(edge[0])
            elif edge[1] in dag_corrected[edge[0]]['par']:
                dag_corrected[edge[0]]['par'].remove(edge[1])
            else:
                dag_corrected[edge[1]]['nei'].remove(edge[0])
                dag_corrected[edge[0]]['nei'].remove(edge[1])
            if (edge[0] in dag_corrected[var]['par']) and (edge[1] in dag_corrected[var]['par']):
                dag_corrected[var]['par'].remove(edge[0])
                dag_corrected[var]['par'].remove(edge[1])
                dag_corrected[var]['nei'].extend(edge)
                dag_corrected[edge[0]]['nei'].append(var)
                dag_corrected[edge[1]]['nei'].append(var)
            tuple_list[var].remove(edge)
            while len(tuple_list[var]):
                bic_candidate = [0] * len(tuple_list[var])
                for edge in tuple_list[var]:
                    dag_attempt = deepcopy(dag_temp)
                    if (edge[0] not in dag_attempt[var]['par']) | (edge[1] not in dag_attempt[var]['par']):
                        dag_attempt = random_orient(dag_attempt, no_v=[var] + edge)
                        if edge[0] in dag_attempt[edge[1]]['par']:
                            dag_attempt[edge[1]]['par'].remove(edge[0])
                        else:
                            dag_attempt[edge[0]]['par'].remove(edge[1])
                        bic_candidate[tuple_list[var].index(edge)] = score.em_weaker(dag_attempt, data, counts, arities,
                                                                                     varnames, var) - bic_temp
                if max(bic_candidate) > 0:
                    bic_temp += max(bic_candidate)
                    id = bic_candidate.index(max(bic_candidate))
                    edge = tuple_list[var][id]
                    if edge[0] in dag_temp[edge[1]]['par']:
                        dag_temp[edge[1]]['par'].remove(edge[0])
                    elif edge[1] in dag_temp[edge[0]]['par']:
                        dag_temp[edge[0]]['par'].remove(edge[1])
                    else:
                        dag_temp[edge[1]]['nei'].remove(edge[0])
                        dag_temp[edge[0]]['nei'].remove(edge[1])
                    if (edge[0] in dag_temp[var]['par']) and (edge[1] in dag_temp[var]['par']):
                        dag_temp[var]['par'].remove(edge[0])
                        dag_temp[var]['par'].remove(edge[1])
                        dag_temp[var]['nei'].extend(edge)
                        dag_temp[edge[0]]['nei'].append(var)
                        dag_temp[edge[1]]['nei'].append(var)
                    if edge[0] in dag_corrected[edge[1]]['par']:
                        dag_corrected[edge[1]]['par'].remove(edge[0])
                    elif edge[1] in dag_corrected[edge[0]]['par']:
                        dag_corrected[edge[0]]['par'].remove(edge[1])
                    else:
                        dag_corrected[edge[1]]['nei'].remove(edge[0])
                        dag_corrected[edge[0]]['nei'].remove(edge[1])
                    if (edge[0] in dag_corrected[var]['par']) and (edge[1] in dag_corrected[var]['par']):
                        dag_corrected[var]['par'].remove(edge[0])
                        dag_corrected[var]['par'].remove(edge[1])
                        dag_corrected[var]['nei'].extend(edge)
                        dag_corrected[edge[0]]['nei'].append(var)
                        dag_corrected[edge[1]]['nei'].append(var)
                    del edge_dict[edge_list.index([var] + tuple_list[var][id])]
                    tuple_list[var].pop(id)
                else:
                    for edge in tuple_list[var]:
                        if edge_list.index([var] + edge) in edge_dict:
                            del edge_dict[edge_list.index([var] + edge)]
                    del tuple_list[var]
                    break
            ske = skeleton(dag_temp)
            for var in tuple_list:
                removed_tuple = []
                for edge in tuple_list[var]:
                    if (edge[0] not in ske[edge[1]]) or (edge[0] not in ske[var]) or (edge[1] not in ske[var]):
                        removed_tuple.append(edge)
                for edge in removed_tuple:
                    tuple_list[var].remove(edge)
                    if edge_list.index([var] + edge) in edge_dict:
                        del edge_dict[edge_list.index([var] + edge)]
            if not bool(edge_dict):
                break
    return dag_corrected