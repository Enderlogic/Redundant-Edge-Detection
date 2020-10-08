# evaluation methods
def compare(target, current):
    '''

    :param target: true CPDAG
    :param current: learned CPDAG
    :return: a dictionary contains TP, FP, FN and f1
    '''

    compare_dict = {}
    tp = 0
    fp = 0
    fn = 0
    for key, value in target.items():
        for par in value['par']:
            if par in current[key]['par']:
                tp = tp + 1
            else:
                fn = fn + 1
        for nei in value['nei']:
            if nei in current[key]['nei']:
                tp = tp + 0.5
            else:
                fn = fn + 0.5
    for key, value in current.items():
        for par in value['par']:
            if par not in target[key]['par']:
                fp = fp + 1
        for nei in value['nei']:
            if nei not in target[key]['nei']:
                fp = fp + 0.5

    compare_dict['tp'] = tp
    compare_dict['fp'] = fp
    compare_dict['fn'] = fn
    compare_dict['f1'] = 2 * tp / (2 * tp + fp + fn)
    return compare_dict


def shd(target, current):
    '''

    :param target: true CPDAG
    :param current: learned CPDAG
    :return: the SHD score of learned graph
    '''
    shd = 0
    for key, value in target.items():
        for par in value['par']:
            if par not in current[key]['par']:
                shd = shd + 1
        for nei in value['nei']:
            if nei not in current[key]['nei']:
                shd = shd + 0.5
    for key, value in current.items():
        for par in value['par']:
            if (par not in target[key]['par']) and (par not in target[key]['nei']) and (key not in target[par]['par']):
                shd = shd + 1
        for nei in value['nei']:
            if (nei not in target[key]['nei']) and (nei not in target[key]['par']) and (key not in target[nei]['par']):
                shd = shd + 0.5
    return int(shd)