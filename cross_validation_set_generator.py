import numpy as np


def get_cross_validation_sets(original: list, folds: int):
    permuted = np.random.permutation(original)
    fold_size = int(len(original)/folds)

    out = []
    for i in range(folds):
        out.append((np.concatenate([permuted[:i*fold_size],
                                   permuted[(i+1)*fold_size:]]),
                    permuted[i*fold_size:(i+1)*fold_size]))
    return out


if __name__=='__main__':
    print(get_cross_validation_sets(list(range(10)), 5))