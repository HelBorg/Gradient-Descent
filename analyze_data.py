import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

BLOCKS_NUM = 5


def train_data(dataset_df):
    dataset_df.drop(columns=[37])

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    features = scaler.fit_transform(dataset_df.values[:, :-1])
    target = dataset_df.values[:, -1]
    dataset = np.concatenate([features, np.array([target]).T], axis=1)

    results = cross_validate(dataset)
    analyze_results(results)


def cross_validate(dataset):
    dataset = np.c_[np.ones(dataset.shape[0]), dataset]
    blocks = np.array_split(dataset[:], BLOCKS_NUM)
    results = []
    for i in range(BLOCKS_NUM):
        block = blocks[i]
        blocks_to_train = np.concatenate([x for x in blocks if x is not block])
        weight = gradient_descent(blocks_to_train)

        result = generate_result(block, blocks_to_train, i, weight)
        results.append(result)
    return results


def gradient_descent(block):
    features = block[:, :-1]
    target = block[:, -1]
    n = block.shape[0]
    weight_k = np.ones(features.shape[1])
    iteration = 1
    Q = 0
    while Q < 0.3 and iteration < 10000:
        loss = np.dot(features, weight_k) - target
        grad = np.dot(features.T, loss) / np.sqrt(n * np.sum(loss ** 2))
        weight_k = weight_k - grad / iteration
        Q = count_r_2(features, target, weight_k)
        iteration += 1
    return weight_k


def generate_result(block, blocks_to_train, i, weight):
    features_train = blocks_to_train[:, :-1]
    target_train = blocks_to_train[:, -1]
    rmse_train = count_rmse(features_train, target_train, weight)
    r_2_train = count_r_2(features_train, target_train, weight)

    features_test = block[:, :-1]
    target_test = block[:, -1]
    r_2_test = count_r_2(features_test, target_test, weight)
    rmse_test = count_rmse(features_test, target_test, weight)

    result = [r_2_test, rmse_test, r_2_train, rmse_train]
    result.extend(weight)
    return result


def analyze_results(results):
    mean = np.mean(results, axis=0).tolist()
    var = np.var(results, axis=0, ddof=0).tolist()
    results.append(mean)
    results.append(var)
    columns = ['r2 test', 'RMSE test', 'r2 train', 'RMSE train']
    columns.extend(['f' + str(i) for i in range(54)])
    results_df = pd.DataFrame(results,
                              index=['T1', 'T2', 'T3', 'T4', 'T5', 'E', 'STD'],
                              columns=columns)
    print(tabulate(results_df, headers='keys', tablefmt='psql'))


def count_rmse(features, target, weight):
    n = features.shape[0]
    loss = np.dot(features, weight) - target
    Q = np.sqrt(np.sum(loss ** 2)) / n
    return Q


def count_r_2(features, target, weight):
    loss = np.dot(features, weight) - target
    disp = target - np.mean(target)
    r_2 = 1 - np.sum(loss ** 2) / np.sum(disp ** 2)
    return r_2


if __name__ == '__main__':
    dataset = pd.read_csv("data/Features_TrainingData.csv", header=None)
    train_data(dataset)
