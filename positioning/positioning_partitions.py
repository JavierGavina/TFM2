import pandas as pd
import sklearn as sk
import numpy as np
import sys
import os
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

sys.path.append("..")

from src.constants import constants
from positioning.utils import get_euclid_per_coord

part_5vs18 = f"../{constants.data.partitions.PARTITION_5VS18}"
part_10vs13 = f"../{constants.data.partitions.PARTITION_10VS13}"
part_15vs8 = f"../{constants.data.partitions.PARTITION_15VS8}"

dict_inv = {v: k for k, v in constants.labels_dictionary_meters.items()}

# print dataframe options to display all columns
pd.set_option('display.max_columns', None)

# print dataframe options to display all rows
pd.set_option('display.max_rows', None)


def main():
    path_partitions_output = "../outputs/positioning_partitions"
    os.makedirs(path_partitions_output, exist_ok=True)

    path_to_processed_radiomap = "processed/processed_radiomap.csv"
    train_5vs18 = pd.read_csv(f"{part_5vs18}/train/{path_to_processed_radiomap}")
    train_10vs13 = pd.read_csv(f"{part_10vs13}/train/{path_to_processed_radiomap}")
    train_15vs8 = pd.read_csv(f"{part_15vs8}/train/{path_to_processed_radiomap}")
    test_5vs18 = pd.read_csv(f"{part_5vs18}/test/{path_to_processed_radiomap}")
    test_10vs13 = pd.read_csv(f"{part_10vs13}/test/{path_to_processed_radiomap}")
    test_15vs8 = pd.read_csv(f"{part_15vs8}/test/{path_to_processed_radiomap}")

    Xtrain_5vs18, ytrain_5vs18 = train_5vs18[constants.aps].to_numpy(), train_5vs18[
        ["Longitude", "Latitude"]].to_numpy()
    Xtest_5vs18, ytest_5vs18 = test_5vs18[constants.aps].to_numpy(), test_5vs18[["Longitude", "Latitude"]].to_numpy()

    Xtrain_10vs13, ytrain_10vs13 = train_10vs13[constants.aps].to_numpy(), train_10vs13[
        ["Longitude", "Latitude"]].to_numpy()
    Xtest_10vs13, ytest_10vs13 = test_10vs13[constants.aps].to_numpy(), test_10vs13[
        ["Longitude", "Latitude"]].to_numpy()

    Xtrain_15vs8, ytrain_15vs8 = train_15vs8[constants.aps].to_numpy(), train_15vs8[
        ["Longitude", "Latitude"]].to_numpy()
    Xtest_15vs8, ytest_15vs8 = test_15vs8[constants.aps].to_numpy(), test_15vs8[["Longitude", "Latitude"]].to_numpy()

    knn1_5vs18 = sk.neighbors.KNeighborsRegressor(n_neighbors=1, n_jobs=cpu_count() - 3).fit(Xtrain_5vs18, ytrain_5vs18)
    knn1_10vs13 = sk.neighbors.KNeighborsRegressor(n_neighbors=1, n_jobs=cpu_count() - 3).fit(Xtrain_10vs13,
                                                                                              ytrain_10vs13)
    knn1_15vs8 = sk.neighbors.KNeighborsRegressor(n_neighbors=1, n_jobs=cpu_count() - 3).fit(Xtrain_15vs8, ytrain_15vs8)

    knn5_5vs18 = sk.neighbors.KNeighborsRegressor(n_neighbors=5, n_jobs=cpu_count() - 3).fit(Xtrain_5vs18, ytrain_5vs18)
    knn5_10vs13 = sk.neighbors.KNeighborsRegressor(n_neighbors=5, n_jobs=cpu_count() - 3).fit(Xtrain_10vs13,
                                                                                              ytrain_10vs13)
    knn5_15vs8 = sk.neighbors.KNeighborsRegressor(n_neighbors=5, n_jobs=cpu_count() - 3).fit(Xtrain_15vs8, ytrain_15vs8)

    rf_5vs18 = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count() - 3).fit(Xtrain_5vs18,
                                                                                               ytrain_5vs18)
    rf_10vs13 = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count() - 3).fit(Xtrain_10vs13,
                                                                                                ytrain_10vs13)
    rf_15vs8 = sk.ensemble.RandomForestRegressor(n_estimators=500, n_jobs=cpu_count() - 3).fit(Xtrain_15vs8,
                                                                                               ytrain_15vs8)

    knn1_5vs18_pred = knn1_5vs18.predict(Xtest_5vs18)
    knn1_10vs13_pred = knn1_10vs13.predict(Xtest_10vs13)
    knn1_15vs8_pred = knn1_15vs8.predict(Xtest_15vs8)

    knn5_5vs18_pred = knn5_5vs18.predict(Xtest_5vs18)
    knn5_10vs13_pred = knn5_10vs13.predict(Xtest_10vs13)
    knn5_15vs8_pred = knn5_15vs8.predict(Xtest_15vs8)

    rf_5vs18_pred = rf_5vs18.predict(Xtest_5vs18)
    rf_10vs13_pred = rf_10vs13.predict(Xtest_10vs13)
    rf_15vs8_pred = rf_15vs8.predict(Xtest_15vs8)

    _, knn1_mean_euclid_5vs18, knn1_std_euclid_5vs18 = get_euclid_per_coord(ytest_5vs18, knn1_5vs18_pred)

    _, knn1_mean_euclid_10vs13, knn1_std_euclid_10vs13 = get_euclid_per_coord(ytest_10vs13, knn1_10vs13_pred)

    _, knn1_mean_euclid_15vs8, knn1_std_euclid_15vs8 = get_euclid_per_coord(ytest_15vs8, knn1_15vs8_pred)

    _, knn_5mean_euclid_5vs18, knn_5std_euclid_5vs18 = get_euclid_per_coord(ytest_5vs18, knn5_5vs18_pred)

    _, knn_5mean_euclid_10vs13, knn_5std_euclid_10vs13 = get_euclid_per_coord(ytest_10vs13, knn5_10vs13_pred)

    _, knn_5mean_euclid_15vs8, knn_5std_euclid_15vs8 = get_euclid_per_coord(ytest_15vs8, knn5_15vs8_pred)

    _, rf_mean_euclid_5vs18, rf_std_euclid_5vs18 = get_euclid_per_coord(ytest_5vs18, rf_5vs18_pred)

    _, rf_mean_euclid_10vs13, rf_std_euclid_10vs13 = get_euclid_per_coord(ytest_10vs13, rf_10vs13_pred)

    _, rf_mean_euclid_15vs8, rf_std_euclid_15vs8 = get_euclid_per_coord(ytest_15vs8, rf_15vs8_pred)

    tabla_metricas_per_coord = pd.DataFrame(columns=["Model", "Partition", "Label", "Mean Euclid"])
    for idx, pos in enumerate(np.unique(ytest_5vs18, axis=0)):
        label = dict_inv[(pos[0], pos[1])]
        idxs_coord = np.where(ytest_5vs18 == pos)[0]
        euclidean_distances = np.sqrt(np.sum((knn1_5vs18_pred[idxs_coord] - pos) ** 2, axis=1))
        mean_euclid = np.mean(euclidean_distances)
        std_euclid = np.std(euclidean_distances)
        tabla_metricas_per_coord = tabla_metricas_per_coord.append(
            {"Model": "KNN(k=1)", "Partition": "5vs18", "Label": label, "Mean Euclid": mean_euclid,
             "Std Euclid": std_euclid},
            ignore_index=True)

    for idx, pos in enumerate(np.unique(ytest_10vs13, axis=0)):
        label = dict_inv[(pos[0], pos[1])]
        idxs_coord = np.where(ytest_10vs13 == pos)[0]
        euclidean_distances = np.sqrt(np.sum((knn1_10vs13_pred[idxs_coord] - pos) ** 2, axis=1))
        mean_euclid = np.mean(euclidean_distances)
        std_euclid = np.std(euclidean_distances)
        tabla_metricas_per_coord = tabla_metricas_per_coord.append(
            {"Model": "KNN(k=1)", "Partition": "10vs13", "Label": label, "Mean Euclid": mean_euclid,
             "Std Euclid": std_euclid},
            ignore_index=True)

    for idx, pos in enumerate(np.unique(ytest_15vs8, axis=0)):
        label = dict_inv[(pos[0], pos[1])]
        idxs_coord = np.where(ytest_15vs8 == pos)[0]
        euclidean_distances = np.sqrt(np.sum((knn1_15vs8_pred[idxs_coord] - pos) ** 2, axis=1))
        mean_euclid = np.mean(euclidean_distances)
        std_euclid = np.std(euclidean_distances)
        tabla_metricas_per_coord = tabla_metricas_per_coord.append(
            {"Model": "KNN(k=1)", "Partition": "15vs8", "Label": label, "Mean Euclid": mean_euclid,
             "Std Euclid": std_euclid},
            ignore_index=True)

    for idx, pos in enumerate(np.unique(ytest_5vs18, axis=0)):
        label = dict_inv[(pos[0], pos[1])]
        idxs_coord = np.where(ytest_5vs18 == pos)[0]
        euclidean_distances = np.sqrt(np.sum((knn5_5vs18_pred[idxs_coord] - pos) ** 2, axis=1))
        mean_euclid = np.mean(euclidean_distances)
        std_euclid = np.std(euclidean_distances)
        tabla_metricas_per_coord = tabla_metricas_per_coord.append(
            {"Model": "KNN(k=5)", "Partition": "5vs18", "Label": label, "Mean Euclid": mean_euclid,
             "Std Euclid": std_euclid},
            ignore_index=True)

    for idx, pos in enumerate(np.unique(ytest_10vs13, axis=0)):
        label = dict_inv[(pos[0], pos[1])]
        idxs_coord = np.where(ytest_10vs13 == pos)[0]
        euclidean_distances = np.sqrt(np.sum((knn5_10vs13_pred[idxs_coord] - pos) ** 2, axis=1))
        mean_euclid = np.mean(euclidean_distances)
        std_euclid = np.std(euclidean_distances)
        tabla_metricas_per_coord = tabla_metricas_per_coord.append(
            {"Model": "KNN(k=5)", "Partition": "10vs13", "Label": label, "Mean Euclid": mean_euclid,
             "Std Euclid": std_euclid},
            ignore_index=True)

    for idx, pos in enumerate(np.unique(ytest_15vs8, axis=0)):
        label = dict_inv[(pos[0], pos[1])]
        idxs_coord = np.where(ytest_15vs8 == pos)[0]
        euclidean_distances = np.sqrt(np.sum((knn5_15vs8_pred[idxs_coord] - pos) ** 2, axis=1))
        mean_euclid = np.mean(euclidean_distances)
        std_euclid = np.std(euclidean_distances)
        tabla_metricas_per_coord = tabla_metricas_per_coord.append(
            {"Model": "KNN(k=5)", "Partition": "15vs8", "Label": label, "Mean Euclid": mean_euclid,
             "Std Euclid": std_euclid},
            ignore_index=True)

    for idx, pos in enumerate(np.unique(ytest_5vs18, axis=0)):
        label = dict_inv[(pos[0], pos[1])]
        idxs_coord = np.where(ytest_5vs18 == pos)[0]
        euclidean_distances = np.sqrt(np.sum((rf_5vs18_pred[idxs_coord] - pos) ** 2, axis=1))
        mean_euclid = np.mean(euclidean_distances)
        std_euclid = np.std(euclidean_distances)

        tabla_metricas_per_coord = tabla_metricas_per_coord.append(
            {"Model": "RF", "Partition": "5vs18", "Label": label, "Mean Euclid": mean_euclid,
             "Std Euclid": std_euclid},
            ignore_index=True)

    for idx, pos in enumerate(np.unique(ytest_10vs13, axis=0)):
        label = dict_inv[(pos[0], pos[1])]
        idxs_coord = np.where(ytest_10vs13 == pos)[0]
        euclidean_distances = np.sqrt(np.sum((rf_10vs13_pred[idxs_coord] - pos) ** 2, axis=1))
        mean_euclid = np.mean(euclidean_distances)
        std_euclid = np.std(euclidean_distances)
        tabla_metricas_per_coord = tabla_metricas_per_coord.append(
            {"Model": "RF", "Partition": "10vs13", "Label": label, "Mean Euclid": mean_euclid,
             "Std Euclid": std_euclid},
            ignore_index=True)

    for idx, pos in enumerate(np.unique(ytest_15vs8, axis=0)):
        label = dict_inv[(pos[0], pos[1])]
        idxs_coord = np.where(ytest_15vs8 == pos)[0]
        euclidean_distances = np.sqrt(np.sum((rf_15vs8_pred[idxs_coord] - pos) ** 2, axis=1))
        mean_euclid = np.mean(euclidean_distances)
        std_euclid = np.std(euclidean_distances)
        tabla_metricas_per_coord = tabla_metricas_per_coord.append(
            {"Model": "RF", "Partition": "15vs8", "Label": label, "Mean Euclid": mean_euclid,
             "Std Euclid": std_euclid},
            ignore_index=True)

    print(tabla_metricas_per_coord.sort_values(by=["Model", "Partition", "Label"]).reset_index(drop=True))

    os.makedirs(f"{path_partitions_output}/tablas", exist_ok=True)
    tabla_metricas_per_coord.to_csv(f"{path_partitions_output}/tablas/tabla_metricas_per_coord.csv", index=False)

    aux = tabla_metricas_per_coord.copy()
    colores = ["orange", "green", "purple"]
    plt.figure(figsize=(10, 20))
    for idx, partition in enumerate(aux.Partition.unique()):
        aux_partition = aux[aux.Partition == partition]
        plt.subplot(3, 1, idx + 1)
        plt.title(partition)
        for idx_model, model in enumerate(aux_partition.Model.unique()):
            aux_model = aux_partition[aux_partition.Model == model].sort_values(by=["Mean Euclid"])
            xaxis = [x for x in range(1, aux_model.shape[0] + 1)]
            plt.plot(xaxis, aux_model["Mean Euclid"], label=model, c=colores[idx_model] )
            plt.errorbar(xaxis, aux_model["Mean Euclid"], yerr=aux_model["Std Euclid"],
                         fmt='o', c=colores[idx_model], capsize=5)
            plt.xticks(xaxis,xaxis)
        plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
