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

    tabla_metricas = tabla_metricas_per_coord.groupby(["Model", "Partition"]) \
        .mean()[["Mean Euclid", "Std Euclid"]] \
        .reset_index()
    tabla_metricas.to_csv(f"{path_partitions_output}/tablas/tabla_metricas.csv", index=False)

    os.makedirs(f"{path_partitions_output}/tablas", exist_ok=True)
    os.makedirs(f"{path_partitions_output}/plots", exist_ok=True)
    tabla_metricas_per_coord.to_csv(f"{path_partitions_output}/tablas/tabla_metricas_per_coord.csv", index=False)

    aux = tabla_metricas_per_coord.copy()
    y_max = np.ceil(np.max(aux["Mean Euclid"] + aux["Std Euclid"]))+0.5
    y_min = np.floor(np.min(aux["Mean Euclid"] - aux["Std Euclid"]))
    colores = ["orange", "green", "purple"]
    plt.figure(figsize=(10, 15))
    for idx, partition in enumerate(aux.Partition.unique()):
        aux_partition = aux[aux.Partition == partition]
        plt.subplot(3, 1, idx + 1)
        plt.title(f"Partición: {partition} ")
        for idx_model, model in enumerate(aux_partition.Model.unique()):
            aux_model = aux_partition[aux_partition.Model == model].sort_values(by=["Mean Euclid"])
            xaxis = [x for x in range(1, aux_model.shape[0] + 1)]
            plt.plot(xaxis, aux_model["Mean Euclid"], label=model, c=colores[idx_model])
            plt.errorbar(xaxis, aux_model["Mean Euclid"], yerr=aux_model["Std Euclid"],
                         fmt='o', c=colores[idx_model], capsize=5)
            plt.xticks(xaxis, xaxis)
        plt.ylim(y_min, y_max)
        plt.ylabel("Error (m)")
        plt.xlabel("Ordered Points (by error)")
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_partitions_output}/plots/errorbar metrics.png")
    plt.show()

    aux = tabla_metricas.sort_values(by=["Mean Euclid"])
    aux["Model_Partition"] = aux["Model"] + "-" + aux["Partition"]
    f = lambda x: "orange" if "KNN(k=1)" in x else "green" if "KNN(k=5)" in x else "purple"
    # obtener un array de colores usando map
    colores = list(map(f, aux["Model_Partition"]))
    labels = list(map(lambda x: x.split("-")[0], aux["Model_Partition"]))
    plt.figure(figsize=(10, 5))
    plt.title("Barplot error per each model-partition")
    plt.bar(aux["Model_Partition"], aux["Mean Euclid"], color=colores)
    # color yerr
    plt.errorbar(aux["Model_Partition"], aux["Mean Euclid"], yerr=aux["Std Euclid"], capthick=3, capsize=6, color="black")
    # customized legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ["orange", "green", "purple"]]
    plt.legend(handles, ["KNN(k=1)", "KNN(k=5)", "RF"])
    plt.xticks(rotation=15)
    plt.savefig(f"{path_partitions_output}/plots/barplot metrics.png")
    plt.show()


if __name__ == "__main__":
    main()
