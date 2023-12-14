import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import tqdm

import sys

sys.path.append("../")
from src.constants import constants


def get_rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def get_mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))


def get_mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def get_mean_euclidean_distance(y_pred, y_true):
    return np.mean(np.sqrt(np.sum((y_pred - y_true) ** 2, axis=1)))


def get_metrics(y_pred, y_true):
    return get_rmse(y_pred, y_true), get_mae(y_pred, y_true), \
        get_mse(y_pred, y_true), get_mean_euclidean_distance(y_pred, y_true)


def get_euclid_per_coord(ytrue, ypred):
    euclidean_distances = np.sqrt(np.sum((ypred - ytrue) ** 2, axis=1))
    coords_unique = np.unique(ytrue, axis=0)
    dist_max, dist_mean, dist_std = [], [], []
    for coord in coords_unique:
        indices = np.where(np.all(ytrue == coord, axis=1))[0]
        if indices.size > 0:
            start, end = indices[0], indices[-1] + 1
            dist_max.append(np.max(euclidean_distances[start:end]))
            dist_mean.append(np.mean(euclidean_distances[start:end]))
            dist_std.append(np.std(euclidean_distances[start:end]))
    return np.array(dist_max), np.array(dist_mean), np.array(dist_std)


def results_knn_training(Xtrain: np.ndarray, Xtest: np.ndarray, ytrain: np.ndarray, ytest: np.ndarray,
                         k_neighbors: list, sort_by_rmse: bool = True) -> pd.DataFrame:
    """
    Obtiene los resultados de entrenar un modelo KNN con los datos de entrada y salida de entrenamiento y testeo para los valores de k_neighbors

    Parameters:
    -----------
    Xtrain: np.ndarray
        Datos de entrada de entrenamiento
    Xtest: np.ndarray
        Datos de entrada de testeo
    ytrain: np.ndarray
        Datos de salida de entrenamiento
    ytest: np.ndarray
        Datos de salida de testeo
    k_neighbors: list
        Lista de valores de k_neighbors para entrenar el modelo
    sort_by_rmse: bool = True
        Ordena los resultados por el valor de rmse en caso de que sea verdadero

    Returns:
    --------
    results: pd.DataFrame
        Dataframe con los resultados de entrenar el modelo KNN con los datos de entrada y salida de entrenamiento y testeo para los valores de k_neighbors

    Examples:
    ---------
    Lectura de datos:

    >>> from src.constants import constants
    >>> import pandas as pd
    >>> import sklearn as sk
    >>> groundtruth = pd.read_csv(f"{constants.data.FINAL_PATH}/groundtruth.csv")

    Obtención de train y test:

    >>> X, y = grountruth[constants.aps].to_numpy(), groundtruth[["Longitude", "Latitude"]].to_numpy()
    >>> Xtrain, Xtest, ytrain, ytest = sk.model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
    >>> results_knn_training(Xtrain, Xtest, ytrain, ytest, k_neighbors=[1, 3, 5, 7, 9])
    """
    results = pd.DataFrame(columns=["rmse", "mse", "mae", "euclid", "k_neighbors"])
    for k in tqdm.tqdm(k_neighbors):
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(Xtrain, ytrain)
        ypred = knn.predict(Xtest)
        rmse, mse, mae, euclid = get_metrics(ytest, ypred)
        results = results.append({"rmse": rmse, "mse": mse, "mae": mae, "euclid": euclid, "k_neighbors": k},
                                 ignore_index=True)

    results.k_neighbors = results.k_neighbors.astype(int)
    if sort_by_rmse:
        results.sort_values(by="rmse", inplace=True)

    return results


def knn_line_plot_metrics(results: pd.DataFrame, path_out: str = "metrics_knn", save_ok: bool = True,
                          title: str = "Métricas kNN para cada k", y_label: str = "Error") -> None:
    """
    Realiza un gráfico de línea con los resultados de entrenar un modelo KNN con los datos de entrada y salida de entrenamiento y testeo para los valores de k_neighbors

    Parameters:
    -----------
    results: pd.DataFrame
        Dataframe con los resultados de entrenar el modelo KNN con los datos de entrada y salida de entrenamiento y testeo para los valores de k_neighbors
    path_out: str="metrics_knn"
        Ruta del directorio del gráfico
    save_ok: bool=True
        Guarda el gráfico en caso de que sea verdadero
    title: str = "Métricas kNN para cada k"
        Título del gráfico
    y_label: str = "Error"
        Etiqueta del eje y

    Returns:
    --------
    None

    Examples:
    ---------
    Obtención de los resultados con las métricas

    >>> results = results_knn_training(Xtrain, Xtest, ytrain, ytest, k_neighbors=[1, 3, 5, 7, 9])

    Gráfico de línea:

    >>> line_plot_metrics(results)
    """

    size_K = results.shape[0]

    plt.scatter([x for x in range(1, size_K + 1)], results["rmse"])
    plt.scatter([x for x in range(1, size_K + 1)], results["mse"])
    plt.scatter([x for x in range(1, size_K + 1)], results["mae"])
    plt.scatter([x for x in range(1, size_K + 1)], results["euclid"])
    plt.plot([x for x in range(1, size_K + 1)], results["rmse"], linestyle="--", label="RMSE")
    plt.plot([x for x in range(1, size_K + 1)], results["mse"], linestyle="--", label="MSE")
    plt.plot([x for x in range(1, size_K + 1)], results["mae"], linestyle="--", label="MAE")
    plt.plot([x for x in range(1, size_K + 1)], results["euclid"], linestyle="--", label="Euclid")

    # anotar el k de cada punto
    for idx, k in enumerate(results["k_neighbors"]):
        plt.annotate(f"K={k}", (idx + 0.9, results["rmse"].tolist()[idx] + 0.1))
        plt.annotate(f"K={k}", (idx + 0.9, results["mse"].tolist()[idx] + 0.1))
        plt.annotate(f"K={k}", (idx + 0.9, results["mae"].tolist()[idx] + 0.1))
        plt.annotate(f"K={k}", (idx + 0.9, results["euclid"].tolist()[idx] + 0.1))
    plt.ylabel(y_label)
    plt.xticks([])
    plt.legend()
    plt.title(title)
    if save_ok:
        plt.savefig(f"{path_out}/knn_line_plot_metrics.png")
    plt.show()


# def results_rf_training(Xtrain: np.ndarray, Xtest: np.ndarray, ytrain: np.ndarray, ytest: np.ndarray,
#                         n_trees: list, sort_by_rmse: bool = True) -> pd.DataFrame:
#     """
#     Obtiene los resultados de entrenar un modelo Random Forest con los datos de entrada y salida de entrenamiento y testeo para los valores de n_trees
#
#     Parameters:
#     -----------
#     Xtrain: np.ndarray
#         Datos de entrada de entrenamiento
#     Xtest: np.ndarray
#         Datos de entrada de testeo
#     ytrain: np.ndarray
#         Datos de salida de entrenamiento
#     ytest: np.ndarray
#         Datos de salida de testeo
#     n_trees: list
#         Lista de valores de n_trees para entrenar el modelo
#     sort_by_rmse: bool = True
#         Ordena los resultados por el valor de rmse en caso de que sea verdadero
#
#     Returns:
#     --------
#     results: pd.DataFrame
#         Dataframe con los resultados de entrenar el modelo Random Forest con los datos de entrada y salida de entrenamiento y testeo para los valores de n_trees
#
#     Examples:
#     ---------
#     Lectura de datos:
#
#     >>> from src.constants import constants
#     >>> import pandas as pd
#     >>> import sklearn as sk
#
#     >>> groundtruth = pd.read_csv(f"{constants.data.FINAL_PATH}/groundtruth.csv")
#
#     Obtención de train y test:
#
#     >>> X, y = grountruth[constants.aps].to_numpy(), groundtruth[["Longitude", "Latitude"]].to_numpy()
#     >>> Xtrain, Xtest, ytrain, ytest = sk.model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
#     >>> metricas = results_rf_training(Xtrain, Xtest, ytrain, ytest, n_trees=[1, 3, 5, 7, 9])
#     """
#
#     results = pd.DataFrame(columns=["rmse", "mse", "mae", "euclid", "n_trees"])
#     for n in tqdm.tqdm(n_trees):
#         rf = RandomForestRegressor(n_estimators=n)
#         rf.fit(Xtrain, ytrain)
#         ypred = rf.predict(Xtest)
#         metrics = get_metrics(ytest, ypred)
#         results = results.append({**metrics, "n_trees": n}, ignore_index=True)
#
#     results.n_trees = results.n_trees.astype(int)
#     if sort_by_rmse:
#         results.sort_values(by="rmse", inplace=True)
#
#     return results

def results_model_training(Xtrain: np.ndarray, Xtest: np.ndarray, ytrain: np.ndarray, ytest: np.ndarray,
                           list_params: list, model_type: str = "random_forest", sort_by_rmse=True):

    option = {"random_forest": (RandomForestRegressor, "n_estimators"),
              "knn": (KNeighborsRegressor, "n_neighbors")}
    result_column = option[model_type][1]
    print(f"training... {model_type} with {result_column}: {list_params}")
    results = pd.DataFrame(columns=["rmse", "mse", "mae", "euclid", result_column])
    for param in tqdm.tqdm(list_params):
        print(f"training {model_type}({result_column} = {param})...")
        model = option[model_type][0](param)
        model.fit(Xtrain, ytrain)
        metrics = get_metrics(ytest, model.predict(Xtest))
        results = results.append({**metrics, result_column: param}, ignore_index=True)

    results[result_column] = results[result_column].astype(int)
    if sort_by_rmse:
        results.sort_values(by="rmse", inplace=True)

    return results


def rf_line_plot_metrics(results: pd.DataFrame, path_out: str = "metrics_rf", save_ok: bool = True,
                         title: str = "Métricas Random Forest para cada n", y_label: str = "Error") -> None:
    """
    Realiza un gráfico de línea con los resultados de entrenar un modelo Random Forest con los datos de entrada y salida de entrenamiento y testeo para los valores de n_trees

    Parameters:
    -----------
    results: pd.DataFrame
        Dataframe con los resultados de entrenar el modelo Random Forest con los datos de entrada y salida de entrenamiento y testeo para los valores de n_trees
    path_out: str="metrics_rf"
        Ruta del directorio del gráfico
    save_ok: bool=True
        Guarda el gráfico en caso de que sea verdadero
    title: str = "Métricas Random Forest para cada n"
        Título del gráfico
    y_label: str = "Error"
        Etiqueta del eje y

    Returns:
    --------
    None

    Examples:
    ---------
    Obtención de los resultados con las métricas

    >>> results = results_rf_training(Xtrain, Xtest, ytrain, ytest, n_trees=[1, 3, 5, 7, 9])

    Gráfico de línea:

    >>> line_plot_metrics(results)
    """

    size_K = results.shape[0]
    x_values = np.arange(1, size_K + 1)

    metrics = ["rmse", "mse", "mae", "euclid"]

    for metric in metrics:
        plt.scatter(x_values, results[metric])
        plt.plot(x_values, results[metric], linestyle="--", label=metric.capitalize())

    # anotar el k de cada punto
    for idx, n in enumerate(results["n_trees"]):
        for metric in metrics:
            plt.annotate(f"N={n}", (idx + 0.9, results[metric].iloc[idx] + 0.1))

    plt.ylabel(y_label)
    plt.xticks([])
    plt.legend()
    plt.title(title)

    if save_ok:
        plt.savefig(f"{path_out}/rf_line_plot_metrics.png")

    plt.show()


def metrics_dist_euclid_per_coord(ytrue, ypred, return_all, option_return: str = "max"):
    euclidean_distances = np.sqrt(np.sum((ypred - ytrue) ** 2, axis=1))

    coords_unique = np.array(list(constants.labels_dictionary_meters_test.values()))

    dist_max, dist_mean, dist_std = [], [], []

    for coord in coords_unique:
        indices = np.where(np.all(ytrue == coord, axis=1))[0]
        if indices.size > 0:
            start, end = indices[0], indices[-1] + 1
            dist_max.append(np.max(euclidean_distances[start:end]))
            dist_mean.append(np.mean(euclidean_distances[start:end]))
            dist_std.append(np.std(euclidean_distances[start:end]))

    options = {"max": np.array(dist_max), "mean": np.array(dist_mean), "std": np.array(dist_std)}

    return (dist_max, dist_mean, dist_std) if return_all else options[option_return]
