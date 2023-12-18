import pandas as pd
import numpy as np

import os
import glob

import tqdm

from src.utils.constants import constants


def parse_windows(n_max: int, window_size: int, step: int):
    """
    Esta función devuelve una lista de ventanas de tamaño window_size y step step.

    Parameters:
    -----------
    n_max: int
        número máximo a generar en la lista de ventanas

    window_size: int
        tamaño de la ventana

    step: int
        tamaño del paso

    Returns:
    --------
    list
        Lista de ventanas (tuplas) con índices (start, end) de cada ventana generada en base a la longitud de la serie temporal, el tamaño de la ventana y el tamaño del paso

    Examples:
    ---------
    >>> parse_windows(n_max=10, window_size=5, step=2)
    >>> [(0, 5), (2, 7), (4, 9), (6, 10)]

    >>> parse_windows(n_max=30, window_size=10, step=5)
    >>> [(0, 10), (5, 15), (10, 20), (15, 25), (20, 30)]

    """
    return [(i, min(i + window_size, n_max)) for i in range(0, n_max, step) if i + window_size <= n_max]


def get_checkpoints_data(dir_data: str, dict_labels) -> pd.DataFrame:
    out_dir = "/".join(dir_data.split("/")[:-1])
    CHECKPOINT_DATA_PATH = f"{out_dir}/checkpoint_groundtruth"
    WIFI_CHECKPOINT = f"{CHECKPOINT_DATA_PATH}/Wifi"
    os.makedirs(CHECKPOINT_DATA_PATH, exist_ok=True)
    os.makedirs(WIFI_CHECKPOINT, exist_ok=True)
    for label, position in tqdm.tqdm(dict_labels.items()):
        # Si no existe el checkpoint con el label, lo creamos
        if str(label) not in ", ".join(os.listdir(WIFI_CHECKPOINT)):
            print(f"Creando checkpoint de label: {label}...")
            position = dict_labels[label]
            # Inicialización WiFi
            wifi_data = pd.DataFrame(
                columns=['AppTimestamp(s)', 'Name_SSID', 'MAC_BSSID', 'RSS'])

            # Lectura de los datos de cada label
            with open(f'{dir_data}/label_{label}.txt', 'r') as fich:
                for linea in fich.readlines():  # Leemos línea
                    texto = linea.rstrip("\n").split(";")  # Quitamos saltos de línea y separamos por ";"

                    if texto[0] == "WIFI":  # Si el primer elemento es ACCE, añadimos datos al acelerómetro
                        wifi_data = pd.concat([wifi_data,
                                               pd.DataFrame({
                                                   "AppTimestamp(s)": [float(texto[1])],
                                                   "Name_SSID": [texto[3]],
                                                   "MAC_BSSID": [texto[4]],
                                                   "RSS": [float(texto[5])],
                                                   "Label": [label],
                                                   "Latitude": [position[1]],
                                                   "Longitude": [position[0]]
                                               })], ignore_index=True)

            # Guardamos los datos en el checkpoint
            wifi_data.to_csv(f"{WIFI_CHECKPOINT}/Wifi_label_{label}.csv", index=False)


def correctWifiFP(wifi_data: pd.DataFrame, t_max_sampling: int, dict_labels_to_meters: dict) -> pd.DataFrame:
    """
    Coge el fingerprint de los datos y aplica el siguiente preprocesado:
            1) Ajuste de frecuencia muestral a 1 muestra / segundo
                    - Cálculo de la media agrupada por segundo y por label en cada una de las balizas
            2) Ajuste de datos ausentes:
                    - Los segundos sin recogida de datos son reemplazados por el mínimo RSS del dataset menos 1
            3) Se escalan los valores entre 0 y 1 (siendo 0 la representación del dato ausente)
                    X = (X - X_min + 1)/X_max

    Parameters:
    -----------
    wifi_data: pd.DataFrame
        pd.DataFrame de los datos correspondientes al WiFi

    t_max_sampling: int
        Tiempo máximo de muestreo por label

    dict_labels_to_meters: dict
        Diccionario que transforma el label a (longitud, latitud) en metros

    Returns:
    --------
    wifi_corrected: pd.DataFrame
        pd.DataFrame de los datos corregidos

    Example:
    --------
    >>> t_max_sampling = constants.T_MAX_SAMPLING # 1140 segundos de recogida de datos por cada label en train (60s en test)
    >>> dict_labels_to_meters = constants.labels_dictionary_meters
    >>> wifi_corrected = correctWifiFP(wifi_data, t_max_sampling=t_max_sampling, dict_labels_to_meters=dict_labels_to_meters)
    """

    list_of_labels = wifi_data.Label.unique().tolist()
    # Cogemos los datos de las balizas que han aparecido en más de una recogida de datos
    aux = wifi_data.query(f'Name_SSID in {constants.aps}')
    # Creamos un dataframe con todos los intervalos de tiempo y todas las balizas
    labels, intervalos_tiempo, ssids = [], [], []
    for lab in list_of_labels:
        for ts in range(0, t_max_sampling + 1, 1):
            for ssid in constants.aps:
                labels.append(lab)
                intervalos_tiempo.append(ts)
                ssids.append(ssid)
    df_intervalos = pd.DataFrame(
        {'Label': labels, 'AppTimestamp(s)': intervalos_tiempo, 'Name_SSID': ssids})  # Dataframe de intervalos

    aux["AppTimestamp(s)"] = aux["AppTimestamp(s)"].round()  # Redondeamos el timestamp a 0 decimales
    aux = aux.groupby(["Label", "AppTimestamp(s)", "Name_SSID"]).mean(numeric_only=True)[
        "RSS"].reset_index()  # Agrupamos por label, timestamp y ssid
    aux_corrected = pd.merge(df_intervalos, aux, on=["Label", "AppTimestamp(s)", "Name_SSID"],
                             how="left")  # Unimos con el dataframe de intervalos
    # Reemplazamos los valores ausentes por el mínimo global
    aux_corrected = aux_corrected.pivot(index=['Label', 'AppTimestamp(s)'], columns='Name_SSID')[
        "RSS"].reset_index()  # Pivotamos el dataframe

    aux_corrected[["Longitude", "Latitude"]] = [dict_labels_to_meters[x] for x in
                                                aux_corrected.Label]  # Añadimos la longitud y latitud

    orden_wifi_columnas = ["AppTimestamp(s)"] + constants.aps + ["Latitude", "Longitude", "Label"]
    wifi_corrected = aux_corrected[orden_wifi_columnas]  # Ordenamos las columnas
    return wifi_corrected


def fix_na_wifi(data: pd.DataFrame) -> pd.DataFrame:
    aux = data.copy()
    min_global = aux[constants.aps].min().min() - 1
    aux[constants.aps] = aux[constants.aps].fillna(min_global)
    return aux


def scale_wifi(data: pd.DataFrame) -> pd.DataFrame:
    aux = data.copy()
    aux[constants.aps] -= aux[constants.aps].min().min() - 1
    aux[constants.aps] /= aux[constants.aps].max().max()
    return aux


def rolling_mean(data: pd.DataFrame, window_size: int, step: int) -> pd.DataFrame:
    """
    Aplica una media móvil a los datos de WiFi, de forma que si hay un hueco en el timestamp, se rellena con el valor anterior si y solo si el valor anterior y posterior son iguales
    Parameters
    ----------
    data: pd.DataFrame
        pd.DataFrame de los datos correspondientes al WiFi
    window_size: int
        Tamaño de la ventana
    step: int
        Tamaño del paso de la ventana

    Returns
    -------
    rolled_data: pd.DataFrame
        pd.DataFrame de los datos con la media aplicada

    """
    n_max = data["AppTimestamp(s)"].max()
    combinaciones = parse_windows(n_max=n_max, window_size=window_size,
                                  step=step)  # Combinaciones de ventanas
    data_columns = ["AppTimestamp(s)"] + constants.aps + ["Latitude", "Longitude", "Label"]  # Columnas de los datos
    rolled_data = pd.DataFrame(columns=data_columns)  # Dataframe vacío
    for lon, lat, lab in data[["Longitude", "Latitude", "Label"]].drop_duplicates().values:  # Recorremos cada coordenada única
        query = data[(data["Longitude"] == lon) & (data["Latitude"] == lat)]  # Filtramos por coordenada
        for start, end in combinaciones:  # Recorremos cada combinación de ventanas
            aux = query[(query["AppTimestamp(s)"] >= start) & (query["AppTimestamp(s)"] < end)]  # Filtramos por ventana
            mean_by_wifi_column = aux[constants.aps].mean(axis=0).tolist()  # Media por cada AP
            row = [end] + mean_by_wifi_column + [lat, lon]  # Fila a añadir
            row.append(lab)  # Añadimos el label
            rolled_data.loc[len(rolled_data)] = row  # Añadimos la fila

    return rolled_data


def interpolacion_pixel_proximo(data: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Aplica una interpolación a los datos de WiFi, de forma que si hay un hueco en el timestamp, se rellena con el valor anterior si y solo si el valor anterior y posterior son iguales

    Parameters:
    -----------
    data: pd.DataFrame
        pd.DataFrame de los datos correspondientes al WiFi

    threshold: int
        Número de segundos sin recogida de datos consecutivos para considerar si rechazar la interpolación

    Returns:
    --------
    interpolated_data: pd.DataFrame
        pd.DataFrame de los datos interpolados

    Example:
    --------

    Lectura del conjunto de datos

    >>> data = pd.read_csv(f"{constants.data.train.FINAL_PATH}/groundtruth.csv")

    Aplicación de la interpolación

    >>> interpolated_data = interpolacion_pixel_proximo(data, threshold=30)
    """
    n_timestamp = data["AppTimestamp(s)"].max()
    coords_unique = data.drop_duplicates(["Longitude", "Latitude"])[["Longitude", "Latitude"]].reset_index()
    interpolated_data = pd.DataFrame(columns=data.columns)

    for _, row in tqdm.tqdm(coords_unique.iterrows()):
        x, y = row["Longitude"], row["Latitude"]
        query = data[(data["Longitude"] == x) & (data["Latitude"] == y)].reset_index()

        for ap in constants.aps:
            for t in range(n_timestamp + 1):
                if query[ap].sum() == 0:
                    break

                if query[ap].iloc[t] == 0:
                    if t == 0:
                        right_values = query.loc[(t + 1):(t + threshold), ap].values
                        idx_min_right = np.argmin(right_values)
                        query.at[t, ap] = right_values[idx_min_right]
                    elif t == n_timestamp:
                        left_values = query.loc[(t - threshold):(t - 1), ap].values
                        idx_max_left = np.argmax(left_values)
                        query.at[t, ap] = left_values[idx_max_left]
                    else:
                        left_values = query.loc[(t - threshold):(t - 1), ap].values
                        right_values = query.loc[(t + 1):(t + threshold), ap].values

                        idx_max_left = np.argmax(left_values)
                        idx_min_right = np.argmin(right_values)

                        if left_values[idx_max_left] == right_values[idx_min_right]:
                            query.at[t, ap] = left_values[idx_max_left]

        # interpolated_data = interpolated_data.append(query) bug compatibilidad pandas
        interpolated_data = pd.concat([interpolated_data, query])

    return interpolated_data


def read_checkpoint(checkpoint_path: str, labels: list) -> pd.DataFrame:
    """
    Lee los datos de un checkpoint y los concatena en un único dataframe

    Parameters:
    -----------
    checkpoint_path: str
        Path del checkpoint

    Returns:
    --------
    df: pd.DataFrame
        pd.DataFrame con los datos concatenados

    Example:
    --------
    Ruta al checkpoint del acelerómetro:

    >>> CHECKPOINT_ACCELEROMETER_PATH = "data/train/checkpoint_groundtruth/Accelerometer"

    Ruta al checkpoint del giroscopio:

    >>> CHECKPOINT_GYROSCOPE_PATH = "data/train/checkpoint_groundtruth/Gyroscope"

    Ruta al checkpoint del magnetómetro:

    >>> CHECKPOINT_MAGNETOMETER_PATH = "data/train/checkpoint_groundtruth/Magnetometer"

    Ruta al checkpoint del WiFi:

    >>> CHECKPOINT_WIFI_PATH = "data/train/checkpoint_groundtruth/Wifi"

    Lectura de los datos del acelerómetro:

    >>> accelerometer = read_checkpoint(CHECKPOINT_DATA_PATH)

    Lectura de los datos del giroscopio:

    >>> gyroscope = read_checkpoint(CHECKPOINT_GYROSCOPE_PATH)

    Lectura de los datos del magnetómetro:

    >>> magnetometer = read_checkpoint(CHECKPOINT_MAGNETOMETER_PATH)

    Lectura de los datos del WiFi:

    >>> wifi = read_checkpoint(CHECKPOINT_WIFI_PATH)
    """

    df = pd.DataFrame()
    for path in glob.glob(os.path.join(checkpoint_path, "*.csv")):
        label = path.split("/")[-1].split("_")[-1].split(".")[0]
        if int(label) in labels:
            df = pd.concat((df, pd.read_csv(path)))
    return df
