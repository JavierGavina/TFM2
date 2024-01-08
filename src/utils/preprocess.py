import pandas as pd
import numpy as np

import os
import glob

import tqdm

from src.utils.constants import constants


def parse_windows(n_max: int, window_size: int, step: int):
    """
    This function returns a list of windows of size 'window_size' with a step size of 'step'.

    Parameters:
    -----------
    n_max: int
        Maximum number to generate in the window list.

    window_size: int
        Size of the window.

    step: int
        Step size.

    Returns:
    --------
    list
        List of windows (tuples) with indices (start, end) for each window generated based on the length of the time series, window size, and step size.

    Examples:
    ---------
    >>> parse_windows(n_max=10, window_size=5, step=2)
    >>> [(0, 5), (2, 7), (4, 9)]

    >>> parse_windows(n_max=30, window_size=10, step=5)
    >>> [(0, 10), (5, 15), (10, 20), (15, 25), (20, 30)]
    """
    return [(i, min(i + window_size, n_max)) for i in range(0, n_max, step) if i + window_size <= n_max]


def get_checkpoints_data(dir_data: str, out_dir: str, dict_labels) -> None:
    """
    Extracts and organizes checkpoint data from labeled sources and saves it in a structured format.

    Parameters:
    -----------
    dir_data: str
        The directory path where the labeled data is stored.

    dict_labels: dict
        A dictionary containing labels and corresponding positions.

    Returns:
    --------
    None
        Saves a pd.DataFrame containing organized checkpoint data with columns: 'AppTimestamp(s)', 'Name_SSID',
        'MAC_BSSID', 'RSS', 'Label', 'Latitude', and 'Longitude'.

    Example:
    --------
    >>> dir_data = '/path/to/data'
    >>> dict_labels = {'label1': (lat1, lon1), 'label2': (lat2, lon2), ...}
    >>> get_checkpoints_data(dir_data, dict_labels)
    """
    CHECKPOINT_DATA_PATH = f"{out_dir}/checkpoint_groundtruth"
    WIFI_CHECKPOINT = f"{CHECKPOINT_DATA_PATH}/Wifi"
    os.makedirs(CHECKPOINT_DATA_PATH, exist_ok=True)
    os.makedirs(WIFI_CHECKPOINT, exist_ok=True)
    for label, position in tqdm.tqdm(dict_labels.items()):
        # If the checkpoint with the label doesn't exist, we create it
        if str(label) not in ", ".join(os.listdir(WIFI_CHECKPOINT)):
            print(f"Creando checkpoint de label: {label}...")
            position = dict_labels[label]
            # Initialize Wi-Fi
            wifi_data = pd.DataFrame(
                columns=['AppTimestamp(s)', 'Name_SSID', 'MAC_BSSID', 'RSS'])

            # Read the data for each label
            with open(f'{dir_data}/label_{label}.txt', 'r') as fich:
                for linea in fich.readlines():  # Read the line
                    texto = linea.rstrip("\n").split(";")  # Drop break lines and split by ";"

                    if texto[0] == "WIFI":  # If the first element is Wi-Fi, add the Wi-Fi data
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

            # Save the data on the checkpoint
            wifi_data.to_csv(f"{WIFI_CHECKPOINT}/Wifi_label_{label}.csv", index=False)


def correctWifiFP(wifi_data: pd.DataFrame, t_max_sampling: int, dict_labels_to_meters: dict) -> pd.DataFrame:
    """ 
    Takes the fingerprint of the data and applies the following preprocessing: 
        1) Adjusts the sampling frequency to 1 sample / second: Calculates the mean grouped by second and by label for each of the beacons 
        2) Adjusts missing data: Seconds without data collection are replaced by the minimum RSS of the dataset minus 1 
        3) Scales the values between 0 and 1 (with 0 being the representation of the missing data):
         X = (X - X_min + 1)/X_max
         
    Parameters:
    -----------
    wifi_data: pd.DataFrame
        pd.DataFrame of the corresponding Wi-Fi data
    
    t_max_sampling: int
        Maximum sampling time per label
    
    dict_labels_to_meters: dict
        Dictionary that transforms the label to (longitude, latitude) in meters
    
    Returns:
    --------
    wifi_corrected: pd.DataFrame
        pd.DataFrame of the corrected data
    
    Example:
    --------
    >>> t_max_sampling = constants.T_MAX_SAMPLING # 1140 seconds of data collection for each label in train (60 s in test)
    >>> dict_labels_to_meters = constants.labels_dictionary_meters
    >>> wifi_corrected = correctWifiFP(wifi_data, t_max_sampling=t_max_sampling, dict_labels_to_meters=dict_labels_to_meters)
    """

    list_of_labels = wifi_data.Label.unique().tolist()
    # We take the data from the beacons that have appeared in more than one data collection
    # (we have it in constants.aps)
    aux = wifi_data.query(f'Name_SSID in {constants.aps}')
    # We create a dataframe with all time intervals and all beacons
    labels, time_interval, ssids = [], [], []
    for lab in list_of_labels:
        for ts in range(0, t_max_sampling + 1, 1):
            for ssid in constants.aps:
                labels.append(lab)
                time_interval.append(ts)
                ssids.append(ssid)
    df_intervalos = pd.DataFrame(
        {'Label': labels, 'AppTimestamp(s)': time_interval, 'Name_SSID': ssids})  # Time intervals DataFrame

    aux["AppTimestamp(s)"] = aux["AppTimestamp(s)"].round()  # Round the timestamps to 0 decimals
    aux = aux.groupby(["Label", "AppTimestamp(s)", "Name_SSID"]).mean(numeric_only=True)[
        "RSS"].reset_index()  # Group by label, timestamp and ssid
    aux_corrected = pd.merge(df_intervalos, aux, on=["Label", "AppTimestamp(s)", "Name_SSID"],
                             how="left")  # Join the DataFrame with the time intervals with the grouped DataFrame
    # Replace the NaN values with the minimum value of the dataset minus 1
    aux_corrected = aux_corrected.pivot(index=['Label', 'AppTimestamp(s)'], columns='Name_SSID')[
        "RSS"].reset_index()  # Pivot the DataFrame

    aux_corrected[["Longitude", "Latitude"]] = [dict_labels_to_meters[x] for x in
                                                aux_corrected.Label]  # Add the coordinates

    wifi_order_columns = ["AppTimestamp(s)"] + constants.aps + ["Latitude", "Longitude", "Label"]
    wifi_corrected = aux_corrected[wifi_order_columns]  # Order the columns
    return wifi_corrected


def fix_na_wifi(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes missing values (NaN) in Wi-Fi data by replacing them with a global minimum value minus one.

    Parameters:
    -----------
    data: pd.DataFrame
        The input DataFrame containing Wi-Fi data with potential missing values.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with missing values in Wi-Fi data replaced by a global minimum value.

    Example:
    --------
    >>> import pandas as pd
    >>> from src.utils.preprocess import fix_na_wifi
    >>> wifi_data = pd.read_csv('wifi_data.csv') # Replace with your actual file path
    >>> cleaned_data = fix_na_wifi(wifi_data)
    """

    aux = data.copy()
    min_global = aux[constants.aps].min().min() - 1
    aux[constants.aps] = aux[constants.aps].fillna(min_global)
    return aux


def scale_wifi(data: pd.DataFrame) -> pd.DataFrame:
    """
    Scales the Wi-Fi data within the DataFrame to a normalized range [0, 1].

    Parameters:
    -----------
    data: pd.DataFrame
        The input DataFrame containing Wi-Fi data to be scaled.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with Wi-Fi data scaled to a normalized range.

    Example:
    --------
    >>> import pandas as pd
    >>> from src.utils.preprocess import scale_wifi
    >>> wifi_data = pd.read_csv('wifi_data.csv') # Replace with your actual file path
    >>> scaled_data = scale_wifi(wifi_data)
    """
    aux = data.copy()
    aux[constants.aps] -= aux[constants.aps].min().min() - 1
    aux[constants.aps] /= aux[constants.aps].max().max()
    return aux


def rolling_mean(data: pd.DataFrame, window_size: int, step: int) -> pd.DataFrame:
    """
    Applies a rolling mean to Wi-Fi data,
    filling gaps in timestamps with the previous value only if the previous and subsequent values are equal.

    Parameters:
    -----------
    data: pd.DataFrame
        Dataframe containing Wi-Fi data.

    window_size: int
        Size of the rolling window.

    step: int
        Step size of the rolling window.

    Returns:
    --------
    rolled_data: pd.DataFrame
        DataFrame with the rolling mean applied to the Wi-Fi data.

    """

    n_max = data["AppTimestamp(s)"].max()
    combinations = parse_windows(n_max=n_max, window_size=window_size,
                                 step=step)  # Window combinations
    data_columns = ["AppTimestamp(s)"] + constants.aps + ["Latitude", "Longitude", "Label"]  # Data columns
    rolled_data = pd.DataFrame(columns=data_columns)  # Empty DataFrame
    for lon, lat, lab in data[["Longitude", "Latitude", "Label"]].drop_duplicates().values:  # Iterate over each unique coordinate
        query = data[(data["Longitude"] == lon) & (data["Latitude"] == lat)]  # Filter by coordinate
        for start, end in combinations:  # Iterate over each window combination
            aux = query[(query["AppTimestamp(s)"] >= start) & (query["AppTimestamp(s)"] < end)]  # Filter by window
            mean_by_wifi_column = aux[constants.aps].mean(axis=0).tolist()  # Mean for each AP
            row = [end] + mean_by_wifi_column + [lat, lon]  # Row to be added
            row.append(lab)  # Add the label
            rolled_data.loc[len(rolled_data)] = row  # Add the row

    return rolled_data


def proximity_pixel_interpolation(data: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Applies interpolation to Wi-Fi data,
    filling gaps in timestamps with the previous value only if the previous and subsequent values are equal.

    Parameters:
    -----------
    data: pd.DataFrame
        DataFrame containing Wi-Fi data.

    threshold: int
        Number of consecutive seconds without data collection to consider rejecting interpolation.

    Returns:
    --------
    interpolated_data: pd.DataFrame
        DataFrame with interpolated data.

    Example:
    --------
    Read the dataset

    >>> Data = pd.read_csv(f"{constants.data.train.FINAL_PATH}/groundtruth.csv")

    Apply interpolation

    >>> Interpolated_data = proximity_pixel_interpolation(data, threshold=30)
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

        interpolated_data = pd.concat([interpolated_data, query])

    return interpolated_data


def read_checkpoint(checkpoint_path: str, labels: list) -> pd.DataFrame:
    """
    Reads data from a checkpoint and concatenates it into a single DataFrame.

    Parameters:
    -----------
    checkpoint_path: str
        Path to the checkpoint.

    Returns:
    --------
    df: pd.DataFrame
        DataFrame with concatenated data.

    Example:
    --------
    Path to the accelerometer checkpoint:

    >>> CHECKPOINT_ACCELEROMETER_PATH = "data/train/checkpoint_groundtruth/Accelerometer"

    Path to the gyroscope checkpoint:

    >>> CHECKPOINT_GYROSCOPE_PATH = "data/train/checkpoint_groundtruth/Gyroscope"

    Path to the magnetometer checkpoint:

    >>> CHECKPOINT_MAGNETOMETER_PATH = "data/train/checkpoint_groundtruth/Magnetometer"

    Path to the Wi-Fi checkpoint:

    >>> CHECKPOINT_WIFI_PATH = "data/train/checkpoint_groundtruth/Wi-Fi"

    Reading accelerometer data:

    >>> accelerometer = read_checkpoint(CHECKPOINT_ACCELEROMETER_PATH)

    Reading gyroscope data:

    >>> gyroscope = read_checkpoint(CHECKPOINT_GYROSCOPE_PATH)

    Reading magnetometer data:

    >>> magnetometer = read_checkpoint(CHECKPOINT_MAGNETOMETER_PATH)

    Reading Wi-Fi data:

    >>> wifi = read_checkpoint(CHECKPOINT_WIFI_PATH)
    """

    df = pd.DataFrame()
    for path in glob.glob(os.path.join(checkpoint_path, "*.csv")):
        label = path.split("/")[-1].split("_")[-1].split(".")[0]
        if int(label) in labels:
            df = pd.concat((df, pd.read_csv(path)))
    return df
