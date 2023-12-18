import numpy as np
import pandas as pd
import warnings
from scipy.interpolate import griddata
import sys

sys.path.append("../../")

from src.utils.constants import constants
from src.utils.preprocess import proximity_pixel_interpolation, parse_windows

warnings.filterwarnings('ignore')


def preprocessGrid(grid_z0: np.ndarray) -> np.ndarray:
    """
    This function returns a matrix with interpolated RSS values for the entire latitude and longitude.
    The interpolated values are performed within the sampling space between (minLongitude, minLatitude) and (maxLongitude, maxLatitude). So, any sampling point outside this space is set to np.nan by default.

    This function fills NaN values with one-dimensional linear interpolation for monotonically increasing sample points.

    Parameters:
    -----------
    grid_z0: matrix with interpolated RSS values for the entire latitude and longitude

    Returns:
    --------
    grid_z0: matrix with interpolated RSS values for the entire latitude and longitude

    Examples:
    ---------

    >>> grid_z0 = np.array([[1, 2, 3], [np.nan, 5, 6], [7, 8, np.nan]])

    >>> preprocessGrid(grid_z0)

    >>> array([[1., 2., 3.],
               [4., 5., 6.],
               [7., 8., 6.]])
    """

    grid_z0 = grid_z0.ravel()
    nx = ny = np.int(np.sqrt(grid_z0.shape[0]))

    # Values above 1 and below 0 are meaningless, set to 1 and 0 respectively
    grid_z0[grid_z0 < 0] = 0
    grid_z0[grid_z0 > 1] = 1

    # Find the indices of known values
    known_indexes = np.where(~np.isnan(grid_z0))[0]

    # Find the indices of values to interpolate
    interp_indexes = np.where(np.isnan(grid_z0))[0]

    # Interpolate NaN values with other values
    interp_values = np.interp(interp_indexes, known_indexes, grid_z0[known_indexes])

    # Replace NaN values with interpolated values
    grid_z0[np.isnan(grid_z0)] = interp_values

    return grid_z0.reshape((nx, ny))


def referencePointMap(dataframe: pd.DataFrame, aps_list: list, batch_size: int = 30, step_size: int = 5,
                      size_reference_point_map: int = 300,
                      return_axis_coords: bool = False):
    """
    This function returns a matrix with interpolated RSS values for each AP (WiFi) for the entire latitude and longitude in the sampling space.

    Parameters:
    -----------
    dataframe: pd.DataFrame
        DataFrame with the RSS values for each AP (WiFi) for each latitude and longitude annotated in a Reference Point (RP)

    aps_list: list
        List of APs (WiFi) to consider for generating the continuous reference map

    batch_size: int = 30
        Time window size for generating each continuous reference map (number of seconds to consider for calculating the mean RSS at each RP)

    step_size: int = 5
        Number of seconds the time window shifts for generating each continuous reference map. If no overlapping is desired, assign the same value as batch_size

    size_reference_point_map: int
        Size of the continuous reference map (number of RPs sampled in each dimension)

    return_axis_coords: bool
        If True, returns the continuous longitude and latitude coordinates of the reference map. If False, returns only the continuous reference map and the labels of the APs (WiFi)

    Returns:
    --------
    reference_point_map: np.ndarray
        Matrix with interpolated RSS values for each AP (WiFi) for the entire latitude and longitude in the sampling space

    APLabel: np.ndarray
        Labels of the APs (WiFi) for each continuous reference map

    -----------------------------------------
    In case return_axis_coords is True:
    -----------------------------------------
    x_g: np.ndarray
        Continuous longitude coordinates of the continuous reference map

    y_g: np.ndarray
        Continuous latitude coordinates of the continuous reference map

    Examples:
    ---------

    Data reading

    >>> dataframe = pd.DataFrame({
            "AppTimestamp(s)": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "Longitude": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "Latitude": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "GEOTECWIFI03": [0, 0, 0.2, 0, 0.2, 0.8, 1, 0.8, 1, 1],
            "eduroam": [1, 0.9, 1, 0.9, 0.8, 0.2, 0.1, 0.2, 0, 0],
        })

    List of APs

    >>> aps_list = ["GEOTECWIFI03", "eduroam"]

    Generating continuous reference map

    >>> reference_point_map, APLabel = referencePointMap(dataframe, aps_list=aps_list, batch_size=5, step_size=2, size_reference_point_map=2)
    """
    nx = ny = size_reference_point_map
    t_max = dataframe["AppTimestamp(s)"].max()
    samples_per_RP = int((t_max - batch_size) / step_size) + 1
    RPMap = np.zeros((samples_per_RP * len(aps_list), nx, ny))
    APLabel = []
    combinaciones = parse_windows(t_max, batch_size, step_size)

    for n_ap in range(0, len(aps_list) * samples_per_RP, samples_per_RP):
        ap = aps_list[n_ap // samples_per_RP]
        for batch, (start, end) in enumerate(combinaciones):
            aux = dataframe[(dataframe["AppTimestamp(s)"] >= start) & (dataframe["AppTimestamp(s)"] < end)]
            aux = aux.groupby(["Longitude", "Latitude"]).mean()[ap].reset_index()
            miny, maxy = min(aux['Latitude']), max(aux['Latitude'])
            minx, maxx = min(aux['Longitude']), max(aux['Longitude'])
            grdi_x = np.linspace(minx, maxx, num=nx, endpoint=False)  # Discrete x coordinates
            grdi_y = np.linspace(miny, maxy, num=ny, endpoint=False)  # Discrete y coordinates
            yg, xg = np.meshgrid(grdi_y, grdi_x, indexing='ij')  # Continuous x and y coordinates
            x_g = xg.ravel()  # Flattened continuous x coordinates
            y_g = yg.ravel()  # Flattened continuous y coordinates
            aux2 = aux.drop([ap], 1)
            points = np.array(aux2)
            values = np.array(aux[ap])
            grid_z0 = griddata(points, values, (x_g, y_g), method='cubic')

            RPMap[batch + n_ap, :, :] = preprocessGrid(grid_z0)
            APLabel.append(ap)

    APLabel = np.array(APLabel)
    if return_axis_coords:
        return RPMap, APLabel, x_g, y_g

    return RPMap, APLabel


def labelEncoding(labels: np.ndarray) -> np.ndarray:
    """
    This function returns an array with the values of the labels encoded as integers.

    Parameters:
    -----------
    labels: np.ndarray
        Array with the values of the labels to encode

    Returns:
    --------
        numericLabels: np.ndarray
            Array with the values of the labels encoded as integers

    Examples:
    ---------

    >>> labels = np.array(["GEOTECWIFI03", "480Invitados", "eduroam", "wpen-uji", "lt1iot", "cuatroochenta", "UJI"])

    >>> labelEncoding(labels)

    >>> array([0, 1, 2, 3, 4, 5, 6])
    """
    numericLabels = labels.copy()
    numericLabels[numericLabels == constants.aps[0]] = 0
    numericLabels[numericLabels == constants.aps[1]] = 1
    numericLabels[numericLabels == constants.aps[2]] = 2
    numericLabels[numericLabels == constants.aps[3]] = 3
    numericLabels[numericLabels == constants.aps[4]] = 4
    numericLabels[numericLabels == constants.aps[5]] = 5
    numericLabels[numericLabels == constants.aps[6]] = 6
    return numericLabels.astype(int)


def labelDecoding(labels: np.ndarray) -> np.ndarray:
    """
    This function returns an array with the values of the labels decoded into strings.

    Parameters:
    -----------
    labels: np.ndarray
        Array with the values of the labels to decode

    Returns:
    --------
    categoricLabels: np.ndarray
        Array with the values of the labels decoded into strings

    Examples:
    ---------

    >>> labels = np.array([0, 1, 2, 3, 4, 5, 6])

    >>> labelDecoding(labels)

    >>> array(["GEOTECWIFI03", "480Invitados", "eduroam", "wpen-uji", "lt1iot", "cuatroochenta", "UJI"])
    """

    categoricLabels = labels.copy()
    if len(categoricLabels.shape) == 2:
        categoricLabels = categoricLabels.reshape(categoricLabels.shape[0], )

    categoricLabels = pd.Series(categoricLabels).map(lambda x: constants.dictionary_decoding[x]).to_numpy()
    return categoricLabels


def get_radiomap_from_rpmap(rpmap, x_coords, y_coords):
    """
    This function takes data from the continuous reference point map and transforms it into a radiomap.

    Parameters:
    -----------
    rpmap: np.ndarray
        Matrix with the RSS values of each AP (Wi-Fi) for every latitude and longitude in the sampling space

    x_coords: np.ndarray
        Continuous longitude coordinates of the continuous reference point map

    y_coords: np.ndarray
        Continuous latitude coordinates of the continuous reference point map

    Returns:
    --------
    radiomap_extended: pd.DataFrame
        DataFrame with the RSS values of each AP (Wi-Fi) for each x and y coordinate in the sampling space

    Examples:
    ---------
    Obtaining the rpmap and coordinates

    >>> dataloader = DataLoader(data_dir=f"{constants.data.FINAL_PATH}/groundtruth.csv",
                                aps_list=constants.aps, batch_size=30, step_size=5,
                                size_reference_point_map=28,
                                return_axis_coords=True)
    >>> X, _, [x_coords, y_coords] = dataloader()

    >>> rpmap = X[:,:,:,0]

    Getting the radiomap using the continuous mesh reference map and the sampling coordinates

    >>> radiomap = get_radiomap_from_rpmap(rpmap, x_coords, y_coords)
    """
    # Constants for the radiomap transformation operation
    samples_per_RP = int(rpmap.shape[0] / len(constants.aps))  # 223 temporal instances per ap
    size_map = rpmap.shape[1]  # 28x28 reference points
    radiomap = np.zeros((samples_per_RP * size_map * size_map,
                         len(constants.aps) + 2))  # 174832 rows, 7 columns Wi-Fi + 2 columns coordinates
    count = 0  # initialize the row counter
    rpmap_flatten = rpmap.reshape(int(rpmap.shape[0]), size_map * size_map)  # flatten each reference map

    for batch in range(samples_per_RP):  # for each temporal instance
        for idx_coord, (x, y) in enumerate(zip(x_coords, y_coords)):  # for each coordinate
            for n_ap in range(len(constants.aps)):  # for each ap
                radiomap[count, n_ap] = rpmap_flatten[
                    batch + n_ap * samples_per_RP, idx_coord]  # save the Wi-Fi signal value in the corresponding row
            radiomap[count, len(constants.aps)] = x  # save the x coordinate
            radiomap[count, len(constants.aps) + 1] = y  # save the y coordinate
            count += 1  # increment the row counter

    radiomap_extended = pd.DataFrame(radiomap,
                                     columns=constants.aps + ["Longitude", "Latitude"])  # convert to a dataframe
    return radiomap_extended


class DataLoader:
    """
    This class is responsible for loading data from continuous reference point maps for each AP (Wi-Fi) and their labels.

    Attributes:
    -----------
    data_dir: str
        Path to the csv file with data from continuous reference point maps for each AP (Wi-Fi) and their labels
    aps_list: list
        List of APs (Wi-Fi) to consider for generating the continuous reference point map
    batch_size: int = 30
        Size of the time window for generating each continuous reference point map (number of seconds to consider for calculating the mean RSS at each RP)
    step_size: int = 5
        Number of seconds the time window shifts for generating each continuous reference point map. If we don't want overlapping windows, we have to assign the same value as batch_size
    size_reference_point_map: int = 300
        Size of the continuous reference point map (number of RPs sampled in each dimension)
    return_axis_coords: bool = False
        If True, returns the x and y coordinates of the continuous reference point map. If False, returns only the continuous reference point map and the labels of the APs (Wi-Fi)

    Methods:
    --------
    __getData(data_dir: str) -> pd.DataFrame:
        This function returns a DataFrame with the data from continuous reference point maps for each AP (Wi-Fi) and their labels

    __call__(*args, **kwargs) -> np.ndarray:
        This function returns the continuous reference point maps for each AP (Wi-Fi) and their labels

    Examples:
    ---------

    With overlapping

    >>> dataloader = DataLoader(data_dir=f"{constants.data.FINAL_PATH}/groundtruth.csv",
                                aps_list=constants.aps, batch_size=30, step_size=5,
                                size_reference_point_map=300,
                                return_axis_coords=False)

    >>> X, y, _ = dataloader()

    Without overlapping

    >>> dataloader = DataLoader(data_dir=f"{constants.data.FINAL_PATH}/groundtruth.csv",
                                aps_list=constants.aps, batch_size=30, step_size=30,
                                size_reference_point_map=300,
                                return_axis_coords=False)

    >>> X, y, _ = dataloader()

    Return x and y coordinates

    >>> dataloader = DataLoader(data_dir=f"{constants.data.FINAL_PATH}/groundtruth.csv",
                                aps_list=constants.aps, batch_size=30, step_size=5,
                                size_reference_point_map=300,
                                return_axis_coords=True)

    >>> X, y, [x_coords, y_coords] = dataloader()
    """

    def __init__(self, data_dir: str, aps_list: list, batch_size: int = 30, step_size: int = 5,
                 size_reference_point_map: int = 300,
                 return_axis_coords: bool = False):
        self.groundtruth = self.__getData(data_dir)
        self.aps_list = aps_list
        self.batch_size = batch_size
        self.step_size = step_size
        self.size_reference_point_map = size_reference_point_map
        self.return_axis_coords = return_axis_coords

    @staticmethod
    def __getData(data_dir):
        groundtruth = pd.read_csv(data_dir)
        new_columns = ["AppTimestamp(s)"] + constants.aps + constants.accelerometer_cols + \
                      constants.gyroscope_cols + constants.magnetometer_cols + ["Latitude", "Longitude", "Label"]

        groundtruth = groundtruth[new_columns]
        print("Interpolating data with nearest pixel if they match")
        groundtruth_interpolated = proximity_pixel_interpolation(groundtruth, threshold=30)
        return groundtruth_interpolated

    def __call__(self, *args, **kwargs):
        [X, y, *coords] = referencePointMap(self.groundtruth, aps_list=self.aps_list,
                                            batch_size=self.batch_size,
                                            step_size=self.step_size,
                                            size_reference_point_map=self.size_reference_point_map,
                                            return_axis_coords=self.return_axis_coords)
        X, y = np.expand_dims(X, axis=-1), np.expand_dims(y, axis=-1)
        return X, y, coords
