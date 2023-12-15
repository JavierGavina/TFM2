import os

import warnings

from utils.constants import constants
from utils.preprocess import correctWifiFP, read_checkpoint, interpolacion_pixel_proximo
from utils.preprocess import fix_na_wifi, rolling_mean, scale_wifi, get_checkpoints_data

# Definición de las constantes de los directorios
CHECKPOINT_DATA_PATH = constants.data.train.CHECKPOINT_DATA_PATH
WIFI_CHECKPOINT = f"{CHECKPOINT_DATA_PATH}/Wifi"

# Tiempo maximo de muestreo por label
t_max_sampling = constants.T_MAX_SAMPLING

# Diccionario de labels a metros
labels_dictionary_meters = constants.labels_dictionary_meters

warnings.filterwarnings("ignore")


def main():
    get_checkpoints_data(constants.data.train.INITIAL_DATA, labels_dictionary_meters)

    wifi_data = read_checkpoint(WIFI_CHECKPOINT, constants.labels_train)
    wifi_corrected = correctWifiFP(wifi_data=wifi_data,
                                   t_max_sampling=t_max_sampling,
                                   dict_labels_to_meters=labels_dictionary_meters)
    wifi_corrected = fix_na_wifi(wifi_corrected)
    wifi_corrected = interpolacion_pixel_proximo(wifi_corrected, threshold=30)

    '''
    Obtenemos wifi sin escalar los datos a 0 - 1
    '''

    raw_wifi = rolling_mean(wifi_corrected, window_size=30, step=5)

    os.makedirs(constants.data.train.RAW_OUT_PATH, exist_ok=True)  # Creamos el directorio en bruto
    raw_wifi.to_csv(f"{constants.data.train.RAW_OUT_PATH}/raw_radiomap.csv", index=False)

    '''
    Obtenemos wifi escalando los datos a 0 - 1
    '''

    proc_wifi = scale_wifi(wifi_corrected)
    proc_wifi = rolling_mean(proc_wifi, window_size=30, step=5)

    os.makedirs(constants.data.train.PROC_OUT_PATH, exist_ok=True)  # Creamos el directorio procesado
    proc_wifi.to_csv(f"{constants.data.train.PROC_OUT_PATH}/processed_radiomap.csv", index=False)


if __name__ == "__main__":
    main()
