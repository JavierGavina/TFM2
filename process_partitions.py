import os, tqdm

import pandas as pd
from src.constants import constants
from src.preprocess import correctWifiFP, read_checkpoint, interpolacion_pixel_proximo
from src.preprocess import fix_na_wifi, rolling_mean, scale_wifi

# Cogemos los ssids que han aparecido en más de una recogida de datos
lista_ssid_candidatos = constants.aps

# Tiempo maximo de muestreo por label
t_max_sampling = constants.T_MAX_SAMPLING

CHECKPOINT_DATA_PATH = constants.data.train.CHECKPOINT_DATA_PATH
WIFI_CHECKPOINT = f"{CHECKPOINT_DATA_PATH}/Wifi"

part_5vs18 = constants.data.partitions.PARTITION_5VS18
part_10vs13 = constants.data.partitions.PARTITION_10VS13
part_15vs8 = constants.data.partitions.PARTITION_15VS8

labels_test_5vs18 = [x for x in range(23) if x not in constants.labels_partition_5vs18]
labels_test_10vs13 = [x for x in range(23) if x not in constants.labels_partition_10vs13]
labels_test_15vs8 = [x for x in range(23) if x not in constants.labels_partition_15vs8]
def main():
    os.makedirs("partitions", exist_ok=True)
    os.makedirs(part_5vs18, exist_ok=True)
    os.makedirs(f"{part_5vs18}/train", exist_ok=True)
    os.makedirs(f"{part_5vs18}/train/raw", exist_ok=True)
    os.makedirs(f"{part_5vs18}/train/processed", exist_ok=True)
    os.makedirs(f"{part_5vs18}/test", exist_ok=True)
    os.makedirs(f"{part_5vs18}/test/raw", exist_ok=True)
    os.makedirs(f"{part_5vs18}/test/processed", exist_ok=True)
    os.makedirs(part_10vs13, exist_ok=True)
    os.makedirs(f"{part_10vs13}/train", exist_ok=True)
    os.makedirs(f"{part_10vs13}/train/raw", exist_ok=True)
    os.makedirs(f"{part_10vs13}/train/processed", exist_ok=True)
    os.makedirs(f"{part_10vs13}/test", exist_ok=True)
    os.makedirs(f"{part_10vs13}/test/raw", exist_ok=True)
    os.makedirs(f"{part_10vs13}/test/processed", exist_ok=True)
    os.makedirs(part_15vs8, exist_ok=True)
    os.makedirs(f"{part_15vs8}/train", exist_ok=True)
    os.makedirs(f"{part_15vs8}/train/raw", exist_ok=True)
    os.makedirs(f"{part_15vs8}/train/processed", exist_ok=True)
    os.makedirs(f"{part_15vs8}/test", exist_ok=True)
    os.makedirs(f"{part_15vs8}/test/raw", exist_ok=True)
    os.makedirs(f"{part_15vs8}/test/processed", exist_ok=True)

    print("=======================================")
    print(f"Partición 5vs18 {constants.labels_partition_5vs18} puntos train")
    print("=======================================")

    part_5v18_train = read_checkpoint(WIFI_CHECKPOINT, constants.labels_partition_5vs18) # Leemos los datos del checkpoint
    part_5v18_train = correctWifiFP(wifi_data=part_5v18_train,
                                    t_max_sampling=t_max_sampling,
                                    dict_labels_to_meters=constants.labels_dictionary_meters) # Corregimos los datos de wifi y obtenemos formato fingerprint

    part_5v18_train = fix_na_wifi(part_5v18_train) # Corregimos los datos ausentes wifi (NaN = min - 1)
    part_5v18_train = interpolacion_pixel_proximo(part_5v18_train, threshold=30) # Interpolamos los datos de wifi

    part_5v18_train_raw = rolling_mean(part_5v18_train, window_size=30, step=5) # Obtenemos wifi sin escalar los datos a 0 - 1
    part_5v18_train_raw.to_csv(f"{part_5vs18}/train/raw/raw_radiomap.csv", index=False) # Guardamos los datos en bruto

    part_5v18_train_proc = scale_wifi(part_5v18_train) # Obtenemos wifi escalando los datos a 0 - 1
    part_5v18_train_proc = rolling_mean(part_5v18_train_proc, window_size=30, step=5) # Obtenemos wifi sin escalar los datos a 0 - 1
    part_5v18_train_proc.to_csv(f"{part_5vs18}/train/processed/processed_radiomap.csv", index=False) # Guardamos los datos procesados

    print("=======================================")
    print(f"Partición 5v18 {labels_test_5vs18} puntos test")
    print("=======================================")

    part_5v18_test = read_checkpoint(WIFI_CHECKPOINT, labels_test_5vs18) # Leemos los datos del checkpoint
    part_5v18_test = correctWifiFP(wifi_data=part_5v18_test,
                                   t_max_sampling=t_max_sampling,
                                   dict_labels_to_meters=constants.labels_dictionary_meters) # Corregimos los datos de wifi y obtenemos formato fingerprint

    part_5v18_test = fix_na_wifi(part_5v18_test) # Corregimos los datos ausentes wifi (NaN = min - 1)
    part_5v18_test = interpolacion_pixel_proximo(part_5v18_test, threshold=30) # Interpolamos los datos de wifi

    part_5v18_test_raw = rolling_mean(part_5v18_test, window_size=30, step=5) # Obtenemos wifi sin escalar los datos a 0 - 1
    part_5v18_test_raw.to_csv(f"{part_5vs18}/test/raw/raw_radiomap.csv", index=False) # Guardamos los datos en bruto

    part_5v18_test_proc = scale_wifi(part_5v18_test) # Obtenemos wifi escalando los datos a 0 - 1
    part_5v18_test_proc = rolling_mean(part_5v18_test_proc, window_size=30, step=5) # Obtenemos wifi sin escalar los datos a 0 - 1
    part_5v18_test_proc.to_csv(f"{part_5vs18}/test/processed/processed_radiomap.csv", index=False) # Guardamos los datos procesados

    print("=======================================")
    print(f"Partición 10vs13 {constants.labels_partition_10vs13} puntos train")
    print("=======================================")

    part_10v13_train = read_checkpoint(WIFI_CHECKPOINT, constants.labels_partition_10vs13)
    part_10v13_train = correctWifiFP(wifi_data=part_10v13_train,
                                     t_max_sampling=t_max_sampling,
                                     dict_labels_to_meters=constants.labels_dictionary_meters)

    part_10v13_train = fix_na_wifi(part_10v13_train)
    part_10v13_train = interpolacion_pixel_proximo(part_10v13_train, threshold=30)

    part_10v13_train_raw = rolling_mean(part_10v13_train, window_size=30, step=5)
    part_10v13_train_raw.to_csv(f"{part_10vs13}/train/raw/raw_radiomap.csv", index=False)

    part_10v13_train_proc = scale_wifi(part_10v13_train)
    part_10v13_train_proc = rolling_mean(part_10v13_train_proc, window_size=30, step=5)
    part_10v13_train_proc.to_csv(f"{part_10vs13}/train/processed/processed_radiomap.csv", index=False)

    print("=======================================")
    print(f"Partición 10vs13 {labels_test_10vs13} puntos test")
    print("=======================================")

    part_10v13_test = read_checkpoint(WIFI_CHECKPOINT, labels_test_10vs13)
    part_10v13_test = correctWifiFP(wifi_data=part_10v13_test,
                                    t_max_sampling=t_max_sampling,
                                    dict_labels_to_meters=constants.labels_dictionary_meters)

    part_10v13_test = fix_na_wifi(part_10v13_test)
    part_10v13_test = interpolacion_pixel_proximo(part_10v13_test, threshold=30)

    part_10v13_test_raw = rolling_mean(part_10v13_test, window_size=30, step=5)
    part_10v13_test_raw.to_csv(f"{part_10vs13}/test/raw/raw_radiomap.csv", index=False)

    part_10v13_test_proc = scale_wifi(part_10v13_test)
    part_10v13_test_proc = rolling_mean(part_10v13_test_proc, window_size=30, step=5)

    part_10v13_test_proc.to_csv(f"{part_10vs13}/test/processed/processed_radiomap.csv", index=False)

    print("=======================================")
    print(f"Partición 15vs8 {constants.labels_partition_15vs8} puntos train")
    print("=======================================")

    part_15v8_train = read_checkpoint(WIFI_CHECKPOINT, constants.labels_partition_15vs8)
    part_15v8_train = correctWifiFP(wifi_data=part_15v8_train,
                                    t_max_sampling=t_max_sampling,
                                    dict_labels_to_meters=constants.labels_dictionary_meters)

    part_15v8_train = fix_na_wifi(part_15v8_train)
    part_15v8_train = interpolacion_pixel_proximo(part_15v8_train, threshold=30)

    part_15v8_train_raw = rolling_mean(part_15v8_train, window_size=30, step=5)
    part_15v8_train_raw.to_csv(f"{part_15vs8}/train/raw/raw_radiomap.csv", index=False)

    part_15v8_train_proc = scale_wifi(part_15v8_train)
    part_15v8_train_proc = rolling_mean(part_15v8_train_proc, window_size=30, step=5)
    part_15v8_train_proc.to_csv(f"{part_15vs8}/train/processed/processed_radiomap.csv", index=False)

    print("=======================================")
    print(f"Partición 15vs8 {labels_test_15vs8} puntos test")
    print("=======================================")

    part_15v8_test = read_checkpoint(WIFI_CHECKPOINT, labels_test_15vs8)
    part_15v8_test = correctWifiFP(wifi_data=part_15v8_test,
                                   t_max_sampling=t_max_sampling,
                                   dict_labels_to_meters=constants.labels_dictionary_meters)

    part_15v8_test = fix_na_wifi(part_15v8_test)
    part_15v8_test = interpolacion_pixel_proximo(part_15v8_test, threshold=30)

    part_15v8_test_raw = rolling_mean(part_15v8_test, window_size=30, step=5)
    part_15v8_test_raw.to_csv(f"{part_15vs8}/test/raw/raw_radiomap.csv", index=False)

    part_15v8_test_proc = scale_wifi(part_15v8_test)
    part_15v8_test_proc = rolling_mean(part_15v8_test_proc, window_size=30, step=5)
    part_15v8_test_proc.to_csv(f"{part_15vs8}/test/processed/processed_radiomap.csv", index=False)


if __name__ == "__main__":
    main()
