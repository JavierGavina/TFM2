class Train:
    """
    Class that contains the constants for the project directories.

    Attributes:
    ___________
        INITIAL_DATA: str
            The directory for raw data (in .txt format).
        CHECKPOINT_DATA_PATH: str
            The directory for checkpoint data for each metric and WiFi.
        MID_PATH: str
            The output directory for each dataset (metrics and WiFi).
        FINAL_PATH: str
            Merged data.
    """
    # Definición de las constantes
    INITIAL_DATA = "data/train/initial_rp_data"  # La dirección de los datos extendida
    CHECKPOINT_DATA_PATH = "output/data/train/checkpoint_groundtruth"
    RAW_OUT_PATH = "output/data/train/raw_radiomap"
    PROC_OUT_PATH = "output/data/train/processed_radiomap"


class Partitions:
    PARTITION_5VS18 = "output/data/partitions/partition_5vs18"
    PARTITION_10VS13 = "output/data/partitions/partition_10vs13"
    PARTITION_15VS8 = "output/data/partitions/partition_15vs8"


class Test:
    """
    Class that contains the constants for the project directories.

    Attributes:
    ___________
        INITIAL_DATA: str
            The directory for raw data (in .txt format).
        CHECKPOINT_DATA_PATH: str
            The directory for checkpoint data for each metric and WiFi.
        MID_PATH: str
            The output directory for each dataset (metrics and WiFi).
        FINAL_PATH: str
            Merged data.
    """

    # Definición de las constantes
    INITIAL_DATA = "data/test/initial_rp_data"  # La dirección de los datos extendida
    CHECKPOINT_DATA_PATH = "output/data/test/checkpoint_groundtruth"
    RAW_OUT_PATH = "output/data/test/raw_radiomap"
    PROC_OUT_PATH = "output/data/test/processed_radiomap"


class Directories:
    """
    Class that contains the constants for the project directories.

    Attributes:
    ___________
        train: Train
            Class containing constants for training data directories.
        test: Test
            Class containing constants for testing data directories.
    """

    train = Train()
    test = Test()
    partitions = Partitions()


class Outputs:
    """
    Class that contains the constants for the directories of the project outputs.

    Attributes:
    ___________
        PATH_OUTPUTS: str
            The root directory for the outputs.
        PATH_RPMAP: str
            The directory for continuous reference maps.
    """

    PATH_OUTPUTS = "output"
    OUT_DATA = "output/data"
    TRAIN_OUT = "output/data/train"
    TEST_OUT = "output/data/test"
    PARTITIONS = "output/data/partitions"
    POSITIONING_PARTITIONS = "output/positioning_partitions"


class constants:
    """
    Class that contains the constants for the project.

    Attributes:
    ___________
        dictionary_decoding: dict
            Dictionary that transforms the label to (longitude, latitude) in meters.
        data: Directories
            Class containing constants for the project directories.
        aps: list
            List of APs (Wi-Fi) to consider for the generation of the continuous reference map.
        magnetometer_cols: list
            List of magnetometer columns.
        accelerometer_cols: list
            List of accelerometer columns.
        gyroscope_cols: list
            List of gyroscope columns.
        labels_dictionary_meters: dict
            Dictionary that transforms the label to (longitude, latitude) in meters.
        outputs: Outputs
            Class containing constants for the directories of all project outputs.
    """

    T_MAX_SAMPLING = 1140  # Maximum number of seconds for sample collection per Reference Point in TRAIN
    T_MAX_SAMPLING_TEST = 60  # Maximum number of seconds for sample collection per Reference Point in TEST

    dictionary_decoding = {
        0: "GEOTECWIFI03", 1: "480Invitados",
        2: "eduroam", 3: "wpen-uji",
        4: "lt1iot", 5: "cuatroochenta",
        6: "UJI"
    }

    aps = ['GEOTECWIFI03', '480Invitados', 'eduroam', 'wpen-uji', 'lt1iot', 'cuatroochenta', 'UJI']

    colors_test = ["red", "blue", "green", "yellow", "purple", "black", "pink", "brown", "orange", "gray", "cyan",
                   "magenta", "lime", "olive", "teal", "navy", "maroon"]

    # Dictionary that transforms the label to (longitude, latitude) in meters TRAIN
    labels_dictionary_meters = {
        0: (0.6, 0), 1: (5.4, 0), 2: (9, 0),
        3: (9, 3), 4: (6, 3), 5: (3, 3),
        6: (0.6, 3), 7: (0.6, 4.8), 8: (3.6, 4.8),
        9: (6, 4.8), 10: (9, 4.8), 11: (9, 7.8),
        12: (6.6, 7.8), 13: (3, 7.8), 14: (0.6, 7.8),
        15: (0.6, 9.6), 16: (3, 9.6), 17: (4.8, 9.6),
        18: (8.4, 9.6), 19: (8.4, 12), 20: (8.4, 14.4),
        21: (3, 14.4), 22: (0, 14.4)
    }

    # Dictionary that transforms the label to (longitude, latitude) in meters TEST
    labels_dictionary_meters_test = {
        0: (3.6, 0), 1: (7.2, 0), 2: (9, 1.2),
        3: (7.2, 3), 4: (4.2, 3), 5: (1.8, 3),
        6: (2.4, 4.2), 7: (7.8, 4.2), 8: (9., 6.6),
        9: (4.8, 7.8), 10: (1.8, 7.8), 11: (0, 9),
        12: (1.8, 10.2), 13: (6.6, 10.2), 14: (8.4, 10.8),
        15: (5.4, 13.8), 16: (1.2, 13.2)
    }

    labels_train = [x for x in range(23)]
    labels_test = [x for x in range(17)]
    labels_partition_5vs18 = [0, 2, 11, 14, 21]
    labels_partition_10vs13 = [0, 2, 3, 5, 9, 13, 19, 20, 21, 22]
    labels_partition_15vs8 = [0, 2, 4, 5, 7, 9, 10, 11, 14, 15, 17, 18, 20, 21, 22]

    data = Directories()
    outputs = Outputs()
