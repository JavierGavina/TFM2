class Train:
    """
    Clase que contiene las constantes de los directorios del proyecto

    Attributes:
    ___________
        INITIAL_DATA: str
            La dirección de los datos en bruto (formato .txt)
        CHECKPOINT_DATA_PATH: str
            La dirección de los datos del checkpoint de cada métrica y wifi
        MID_PATH: str
            La dirección de salida de cada dataset (metricas y wifi)
        FINAL_PATH: str
            Datos unidos
    """
    # Definición de las constantes
    INITIAL_DATA = "data/train/initial_rp_data"  # La dirección de los datos extendida
    CHECKPOINT_DATA_PATH = "data/train/checkpoint_groundtruth"
    RAW_OUT_PATH = "data/train/raw_radiomap"
    PROC_OUT_PATH = "data/train/processed_radiomap"


class Partitions:
    PARTITION_5VS18 = "data/partitions/partition_5vs18"
    PARTITION_10VS13 = "data/partitions/partition_10vs13"
    PARTITION_15VS8 = "data/partitions/partition_15vs8"


class Test:
    """
    Clase que contiene las constantes de los directorios del proyecto

    Attributes:
    ___________
        INITIAL_DATA: str
            La dirección de los datos en bruto (formato .txt)
        CHECKPOINT_DATA_PATH: str
            La dirección de los datos del checkpoint de cada métrica y wifi
        MID_PATH: str
            La dirección de salida de cada dataset (metricas y wifi)
        FINAL_PATH: str
            Datos unidos
    """
    # Definición de las constantes
    INITIAL_DATA = "data/test/initial_rp_data"  # La dirección de los datos extendida
    CHECKPOINT_DATA_PATH = "data/test/checkpoint_groundtruth"
    RAW_OUT_PATH = "data/test/raw_radiomap"
    PROC_OUT_PATH = "data/test/processed_radiomap"


class Directories:
    """
    Clase que contiene las constantes de los directorios del proyecto

    Attributes:
    ___________
        train: Train
            Clase que contiene las constantes de los directorios de los datos de entrenamiento
        test: Test
            Clase que contiene las constantes de los directorios de los datos de testeo
    """
    train = Train()
    test = Test()
    partitions = Partitions()


class Positioning:
    """
    Clase que contiene las constantes de los outputs de posicionamiento del proyecto

    Attributes:
    ___________
        POSITIONING: str
            La dirección de los modelos de posicionamiento
        POSITIONING_ESTIMATION: str
            La dirección de los modelos de estimación de posicionamiento
    """
    positioning_path = "outputs/positioning"
    without_rpmap = f"{positioning_path}/without_rpmap"
    rpmap = f"{positioning_path}/rpmap"
    rpmap_data_augmentation = f"{positioning_path}/rpmap_data_augmentation"


class Architectures:
    """
    Clase que contiene las constantes de los directorios de las arquitecturas de los modelos
    """
    arquitectures = "outputs/model_architecture"
    cgan_300 = f"{arquitectures}/cGAN_300_300"
    cgan_28 = f"{arquitectures}/cGAN_28_28"


class Models:
    """
    Clase que contiene las constantes de los directorios de los modelos del proyecto
    """
    training = "outputs/process_training"
    cgan_300 = "outputs/process_training/cGAN_300_300"
    cgan_28 = "outputs/process_training/cGAN_28_28"
    wasserstein_cgan_28 = "outputs/process_training/wasserstein_cGAN_28_28"
    wasserstein_cgan_gp_28 = "outputs/process_training/wasserstein_cGAN_GP_28_28"


class RPMAP:
    PATH_RPMAP = "outputs/RPMap"
    rpmap_300_overlapping = f"{PATH_RPMAP}/rpmap_300_overlapping"
    rpmap_300_sinOverlapping = f"{PATH_RPMAP}/rpmap_300_sinOverlapping"
    rpmap_28_overlapping = f"{PATH_RPMAP}/rpmap_28_overlapping"


class Outputs:
    """
    Clase que contiene las constantes de los directorios de los outputs del proyecto

    Attributes:
    ___________
        PATH_OUTPUTS: str
            La dirección raíz de los outputs
        PATH_RPMAP: str
            La dirección de los mapas de referencia continua
    """
    PATH_OUTPUTS = "outputs"
    GENERATIVE_METRICS = "outputs/generative_metrics"
    models = Models()
    rpmap = RPMAP()
    architectures = Architectures()
    positioning = Positioning()


class constants:
    """
    Clase que contiene las constantes del proyecto

    Attributes:
    ___________
        dictionary_decoding: dict
            Diccionario que transforma el label a (longitud, latitud) en metros
        data: Directories
            Clase que contiene las constantes de los directorios del proyecto
        aps: list
            Lista de APs (wifi) a considerar para la generación del mapa de referencia continua
        magnetometer_cols: list
            Lista de columnas del magnetómetro
        accelerometer_cols: list
            Lista de columnas del acelerómetro
        gyroscope_cols: list
            Lista de columnas del giroscopio
        labels_dictionary_meters: dict
            Diccionario que transforma el label a (longitud, latitud) en metros
        outputs: Outputs
            Clase que contiene las constantes de los directorios de todas las salidas del proyecto
    """

    T_MAX_SAMPLING = 1140  # Número de segundos máximo de recogida de muestras por cada Reference Point en TRAIN
    T_MAX_SAMPLING_TEST = 60  # Número de segundos máximo de recogida de muestras por cada Reference Point en TEST

    dictionary_decoding = {
        0: "GEOTECWIFI03", 1: "480Invitados",
        2: "eduroam", 3: "wpen-uji",
        4: "lt1iot", 5: "cuatroochenta",
        6: "UJI"
    }

    aps = ['GEOTECWIFI03', '480Invitados', 'eduroam', 'wpen-uji', 'lt1iot', 'cuatroochenta', 'UJI']

    colors_test = ["red", "blue", "green", "yellow", "purple", "black", "pink", "brown", "orange", "gray", "cyan",
                   "magenta", "lime", "olive", "teal", "navy", "maroon"]

    # Diccionario que transforma el label a (longitud, latitud) en metros TRAIN
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

    # Diccionario que transforma el label a (longitud, latitud) en metros TEST
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
