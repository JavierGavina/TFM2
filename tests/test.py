import unittest
import os
import pandas as pd

from src.utils import preprocess
from src.utils.constants import constants


class TestUtilsPreprocess(unittest.TestCase):

    def test_parse_windows(self):
        """
        This test checks if the windows are correctly parsed.
        """
        # Case 1
        result = preprocess.parse_windows(10, 5, 2)
        expected_result = [(0, 5), (2, 7), (4, 9)]
        self.assertEqual(result, expected_result)

        # Case 2
        result = preprocess.parse_windows(10, 5, 5)
        expected_result = [(0, 5), (5, 10)]
        self.assertEqual(result, expected_result)

        # Case 3
        result = preprocess.parse_windows(10, 5, 100)
        expected_result = [(0, 5)]
        self.assertEqual(result, expected_result)

    def test_get_checkpoint_data(self):
        path_checkpoint_train = f"../{constants.data.train.CHECKPOINT_DATA_PATH}/Wifi"
        path_checkpoint_test = f"../{constants.data.test.CHECKPOINT_DATA_PATH}/Wifi"

        wifi_out_train = sorted(os.listdir(path_checkpoint_train))
        wifi_out_test = sorted(os.listdir(path_checkpoint_test))

        expected_out_train = sorted([f"Wifi_label_{x}.csv" for x in constants.labels_train])
        expected_out_test = sorted([f"Wifi_label_{x}.csv" for x in constants.labels_test])

        self.assertEqual(wifi_out_train, expected_out_train,
                         msg="The checkpoint data for train is not the expected")
        self.assertEqual(wifi_out_test, expected_out_test,
                         msg="The checkpoint data for test is not the expected")

        wifi_checkpoint_0_columns = pd.read_csv(f"{path_checkpoint_train}/{wifi_out_train[0]}").columns.tolist()
        expected_columns = ["AppTimestamp(s)", "Name_SSID", "MAC_BSSID", "RSS", "Label", "Latitude", "Longitude"]

        self.assertEqual(wifi_checkpoint_0_columns, expected_columns,
                         msg="The columns for the checkpoint data for train is not the expected")

    def test_correctWifiFP(self):
        """
        This test checks if the Wi-Fi data from the checkpoint is correctly corrected (adapted to a Fingerprint format).
        """

        # read wifi data
        wifi_data = pd.read_csv(f"../{constants.data.train.CHECKPOINT_DATA_PATH}/Wifi/Wifi_label_0.csv")
        # correct wifi data
        corrected_wifi_data = preprocess.correctWifiFP(wifi_data, constants.T_MAX_SAMPLING,
                                                       constants.labels_dictionary_meters)

        # check if the columns are the expected
        expected_columns = ["AppTimestamp(s)"] + constants.aps + ["Latitude", "Longitude", "Label"]
        self.assertEqual(corrected_wifi_data.columns.tolist(), expected_columns,
                         msg="The columns for the corrected wifi data are not the expected")

        # check if we have an AppTimestamp for each second
        self.assertEqual(corrected_wifi_data["AppTimestamp(s)"].nunique(), constants.T_MAX_SAMPLING,
                         msg="The number of AppTimestamp(s) for the corrected wifi data is not the expected")

    def check_fix_na_wifi(self):
        """
        This test checks if the Wi-Fi data has correct the NA values.
        """

        # read wifi data and fix NA values
        wifi_data = preprocess.read_checkpoint(f"../{constants.data.train.CHECKPOINT_DATA_PATH}/Wifi",
                                               constants.labels_train)
        wifi_corrected = preprocess.correctWifiFP(wifi_data=wifi_data,
                                                  t_max_sampling=constants.T_MAX_SAMPLING,
                                                  dict_labels_to_meters=constants.labels_dictionary_meters)
        wifi_corrected = preprocess.fix_na_wifi(wifi_corrected)

        # check if there exists NA values
        self.assertFalse(wifi_corrected.isna().values.any(),
                         msg="There are NA values in the corrected wifi data")

    def test_scale_wifi(self):
        """
        This test checks if the Wi-Fi data is correctly scaled to the range of [0-1].
        """
        processed_wifi = pd.read_csv(f"../{constants.data.train.PROC_OUT_PATH}/processed_radiomap.csv")

        # check if the values are in the range of [0-1]
        self.assertTrue((processed_wifi[constants.aps] >= 0).all().all(),
                        msg="There are values lower than 0 in the processed wifi data")
        self.assertTrue((processed_wifi[constants.aps] <= 1).all().all(),
                        msg="There are values higher than 1 in the processed wifi data")

    def test_rolling_mean(self):
        """
        This test checks if the rolling mean is correctly applied to the Wi-Fi data.

        If we have 10 s of data, and we want to apply a rolling mean with the parameters n_max=10, window_size=5, step_size=5,
        it should return two windows of 5 s each one, and the AppTimestamp(s) should be [5, 10]

        """

        input_data = pd.DataFrame({
            "AppTimestamp(s)": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "GEOTECWIFI03": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "480Invitados": [3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4],
            "eduroam": [5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6],
            "wpen-uji": [9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10],
            "lt1iot": [11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12],
            "cuatroochenta": [13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14],
            "UJI": [15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16],
            "Latitude": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "Longitude": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "Label": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        })

        expected_output = pd.DataFrame({
            "AppTimestamp(s)": [5.0, 10.0],
            "GEOTECWIFI03": [1.0, 2.0],
            "480Invitados": [3.0, 4.0],
            "eduroam": [5.0, 6.0],
            "wpen-uji": [9.0, 10.0],
            "lt1iot": [11.0, 12.0],
            "cuatroochenta": [13.0, 14.0],
            "UJI": [15.0, 16.0],
            "Latitude": [0.0, 0.0],
            "Longitude": [0.0, 0.0],
            "Label": [0.0, 0.0]
        })

        result = preprocess.rolling_mean(input_data, window_size=5, step=5)
        print(result)
        self.assertTrue(result.equals(expected_output),
                        msg="The rolling mean is not correctly applied to the Wi-Fi data")


class TestDirectories(unittest.TestCase):
    """
    This class contains tests to verify the existence of necessary directories and files.
    """

    def test_resources(self):
        """
        This test checks if the necessary directories and files exist in the correct locations.
        """
        self.assertTrue(os.path.exists("../data"), "The data directory does not exist")
        self.assertTrue(os.path.exists("../data/train"), "The data/train directory does not exist")
        self.assertTrue(os.path.exists("../data/train/initial_rp_data"),
                        "The data/train/initial_rp_data directory does not exist")
        self.assertGreater(len(os.listdir("../data/train/initial_rp_data")), 0, "The train data is empty")
        self.assertTrue(os.path.exists("../data/test"), "The data/test directory does not exist")
        self.assertTrue(os.path.exists("../data/test/initial_rp_data"),
                        "The data/test/initial_rp_data directory does not exist")
        self.assertGreater(len(os.listdir("../data/test/initial_rp_data")), 0, "The test data is empty")


class TestProcessOutputs(unittest.TestCase):
    """
    This class contains tests to verify the outputs of the data processing steps.
    """

    def test_process_train(self):
        """
        This test checks if the processed training data is correctly saved in the expected directories.
        """

        self.assertTrue(os.path.exists(f"../{constants.outputs.PATH_OUTPUTS}"),
                        f"The {constants.outputs.PATH_OUTPUTS} directory does not exist")
        self.assertTrue(os.path.exists(f"../{constants.outputs.OUT_DATA}"),
                        f"The {constants.outputs.OUT_DATA} directory does not exist")
        self.assertTrue(os.path.exists(f"../{constants.outputs.TRAIN_OUT}"),
                        f"The {constants.outputs.TRAIN_OUT} directory does not exist")
        self.assertEqual(os.listdir(f"../{constants.outputs.TRAIN_OUT}"),
                         ["checkpoint_groundtruth", "processed_radiomap", "raw_radiomap"],
                         f"The {constants.outputs.TRAIN_OUT} dir should have checkpoint_groundtruth, processed_radiomap and raw_radiomap")
        raw_radiomap_train_labels = sorted(
            pd.read_csv(f"../{constants.data.train.RAW_OUT_PATH}/raw_radiomap.csv").Label.unique().astype(int).tolist())
        processed_radiomap_train_labels = sorted(
            pd.read_csv(f"../{constants.data.train.PROC_OUT_PATH}/processed_radiomap.csv").Label.unique().astype(
                int).tolist())
        expected = sorted([int(x.split(".")[0].split("_")[1])
                           for x in os.listdir(f"../{constants.data.train.INITIAL_DATA}")
                           if x.endswith(".txt")])
        # Check the labels from resource and from outputs are the same
        self.assertEqual(raw_radiomap_train_labels, expected,
                         msg="The labels from raw_radiomap and initial_rp_data are different")
        self.assertEqual(processed_radiomap_train_labels, expected,
                         msg="The labels from processed_radiomap and initial_rp_data are different")

    def test_process_test(self):
        """
        This test checks if the processed testing data is correctly saved in the expected directories.
        """

        self.assertTrue(os.path.exists(f"../{constants.outputs.TEST_OUT}"),
                        f"The {constants.outputs.TEST_OUT} directory does not exist")
        self.assertEqual(os.listdir(f"../{constants.outputs.TEST_OUT}"),
                         ["checkpoint_groundtruth", "processed_radiomap", "raw_radiomap"],
                         f"The {constants.outputs.TEST_OUT} dir should have checkpoint_groundtruth, processed_radiomap and raw_radiomap")
        raw_radiomap_test_labels = sorted(
            pd.read_csv(f"../{constants.data.test.RAW_OUT_PATH}/raw_radiomap.csv").Label.unique().astype(int).tolist())
        processed_radiomap_test_labels = sorted(
            pd.read_csv(f"../{constants.data.test.PROC_OUT_PATH}/processed_radiomap.csv")
            .Label.unique().astype(int).tolist())
        expected = sorted([int(x.split(".")[0].split("_")[1])
                           for x in os.listdir(f"../{constants.data.test.INITIAL_DATA}")
                           if x.endswith(".txt")])
        # Check the labels from resource and from outputs are the same
        self.assertEqual(raw_radiomap_test_labels, expected,
                         msg="The labels from raw_radiomap and initial_rp_data are different")
        self.assertEqual(processed_radiomap_test_labels, expected,
                         msg="The labels from processed_radiomap and initial_rp_data are different")

    def test_process_partitions_dir(self):
        """
        This test checks if the data partitions are correctly saved in the expected directories.
        """

        # Check all the partitions directory exists
        self.assertTrue(os.path.exists(f"../{constants.outputs.PARTITIONS}"),
                        msg="The partitions directory does not exist"),
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_5VS18}"),
                        msg=f"{constants.data.partitions.PARTITION_5VS18} does not exist")

        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_10VS13}"),
                        msg=f"{constants.data.partitions.PARTITION_10VS13} does not exist")

        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_15VS8}"),
                        msg=f"{constants.data.partitions.PARTITION_15VS8} does not exist")

    def test_partition_5vs18(self):
        """
        This test checks if the partition 5vs18 is correctly saved in the expected directories.
        """

        # Check the partition 5vs18 is correctly saved
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_5VS18}/train"),
                        msg=f"{constants.data.partitions.PARTITION_5VS18}/train does not exist")
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_5VS18}/train/raw"),
                        msg=f"{constants.data.partitions.PARTITION_5VS18}/train/raw does not exist")
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_5VS18}/train/processed"),
                        msg=f"{constants.data.partitions.PARTITION_5VS18}/train/processed does not exist")
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_5VS18}/test"),
                        msg=f"{constants.data.partitions.PARTITION_5VS18}/test does not exist")
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_5VS18}/test/raw"),
                        msg=f"{constants.data.partitions.PARTITION_5VS18}/test/raw does not exist")
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_5VS18}/test/processed"),
                        msg=f"{constants.data.partitions.PARTITION_5VS18}/test/processed does not exist")

        # Check the labels from constants and from outputs are the same
        raw_radiomap_train_labels = sorted(
            pd.read_csv(
                f"../{constants.data.partitions.PARTITION_5VS18}/train/raw/raw_radiomap.csv").Label.unique().astype(
                int).tolist())
        processed_radiomap_train_labels = sorted(
            pd.read_csv(
                f"../{constants.data.partitions.PARTITION_5VS18}/train/processed/processed_radiomap.csv").Label.unique().astype(
                int).tolist())
        raw_radiomap_test_labels = sorted(
            pd.read_csv(
                f"../{constants.data.partitions.PARTITION_5VS18}/test/raw/raw_radiomap.csv").Label.unique().astype(
                int).tolist())
        processed_radiomap_test_labels = sorted(
            pd.read_csv(f"../{constants.data.partitions.PARTITION_5VS18}/test/processed/processed_radiomap.csv")
            .Label.unique().astype(int).tolist())

        expected_train = sorted(constants.labels_partition_5vs18)
        expected_test = sorted([x for x in constants.labels_train if x not in expected_train])

        self.assertEqual(raw_radiomap_train_labels, expected_train,
                         msg="The labels from raw_radiomap on partition 5vs18 train and expected labels are different")
        self.assertEqual(processed_radiomap_train_labels, expected_train,
                         msg="The labels from processed_radiomap on partition 5vs18 train and expected labels are different")
        self.assertEqual(raw_radiomap_test_labels, expected_test,
                         msg="The labels from raw_radiomap on partition 5vs18 test and expected labels are different")
        self.assertEqual(processed_radiomap_test_labels, expected_test,
                         msg="The labels from processed_radiomap on partition 5vs18 test and expected labels are different")

    def test_partition_10vs13(self):
        """
        This test checks if the partition 10vs13 is correctly saved in the expected directories.
        """

        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_10VS13}/train"),
                        msg=f"{constants.data.partitions.PARTITION_10VS13}/train does not exist")
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_10VS13}/train/raw"),
                        msg=f"{constants.data.partitions.PARTITION_10VS13}/train/raw does not exist")
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_10VS13}/train/processed"),
                        msg=f"{constants.data.partitions.PARTITION_10VS13}/train/processed does not exist")
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_10VS13}/test"),
                        msg=f"{constants.data.partitions.PARTITION_10VS13}/test does not exist")
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_10VS13}/test/raw"),
                        msg=f"{constants.data.partitions.PARTITION_10VS13}/test/raw does not exist")
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_10VS13}/test/processed"),
                        msg=f"{constants.data.partitions.PARTITION_10VS13}/test/processed does not exist")

        # Check the labels from constants and from outputs are the same
        raw_radiomap_train_labels = sorted(
            pd.read_csv(
                f"../{constants.data.partitions.PARTITION_10VS13}/train/raw/raw_radiomap.csv").Label.unique().astype(
                int).tolist())

        processed_radiomap_train_labels = sorted(
            pd.read_csv(
                f"../{constants.data.partitions.PARTITION_10VS13}/train/processed/processed_radiomap.csv").Label.unique().astype(
                int).tolist())

        raw_radiomap_test_labels = sorted(
            pd.read_csv(
                f"../{constants.data.partitions.PARTITION_10VS13}/test/raw/raw_radiomap.csv").Label.unique().astype(
                int).tolist())

        processed_radiomap_test_labels = sorted(
            pd.read_csv(
                f"../{constants.data.partitions.PARTITION_10VS13}/test/processed/processed_radiomap.csv").Label.unique().astype(
                int).tolist())

        expected_train = sorted(constants.labels_partition_10vs13)
        expected_test = sorted([x for x in constants.labels_train if x not in expected_train])

        self.assertEqual(raw_radiomap_train_labels, expected_train,
                         msg="The labels from raw_radiomap on partition 10vs13 train and expected labels are different")
        self.assertEqual(processed_radiomap_train_labels, expected_train,
                         msg="The labels from processed_radiomap on partition 10vs13 train and expected labels are different")
        self.assertEqual(raw_radiomap_test_labels, expected_test,
                         msg="The labels from raw_radiomap on partition 10vs13 test and expected labels are different")
        self.assertEqual(processed_radiomap_test_labels, expected_test,
                         msg="The labels from processed_radiomap on partition 10vs13 test and expected labels are different")

    def test_partition_15vs8(self):
        """
        This test checks if the partition 15vs8 is correctly saved in the expected directories.
        """

        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_15VS8}/train"),
                        msg=f"{constants.data.partitions.PARTITION_15VS8}/train does not exist")
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_15VS8}/train/raw"),
                        msg=f"{constants.data.partitions.PARTITION_15VS8}/train/raw does not exist")
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_15VS8}/train/processed"),
                        msg=f"{constants.data.partitions.PARTITION_15VS8}/train/processed does not exist")
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_15VS8}/test"),
                        msg=f"{constants.data.partitions.PARTITION_15VS8}/test does not exist")
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_15VS8}/test/raw"),
                        msg=f"{constants.data.partitions.PARTITION_15VS8}/test/raw does not exist")
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_15VS8}/test/processed"),
                        msg=f"{constants.data.partitions.PARTITION_15VS8}/test/processed does not exist")

        # Check the labels from constants and from outputs are the same
        raw_radiomap_train_labels = sorted(
            pd.read_csv(
                f"../{constants.data.partitions.PARTITION_15VS8}/train/raw/raw_radiomap.csv").Label.unique().astype(
                int).tolist())
        processed_radiomap_train_labels = sorted(
            pd.read_csv(
                f"../{constants.data.partitions.PARTITION_15VS8}/train/processed/processed_radiomap.csv").Label.unique().astype(
                int).tolist())
        raw_radiomap_test_labels = sorted(
            pd.read_csv(
                f"../{constants.data.partitions.PARTITION_15VS8}/test/raw/raw_radiomap.csv").Label.unique().astype(
                int).tolist())
        processed_radiomap_test_labels = sorted(
            pd.read_csv(f"../{constants.data.partitions.PARTITION_15VS8}/test/processed/processed_radiomap.csv")
            .Label.unique().astype(int).tolist())

        expected_train = sorted(constants.labels_partition_15vs8)
        expected_test = sorted([x for x in constants.labels_train if x not in expected_train])

        self.assertEqual(raw_radiomap_train_labels, expected_train,
                         msg="The labels from raw_radiomap on partition 15vs8 train and expected labels are different")

        self.assertEqual(processed_radiomap_train_labels, expected_train,
                         msg="The labels from processed_radiomap on partition 15vs8 train and expected labels are different")

        self.assertEqual(raw_radiomap_test_labels, expected_test,
                         msg="The labels from raw_radiomap on partition 15vs8 test and expected labels are different")

        self.assertEqual(processed_radiomap_test_labels, expected_test,
                         msg="The labels from processed_radiomap on partition 15vs8 test and expected labels are different")


if __name__ == '__main__':
    unittest.main()
