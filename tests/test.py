import unittest
import os
import pandas as pd

from src.utils import preprocess
from src.utils.constants import constants


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

    def test_process_partitions(self):
        """
        This test checks if the data partitions are correctly saved in the expected directories.
        """

        # Check all the partitions directory exists
        self.assertTrue(os.path.exists(f"../{constants.outputs.PARTITIONS}"),
                        msg="The partitions directory does not exist"),
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_5VS18}"),
                        msg=f"{constants.data.partitions.PARTITION_5VS18} does not exist")
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
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_10VS13}"),
                        msg=f"{constants.data.partitions.PARTITION_10VS13} does not exist")
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
        self.assertTrue(os.path.exists(f"../{constants.data.partitions.PARTITION_15VS8}"),
                        msg=f"{constants.data.partitions.PARTITION_15VS8} does not exist")
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


# class TestOutputsDirectories(unittest.TestCase):
#
#     def test_directories(self):
#         self.assertTrue(os.path.exists("output"))
#         self.assertTrue(os.path.exists("output/data"))
#         self.assertTrue(os.path.exists("output/data/train"))
#         self.assertTrue(os.path.exists("output/data/test"))
#         self.assertTrue(os.path.exists("output/data/partitions"))
#         self.assertTrue(os.path.exists("output/positioning_partitions"))


class TestUtilsSource(unittest.TestCase):

    def test_parse_windows(self):
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


if __name__ == '__main__':
    unittest.main()
