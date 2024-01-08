import unittest
import sys
import os

from src.process_train import processTrain
from src.process_test import processTest
from src.process_partitions import processPartitions
from src.positioning_partitions import getPositioningWithPartitions
from src.utils import preprocess
from src.utils import constants


class TestDirectories(unittest.TestCase):

    def test_resources(self):
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

    def test_process_train(self):
        processTrain()
        self.assertTrue(os.path.exists("data"), "The data directory does not exist")
        self.assertTrue(os.path.exists("data/train"), "The data/train directory does not exist")
        self.assertEqual(os.listdir("data/train"), ["checkpoint_groundtruth", "processed_radiomap", "raw_radiomap"],
                         "The data/train should have checkpoint_groundtruth, processed_radiomap and raw_radiomap")


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
