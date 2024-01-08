from src.process_train import processTrain
from src.process_test import processTest
from src.process_partitions import processPartitions
from src.positioning_partitions import getPositioningWithPartitions
from src.utils.constants import constants
import os

if __name__ == "__main__":
    os.makedirs(constants.outputs.PATH_OUTPUTS, exist_ok=True)
    os.makedirs(constants.outputs.OUT_DATA, exist_ok=True)
    os.makedirs(constants.outputs.TRAIN_OUT, exist_ok=True)
    os.makedirs(constants.outputs.TEST_OUT, exist_ok=True)
    os.makedirs(constants.outputs.PARTITIONS, exist_ok=True)
    os.makedirs(constants.outputs.POSITIONING_PARTITIONS, exist_ok=True)

    processTrain()
    processTest()
    processPartitions()
    getPositioningWithPartitions()