from src.process_train import processTrain
from src.process_test import processTest
from src.process_partitions import processPartitions
from src.positioning_partitions import getPositioningWithPartitions
import os

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/data", exist_ok=True)
    os.makedirs("output/data/train", exist_ok=True)
    os.makedirs("output/data/test", exist_ok=True)
    os.makedirs("output/data/partitions", exist_ok=True)
    os.makedirs("output/positioning_partitions", exist_ok=True)

    processTrain()
    processTest()
    processPartitions()
    getPositioningWithPartitions()