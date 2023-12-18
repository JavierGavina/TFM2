from src.process_train import processTrain
from src.process_test import processTest
from src.process_partitions import processPartitions
from src.positioning_partitions import getPositioningWithPartitions

if __name__ == "__main__":
    processTrain()
    processTest()
    processPartitions()
    getPositioningWithPartitions()