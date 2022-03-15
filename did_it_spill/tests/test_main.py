import unittest
from torch.utils.data import DataLoader, Dataset
import sys
from pathlib import Path

#### needed for testing #####
this_path = str(Path(__file__).parent.parent.parent)
sys.path.append(this_path)
#############################

import did_it_spill


class DatasetForTesting(Dataset):

    def __init__(self, start=0, end=10):
        assert start >= 0
        assert end >= 0
        assert end >= start

        self.x_data = list(range(start, start + end))
        self.y_data = list(range(start, start + end))

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


class TestMain(unittest.TestCase):

    __expected_length_of_overlap: int = 0
    __expected_indexes_in_overlap: set = set()

    def test_overlap(self):

        train_dataset = DatasetForTesting(start=0, end=10)
        test_dataset = DatasetForTesting(start=0, end=5)

        train_dataloader = self.__create_dataloader(train_dataset)
        test_dataloader = self.__create_dataloader(test_dataset)

        self.__expected_length_of_overlap = 5
        self.__expected_indexes_in_overlap = {0, 1, 2, 3, 4}
        spills = did_it_spill.check_spill(train_dataloader, test_dataloader)

        self.__check_expected_values(spills)

    def test_no_overlap(self):
        train_dataset = DatasetForTesting(start=0, end=5)
        test_dataset = DatasetForTesting(start=10, end=15)

        train_dataloader = self.__create_dataloader(train_dataset)
        test_dataloader = self.__create_dataloader(test_dataset)

        self.__expected_length_of_overlap = 0
        self.__expected_indexes_in_overlap = set()
        spills = did_it_spill.check_spill(train_dataloader, test_dataloader)

        self.__check_expected_values(spills)

    @staticmethod
    def __create_dataloader(dataset):
        return DataLoader(dataset=dataset, shuffle=False, batch_size=1)

    def __check_expected_values(self, spills):
        self.assertEqual(self.__expected_length_of_overlap, len(spills))
        for spill in spills:
            self.assertTrue(spill[0] in self.__expected_indexes_in_overlap)
            self.assertTrue(spill[1] in self.__expected_indexes_in_overlap)
            self.assertTrue(spill[0] == spill[1])

