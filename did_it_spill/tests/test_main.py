from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import unittest
import sys
import torchvision
import torchvision.transforms as transforms

#### needed for testing #####
this_path = str(Path(__file__).parent.parent.parent)
sys.path.append(this_path)
#############################

import did_it_spill


class IntegerDatasetForTesting(Dataset):

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


class CharacterDatasetForTesting(Dataset):

    data = list()

    def __init__(self, size=1):
        assert size > 0
        self.__generate_data(size=size)

    def __generate_data(self, size):
        resulting_data_list = list()

        for i in range(size):
            i = i % 26
            char_i = i + 97 # i=0 this will be an "a"
            amount_of_char = i // 26 # when size is greater than 26 this dataset will no longer be unique
            string_char_representation = str(chr(char_i))*(amount_of_char+1)
            resulting_data_list.append(string_char_representation)
        self.data = resulting_data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], 1


class TestMain(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super(TestMain, cls).setUpClass()
        # taken from pytorch tutorial for classification: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        transform = transforms.ToTensor()
        cls.cifar10_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        cls.cifar10_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)

    def test_overlap_with_batchsize_1(self):
        self.__test_overlap()

    def test_no_overlap_with_batchsize_1(self):
        self.__test_no_overlap()

    def test_overlap_with_batchsize_2(self):
        self.__test_overlap(batch_size=2)

    def test_no_overlap_with_batchsize_2(self):
        self.__test_no_overlap(batch_size=2)

    def test_overlap_stringdata_with_batchsize_1(self):
        train_dataset = CharacterDatasetForTesting(size=10)
        test_dataset = CharacterDatasetForTesting(size=2)

        train_dataloader = self.__create_dataloader(train_dataset, batch_size=1, shuffle=False)
        test_dataloader = self.__create_dataloader(test_dataset, batch_size=1, shuffle=False)

        # raised since the dataloader will not pack string or chars into a tensor
        self.assertRaises(AttributeError, did_it_spill.check_spill, train_dataloader, test_dataloader)

    def test_cifar10_no_spills(self):
        train_dataloader = self.__create_dataloader(self.cifar10_train)
        test_dataloader = self.__create_dataloader(self.cifar10_test)

        spills = did_it_spill.check_spill(train_dataloader, test_dataloader)

        expected_indexes_in_overlap = set()

        self.__check_expected_values(spills, expected_indexes_in_overlap)

    def test_cifar10_full_spills(self):
        test_dataloader = self.__create_dataloader(self.cifar10_test)

        spills = did_it_spill.check_spill(test_dataloader, test_dataloader)

        #since we compare the identical dataloaders all entries should overlap
        expected_indexes_in_overlap = set(range(len(test_dataloader)))

        self.__check_expected_values(spills, expected_indexes_in_overlap)

    def __test_overlap(self, batch_size=1, shuffle=False):
        train_dataset = IntegerDatasetForTesting(start=0, end=10)
        test_dataset = IntegerDatasetForTesting(start=0, end=5)

        train_dataloader = self.__create_dataloader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = self.__create_dataloader(test_dataset, batch_size=batch_size, shuffle=shuffle)

        expected_indexes_in_overlap = {0, 1, 2, 3, 4}
        spills = did_it_spill.check_spill(train_dataloader, test_dataloader)

        self.__check_expected_values(spills, expected_indexes_in_overlap)

    def __test_no_overlap(self, batch_size=1, shuffle=False):
        train_dataset = IntegerDatasetForTesting(start=0, end=5)
        test_dataset = IntegerDatasetForTesting(start=10, end=15)

        train_dataloader = self.__create_dataloader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = self.__create_dataloader(test_dataset, batch_size=batch_size, shuffle=shuffle)

        expected_indexes_in_overlap = set()
        spills = did_it_spill.check_spill(train_dataloader, test_dataloader)

        self.__check_expected_values(spills, expected_indexes_in_overlap)

    @staticmethod
    def __create_dataloader(dataset, batch_size=1, shuffle=False):
        return DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size)

    def __check_expected_values(self, spills, expected_indexes_in_overlap):
        seen = dict()
        for spill in spills:
            self.assertTrue(spill[0] in expected_indexes_in_overlap)
            self.assertTrue(spill[1] in expected_indexes_in_overlap)
            self.assertTrue(spill[0] == spill[1])
            seen[spill[0]] = True

        self.assertTrue(len(seen) == len(spills))


