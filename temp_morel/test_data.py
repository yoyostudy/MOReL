import torch

from data_helper import *

class TestDevice:

    def test_cuda(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        assert device == "cuda:0", "GPU not available"

class TestData:

    def test_filePath(self):
        load_data(data_path="ww-high-1000-train.npz")

    def test_dataset(self):
        data_path="ww-high-1000-train.npz"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dataset = OfflineRLDataset(data_path=data_path, device=device)
        print("sample_size", dataset.get_sample_size())
        print("action_dim", dataset.get_action_dim())
        print("obs_dim", dataset.get_obs_dim())    

    def test_1(self):
        assert 1 == 1


