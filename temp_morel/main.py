from morel import Morel
from test_data import *
from data_helper import OfflineRLDataset
from torch.utils.data import DataLoader, Dataset

if __name__ == "__main__":
    #morel.train()
    #morel.eval()

    data_path="ww-high-1000-train.npz"
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    dataset = OfflineRLDataset(data_path=data_path, device=device)

    obs_dim = dataset.get_obs_dim()
    action_dim = dataset.get_action_dim()

    dataloader = DataLoader(dataset, batch_size = 128, shuffle=True)

    # for item in dataloader:
    #     print(type(item))
    #     print(len(item))
    #     print(item['obs'])

    morel_agent = Morel(obs_dim=obs_dim, action_dim=action_dim)

    morel_agent.train(dataloader=dataloader, dynamics_data=dataset)

    morel_agent.eval()


    # val_data_path = 'ww-high-100-val.npz'
    # val_dataset = OfflineRLDataset(data_path=val_data_path, device=device)
    # val_dataloader = DataLoader(val_dataset, batch_size = 128, shuffle = True)
    # morel_agent.eval(val_dataset)




