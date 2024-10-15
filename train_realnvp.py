import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from realnvp import RealNVP
from Progress import Progress
from sys import argv

class EmbeddingDataset(Dataset):
    def __init__(self, embed_dir):
        self.embed_dir = embed_dir
        self.embed_files = [x for x in os.listdir(embed_dir) if os.path.isfile(os.path.join(embed_dir, x))]

    def __len__(self):
        return len(self.embed_files)

    def __getitem__(self, i):
        embed_path = os.path.join(self.embed_dir, self.embed_files[i])
        x = np.fromfile(embed_path, dtype=np.float32)
        return torch.from_numpy(x)

def train_loop(nvp, data, opt, num_epochs, device):
    for epoch in range(num_epochs):
        print('Epoch:', epoch+1)
        metrics = {
            'loss': 0.0,
            'log_Px': 0.0,
        }
        for x in Progress(data, metrics):
            log_Px, log_Pu, log_det = nvp(x.to(device))
            log_Px = torch.mean(log_Px)
            log_Pu = torch.mean(log_Pu)
            log_det = torch.mean(log_det)

            loss = -log_Pu - log_det

            opt.zero_grad()
            loss.backward()
            opt.step()

            metrics['loss'] = 0.9 * metrics['loss'] + 0.1 * loss.item()
            metrics['log_Px'] = 0.9 * metrics['log_Px'] + 0.1 * log_Px.item()

def main():

    if len(argv) <= 1:
        print(f'Usage: {argv[0]} TRAIN_FEATURES_DIRECTORY')
        print('')
        print('This script will load features from TRAIN_FEATURES_DIRECTORY, and train a Normalizing Flow')
        print('The features are assumed to be extracted from Faster-RCNN, and have a dimension of 1024')
        print('Finally, this script will store the trained Flow in frcnn_nvp.pt')
        return

    train_dir = argv[1]
    learn_rate = 1e-4
    l2 = 1e-5

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    nvp = RealNVP(1024, 512, 5).to(device)

    opt = torch.optim.Adam(nvp.parameters(), lr=learn_rate, weight_decay=l2)

    dataset = EmbeddingDataset(train_dir)

    dl128 = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    train_loop(nvp, dl128, opt, 100, device)

    dl256 = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    train_loop(nvp, dl256, opt, 50, device)

    torch.save(nvp.state_dict(), 'frcnn_nvp.pt')

if __name__ == '__main__': main()
