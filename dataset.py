import torch.utils.data as data
import torch
import numpy as np

class DatasetFromList(data.Dataset):

    def __init__(self, file_path, flag=None):
        super(DatasetFromList, self).__init__()
        self.flag = flag
        with open(file_path) as f:
            self.list = [line.strip() for line in f]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        path, index, label = self.list[idx].split(' ')
        data = np.load('luna_' + self.flag + '_features/chunk_'+path + '.npy')[int(index),:,:,:]
        data  = np.expand_dims(data, axis=0)
        label = np.float64(label)
        return torch.from_numpy(data).float(), label.astype(torch.LongTensor)
