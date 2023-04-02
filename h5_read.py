import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

def load_data(file_path):
    all_data = []
    all_label = []
    with h5py.File(file_path, "r") as f:
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud



class H5Dataset(Dataset):
    def __init__(self, file_path, num_points, partition = "train"):
        self.data, self.label = load_data(file_path)
        self.num_points = num_points
        self.partition = partition
        
    
    def __getitem__(self, item): 
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label
        

    def __len__(self):
        return self.data.shape[0]
        


# torch.from_numpy(self.index[index,:]).float()
# torch.from_numpy(np.array(self.labels_values[index])),
# torch.from_numpy(np.array(self.index_values[index]))

 # with h5py.File(file_path, "r") as f:
        #     # group = f.create_group('a_group')
        #     # group.create_dataset(name='matrix', data=np.zeros((10, 10)), chunks=True, compression='gzip')
        #     # #h5_file = h5py.File(file_path , 'r')