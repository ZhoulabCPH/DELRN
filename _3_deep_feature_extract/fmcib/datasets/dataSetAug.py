from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import SimpleITK as sitk



class MyDatasetAug(Dataset):
    def __init__(self, imgs_path, labels, subtrahend, divisor, winMin, winMax, transform=None):
        assert len(imgs_path) == labels.shape[0]

        self.subtrahend = subtrahend
        self.divisor = divisor

        self.imgs_path = imgs_path
        self.labels = labels

        self.transform = transform

        self.imgs_data = [sitk.GetArrayFromImage(sitk.IntensityWindowing(sitk.ReadImage(img_path),
                                                   windowMinimum=winMin, windowMaximum=winMax,
                                                   outputMinimum=winMin, outputMaximum=winMax))
                           for img_path in self.imgs_path]

        print('init')

    def __getitem__(self, idx):
        data = self.imgs_data[idx]

        if self.subtrahend != 0 and self.divisor != 0:
            data = (data - self.subtrahend) / self.divisor
        if self.transform is not None:
            data = self.transform(data)

        data = np.expand_dims(data, axis=3)
        label = int(self.labels[idx])
        targets_path = self.imgs_path[idx]
        return data, label, targets_path

    def __len__(self):
        return len(self.imgs_path)
