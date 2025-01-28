from torch.utils.data import Dataset, DataLoader, Sampler


class MyDatasetFusion(Dataset):
    def __init__(self, pid, deep_features, radio_features, labels):

        self.pid = pid
        self.deep_features = deep_features
        self.radio_features = radio_features
        self.labels = labels

        print('init')

    def __getitem__(self, idx):

        pid = self.pid[idx]

        deep_feature = self.deep_features.loc[self.deep_features[''] == str(pid)].drop(columns=['']).values[0]
        radio_feature = self.radio_features.loc[self.radio_features[''] == str(pid)].drop(columns=['']).values[0]
        label = self.labels.loc[self.radio_features[''] == str(pid)].drop(columns=['']).values[0]

        return {'pid':pid, 'deep_f':deep_feature, 'radio_f':radio_feature, 'label':label}

    def __len__(self):
        return len(self.pid)
