import sys
from _3_deep_feature_extract.fmcib.datasets.dataSetAug import MyDatasetAug
from _3_deep_feature_extract.fmcib.models import fmcib_model
import argparse
import time
import os
import numpy as np
import pandas as pd
from warmup_scheduler import GradualWarmupScheduler
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.transforms import (
    Compose,
    RandFlip
)


def deep_feat_extractor(dataloader, model, device, mode, save_path):
    print(mode+' cohort_'+key)
    model.eval()
    test_batch_names = []
    test_batch_label = []
    deep_feature = np.zeros((1, 256), dtype=np.float32)

    test_bar = tqdm(dataloader, file=sys.stdout)
    with torch.no_grad():
        for step, data in enumerate(test_bar):
            test_tokens, test_targets, targets_path = data
            test_tokens = test_tokens.to(device)
            outputs, heads_out = model(test_tokens)

            test_batch_names.append(targets_path)
            test_batch_label.append(test_targets)
            deep_feature = np.vstack((deep_feature, heads_out.detach().cpu().numpy()))

        deep_features = deep_feature[1:, :]
        ct_names = np.array(test_batch_names).squeeze()
        feature_name = ["CT_Name"] + [f"Feature{i}" for i in range(256)]
        features = np.hstack((ct_names.reshape(-1, 1), deep_features))
        datasets = pd.DataFrame(features, columns=feature_name)
        datasets.to_csv(save_path, index=False)


def read_and_merge_data(label_file, file_path, patient_id_col='', label_col='', img_path_col=''):

    label_data = pd.read_csv(label_file)[[patient_id_col, label_col]]
    label_data[img_path_col] = label_data[patient_id_col].apply(lambda x: f"{file_path + str(x)}.nii.gz")

    return label_data


def get_parser():
    parser = argparse.ArgumentParser(
        description='Model')
    parser.add_argument('--epochs', default=1,
                        type=int, help='number of epochs')
    parser.add_argument('--loss', type=str, default='BCEW')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', default=8e-7, type=float,
                        help='initial learning rate (default: 0.05)')
    parser.add_argument('--momentum', default=0.7, type=float)
    parser.add_argument('--weight_decay', default=1e-7, type=float)
    parser.add_argument('--Tmax', default=60, type=float)
    parser.add_argument('--eta_min', default=0, type=float)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--num_worker', default=0, type=int)
    parser.add_argument('--warmup_epoch', default=10, type=int)
    parser.add_argument('--finetune', default=False, type=bool)
    parser.add_argument('--subtrahend', default=-125, type=int)
    parser.add_argument('--divisor', default=350, type=int)
    parser.add_argument('--winMin', default=-125, type=int)
    parser.add_argument('--winMan', default=225, type=int)
    parser.add_argument('--aug', default=True, type=bool)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--act_function', default='GELU', type=str)

    return parser.parse_args()



if __name__ == '__main__':
    args = get_parser()
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    if args.aug:
        transform = Compose(
            [
                RandFlip(prob=1, spatial_axis=0)
            ]
        )
    else:
        transform = None

    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


    train_cohort_name = ''
    label_path = ''
    datasets_val = ['']
    label_path_val = ['']
    test_dataloader = {}
    pid_col_name = ''
    label_col_name = ''
    img_path_col = ''
    data_file_name_prefix = ''
    img_file_name = f''

    # 队列划分
    rt = pd.read_csv(r'')
    rt_cohort = {'': rt}

    train_all = read_and_merge_data(f'', img_file_name)
    train_test = []
    if train_cohort_name == '':
        train_test = pd.merge(train_all, rt, how='inner', on=pid_col_name)


    X_train, X_val, y_train, y_val = train_test_split(train_test[img_path_col].values,
                                                      train_test[label_col_name].values, test_size=0.2,
                                                      stratify=train_test[label_col_name].values)

    train_dataset = MyDatasetAug(X_train, y_train, subtrahend=args.subtrahend, divisor=args.divisor, winMin=args.winMin, winMax=args.winMan)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=False, num_workers=args.num_worker, drop_last=False)

    val_dataset = MyDatasetAug(X_val, y_val, subtrahend=args.subtrahend, divisor=args.divisor, winMin=args.winMin, winMax=args.winMan)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=args.num_worker, drop_last=False)

    for label_path, cohort_name in zip(label_path_val, datasets_val):
        if cohort_name == '':
            test_all = read_and_merge_data(f'', f'')
            test_data = MyDatasetAug(test_all[img_path_col], test_all[label_col_name], subtrahend=args.subtrahend, divisor=args.divisor, winMin=args.winMin, winMax=args.winMan)
            test_dataloader[''] = torch.utils.data.DataLoader(test_data, batch_size=1,
                                                               shuffle=False, num_workers=args.num_worker,
                                                               drop_last=False)

    eval_path = r''
    model = fmcib_model(eval_mode=True, heads=[4096, 2048, 256, 1], actFunction=args.act_function, eval_path=eval_path, device_num=args.device).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_negative_samples = np.count_nonzero(train_test[label_col_name].values == 0)
    total_positive_samples = np.count_nonzero(train_test[label_col_name].values == 1)
    logit_pos_weight = torch.tensor([total_negative_samples / total_positive_samples]).to(device)
    criterion = ''
    if args.loss == 'BCEW':
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=logit_pos_weight)
        print('USing BCEWithLogitsLoss')
    elif args.loss == 'BCE':
        criterion = torch.nn.BCELoss()
        print('USing BCELoss')

    # Feature extract
    hp = {''}
    scheduler1 = CosineAnnealingLR(
        optimizer, T_max=args.Tmax, eta_min=args.eta_min)
    scheduler2 = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=args.warmup_epoch, after_scheduler=scheduler1)

    for epoch in range(args.epochs):
        deep_feat_extractor(train_dataloader, model, device, 'val', f'')
        deep_feat_extractor(val_dataloader, model, device, 'val', f'')
        for key in test_dataloader.keys():
            deep_feat_extractor(test_dataloader[key], model, device, 'test', f'')


