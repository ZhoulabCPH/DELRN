import sys
from model_fusion.datasets.dataSetFusion import MyDatasetFusion
import argparse
import time
import os
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from model_fusion.ct_mcvae import tools
from sklearn.metrics import roc_curve, auc, confusion_matrix
from _5_feature_fusion.model.model_mcvae import mcVAE_BRCA_mut


def get_cm(AllLabels, AllValues):
    Auc = 0
    m = t = 0
    if len(AllValues) > 10:
        fpr, tpr, threshold = roc_curve(AllLabels, AllValues, pos_label=1)
        Auc = auc(fpr, tpr)

        for i in range(len(threshold)):
            if tpr[i] - fpr[i] > m:
                m = abs(-fpr[i] + tpr[i])
                t = threshold[i]

    AllPred = [int(i >= t) for i in AllValues]
    Acc = sum([AllLabels[i] == AllPred[i]
               for i in range(len(AllPred))]) / len(AllPred)

    Pos_num = sum(AllLabels)
    Neg_num = len(AllLabels) - Pos_num
    cm = confusion_matrix(AllLabels, AllPred)

    Pos_rate = 0
    Neg_rate = 0
    if Pos_num != 0 and Neg_num != 0:
        Pos_rate = cm[1][1] / Pos_num * 100
        Neg_rate = cm[0][0] / Neg_num * 100

    return Auc, Acc, t, Neg_rate, Pos_rate, len(AllLabels), Pos_num, Neg_num


def train(dataloader, model, criterion, optimizer, epoch):
    model.train()
    loss_list = []  # 每次训练的loss 用于展示数据
    avg_loss = 0
    preds = np.array([])  # 预测值 prediction -s 总数
    trues = np.empty(shape=[0])  # 真值,总数
    AUC = 0

    # 每次都给一个batch
    bar = tqdm(dataloader)  # 进度条显示数据
    for step, data in enumerate(bar):

        batch_preds = np.array([])  # 每个batch训练的预测值
        batch_trues = np.empty(shape=[0])  # 每次batch训练的真值

        pid = data['pid']
        deep_f = data['deep_f']
        radio_f = data['radio_f']
        targets = data['label']

        deep_f = deep_f.cuda(args.device).float()
        radio_f = radio_f.cuda(args.device).float()
        targets = targets.cuda(args.device).float()

        optimizer.zero_grad()  # 优化器清零
        outputs = model(radio_f, deep_f)  # 得到预测值
        loss = criterion(outputs, targets.float().squeeze())  # 使用损失函数进行比对
        loss.backward()  # 反向传播
        optimizer.step()  # 使用优化器

        batch_preds = np.append(batch_preds, outputs.detach().cpu().numpy())  # 统计每个batch的值
        batch_trues = np.concatenate((batch_trues, targets.detach().cpu().numpy().squeeze()), axis=0)
        preds = np.append(preds, batch_preds)  # 转换为数组的形式，并统计总预测值
        trues = np.concatenate((trues, batch_trues), axis=0)  # 转换为数组的形式，并统计总真值

        loss_list.append(loss.cpu().detach().numpy())
        AUC, sklearn_accuracy, threshold, _, _, _, _, _ = get_cm(trues, preds)
        avg_loss = np.average(np.array(loss_list))

        bar.desc = "[train__eppch__bar] Epoch:{} loss:{:.4f} accuracy:{:.4f} AUC:{:.4f} threshold:{:.4f}".format(
            epoch + 1, avg_loss, sklearn_accuracy, AUC, threshold)



    return avg_loss, AUC


def validate(dataloader, model, criterion, device, mode, epoch, key):
    print(mode+' cohort_'+key)
    model.eval()
    loss_list = []  # 每次训练的loss 用于展示数据
    avg_loss = 0
    preds = np.array([])  # 预测值 prediction -s 总数
    trues = np.empty(shape=[0])  # 真值,总数
    AUC = 0

    bar = tqdm(dataloader, file=sys.stdout)  # 进度条显示数据
    with torch.no_grad():  # 这句话就将这里面的语句不去关注梯度信息
        for step, data in enumerate(bar):

            batch_preds = np.array([])  # 每个batch训练的预测值
            batch_trues = np.empty(shape=[0])  # 每次batch训练的真值

            # pid = data['pid']
            deep_f = data['deep_f']
            radio_f = data['radio_f']
            targets = data['label']

            deep_f = deep_f.cuda(args.device).float()
            radio_f = radio_f.cuda(args.device).float()
            targets = targets.cuda(args.device).float()

            optimizer.zero_grad()  # 优化器清零
            outputs = model(radio_f, deep_f)  # 得到预测值

            loss = criterion(outputs, targets.float().squeeze(1))  # 使用损失函数进行比对
            loss_list.append(loss.cpu().detach().numpy())
            avg_loss = np.average(np.array(loss_list))

            batch_preds = np.append(batch_preds, outputs.detach().cpu().numpy())  # 统计每个batch的值
            batch_trues = np.concatenate((batch_trues, targets.detach().cpu().numpy().squeeze(1)), axis=0)
            preds = np.append(preds, batch_preds)  # 转换为数组的形式，并统计总预测值
            trues = np.concatenate((trues, batch_trues), axis=0)  # 转换为数组的形式，并统计总真值

            AUC, sklearn_accuracy, threshold, _, _, _, _, _ = get_cm(trues, preds)

            bar.desc = "[val__eppch__bar] Epoch:{} loss:{:.4f} accuracy:{:.4f} AUC:{:.4f} threshold:{:.4f}".format(
                epoch + 1, avg_loss, sklearn_accuracy, AUC, threshold)

    return avg_loss, AUC


def read_and_merge_data(label_file, file_path, patient_id_col='', label_col='', img_path_col=''):
    label_data = pd.read_csv(label_file)[[patient_id_col, label_col]]
    label_data[img_path_col] = label_data[patient_id_col].apply(lambda x: f"{file_path + str(x)}.nii.gz")

    return label_data


def feature_process(feature_path):
    feature_ori = pd.read_csv(feature_path)
    return feature_ori


def get_parser():
    parser = argparse.ArgumentParser(
        description='Model')
    parser.add_argument('--epochs', default=50,
                        type=int, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='initial learning rate (default: 0.05)')
    parser.add_argument('--momentum', default=0.7, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--Tmax', default=50, type=float)
    parser.add_argument('--eta_min', default=0, type=float)
    parser.add_argument('--warmup_epoch', default=5, type=int)
    parser.add_argument('--lat_dim', default=2, type=int)
    parser.add_argument('--n_channels', default=2, type=int)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--checkpoint_dir', default='./checkpoint', type=str)
    parser.add_argument('--actf', default='ReLU', type=str)

    return parser.parse_args()



if __name__ == '__main__':
    args = get_parser()
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    seed = 3407
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
    args.feature_fold = ''
    test_dataloader = {}
    pid_col_name = ''
    label_col_name = ''
    img_path_col = ''
    dims_1 = [256]
    dims_2 = [64]
    lays = [3]
    data_file_name_prefix = ''
    img_file_name = f''
    rt = pd.read_csv(r'')
    rt_renew_cohort = {'RT': rt}

    train_all = read_and_merge_data(f'', img_file_name)

    train_test = None
    if train_cohort_name == '':
        train_test = pd.merge(train_all, rt, how='inner', on=pid_col_name)

    train_deep_feat_pd = pd.read_csv(r'')
    val_deep_feat_pd = pd.read_csv(r'')

    radio_intra_features = feature_process('')
    radio_intra_features['患者序号'] = radio_intra_features['患者序号'].astype(str)
    radio_intra_features, scaler = tools.scale_on_min_max_train(radio_intra_features, pid_col_name)
    labels = train_test[[pid_col_name, label_col_name]]


    X_train, X_val, y_train, y_val = train_test_split(train_test['患者序号'].values,
                                                      train_test[label_col_name].values, test_size=0.2,
                                                      stratify=train_test[label_col_name].values)

    train_dataset = MyDatasetFusion(pid=X_train, deep_features=train_deep_feat_pd, radio_features=radio_intra_features, labels=labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, drop_last=True)

    val_dataset = MyDatasetFusion(pid=X_val, deep_features=val_deep_feat_pd, radio_features=radio_intra_features, labels=labels)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, drop_last=False)

    for dim_1, dim_2 in zip(dims_1, dims_2):
        for lay in lays:
            # Creating model
            model = mcVAE_BRCA_mut(radio_f_dim=707, deep_f_dim=256, output_dim=128, head_hidden_1=dim_1,
                                   head_hidden_2=dim_2, head_id=lay, actf=args.actf).cuda(args.device)

            total_negative_samples = np.count_nonzero(train_test[label_col_name].values == 0)
            total_positive_samples = np.count_nonzero(train_test[label_col_name].values == 1)
            logit_pos_weight = torch.tensor([total_negative_samples / total_positive_samples]).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            criterion = torch.nn.BCEWithLogitsLoss()
            scheduler1 = CosineAnnealingLR(
                optimizer, T_max=args.Tmax, eta_min=args.eta_min)

            scheduler2 = GradualWarmupScheduler(
                optimizer, multiplier=1, total_epoch=args.warmup_epoch, after_scheduler=scheduler1)

            early_stop = 0
            min_loss = 999999
            for epoch in range(args.epochs):
                early_stop = early_stop + 1
                AUC_list = []
                epoch_record = {'Epoch': epoch}

                if early_stop > 25:
                    print('Early stop!')
                    break

                epoch_record = {'Epoch': epoch}
                train_loss, train_AUC = train(train_dataloader, model, criterion, optimizer, device, epoch)
                AUC_list.append(train_AUC)
                epoch_record['train_AUC'] = train_AUC

                scheduler2.step()
                if epoch < args.warmup_epoch:
                    print('current learning rate: {}'.format(scheduler2.get_last_lr()))
                else:
                    print('current learning rate: {}'.format(scheduler1.get_last_lr()))

                if train_loss < min_loss:
                    min_loss = train_loss
                    epoch_test = epoch
                    early_stop = 0

                _, val_AUC = validate(val_dataloader, model, criterion, device, 'val', epoch, '')
                AUC_list.append(val_AUC)
                epoch_record['val_AUC'] = val_AUC
                filename = f""
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                             optimizer=optimizer.state_dict())
                torch.save(state, filename)


