import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, confusion_matrix
import argparse
import os
import numpy as np
import pandas as pd
from warmup_scheduler import GradualWarmupScheduler
from _2_deep_model_train.fmcib.models import fmcib_model
import random
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn.functional as F
from _2_deep_model_train.fmcib.datasets.dataSetAug import MyDatasetAug
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.transforms import (
    Compose,
    RandAxisFlip,
    RandRotate90,
)


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


def train(dataloader, model, criterion, optimizer, device, epoch, loss_name):
    # print('len(dataloader): {}'.format(len(dataloader)))
    model.train()
    train_loss_list = []  # 每次训练的loss 用于展示数据
    train_preds = np.array([])  # 预测值 prediction -s 总数
    train_trues = np.empty(shape=[0])  # 真值,总数
    visual_acc = []
    train_bar = tqdm(dataloader)  # 进度条显示数据
    threshold = 0
    AUC = 0

    # 每次都给一个batch
    for step, data in enumerate(train_bar):

        train_batch_preds = np.array([])  # 每个batch训练的预测值
        train_batch_trues = np.empty(shape=[0])  # 每次batch训练的真值

        tokens, targets, targets_path = data  # 获取模型中的数据 特征、目标值 均为一个batch数组
        tokens = tokens.to(device)
        targets = targets.to(device)  # 根据device选择设备 GPU or CPU
        optimizer.zero_grad()  # 优化器清零
        outputs = model(tokens)  # 得到预测值

        train_outputs = []
        if loss_name == 'BCE' or loss_name == 'BCEW':
            # BCE
            loss = criterion(outputs.squeeze(), targets.float())  # 使用损失函数进行比对
            loss.backward()  # 反向传播
            optimizer.step()  # 使用优化器
            train_loss_list.append(loss.item())  # 以数组的形式，添加到训练损失总值中
            train_outputs = outputs.squeeze()  # 比较并输出一组元素中最大值所在的索引 argmax(1) 横向比较
        elif loss_name == 'CE':
            # CE
            loss = criterion(outputs, targets.long())  # 使用损失函数进行比对
            loss.backward()  # 反向传播
            optimizer.step()  # 使用优化器
            train_loss_list.append(loss.item())  # 以数组的形式，添加到训练损失总值中
            train_outputs = outputs.argmax(dim=1)  # 比较并输出一组元素中最大值所在的索引 argmax(1) 横向比较

        train_batch_preds = np.append(train_batch_preds, train_outputs.detach().cpu().numpy())  # 统计每个batch的值
        train_batch_trues = np.concatenate((train_batch_trues, targets.detach().cpu().numpy()), axis=0)

        train_preds = np.append(train_preds, train_batch_preds)  # 转换为数组的形式，并统计总预测值
        train_trues = np.concatenate((train_trues, train_batch_trues), axis=0)  # 转换为数组的形式，并统计总真值
        sklearn_accuracy = 0

        if loss_name == 'BCE' or loss_name == 'BCEW':
            AUC, sklearn_accuracy, threshold, _, _, _, _, _ = get_cm(train_trues, train_preds)
        elif loss_name == 'CE':
            sklearn_accuracy = accuracy_score(train_trues, train_preds)

            visual_acc.append(sklearn_accuracy)   # 统计每个Batch的准确值
        avg_loss = np.average(np.array(train_loss_list))

        train_bar.desc = "[train__eppch__bar] Epoch:{} loss:{:.4f} accuracy:{:.4f} AUC:{:.4f} threshold:{:.4f}".format(
            epoch + 1, avg_loss, sklearn_accuracy, AUC, threshold)

    return AUC, train_loss_list, visual_acc

def validate(dataloader, model, criterion, device, epoch, mode, loss_name, key):
    print(mode+' cohort_'+key)
    model.eval()
    test_preds = []
    test_trues = []
    test_batch_outputs = []
    test_loss_list = []
    visual_acc = []
    threshold = 0
    Neg_rate = 0
    Pos_rate = 0
    All_num, Pos_num, Neg_num = 0, 0, 0
    AUC = 0

    test_bar = tqdm(dataloader, file=sys.stdout)
    with torch.no_grad():  # 这句话就将这里面的语句不去关注梯度信息
        for step, data in enumerate(test_bar):
            test_batch_preds = []
            test_batch_trues = []
            test_tokens, test_targets, targets_path = data
            test_tokens = test_tokens.to(device)
            test_targets = test_targets.to(device)
            outputs = model(test_tokens)

            test_outputs = []
            sm_outputs = []
            test_accuracy = 0

            if loss_name == 'BCE' or loss_name == 'BCEW':
                # BCE
                loss = criterion(outputs.squeeze(), test_targets.float())  # 使用损失函数进行比对
                test_loss_list.append(loss.item())  # 以数组的形式，添加到训练损失总值中
                test_outputs = outputs.squeeze()  # 比较并输出一组元素中最大值所在的索引 argmax(1) 横向比较
            elif loss_name == 'CE':
                # CE
                loss = criterion(outputs, test_targets.long())  # 使用损失函数进行比对
                test_loss_list.append(loss.item())  # 以数组的形式，添加到训练损失总值中
                test_outputs = outputs.argmax(dim=1)  # 比较并输出一组元素中最大值所在的索引 argmax(1) 横向比较
                sm_outputs = F.softmax(outputs, dim=1)

            test_loss_list.append(loss.item())
            test_batch_preds.extend(test_outputs.detach().cpu().numpy())
            test_batch_trues.extend(test_targets.detach().cpu().numpy())

            test_preds.extend(test_batch_preds)
            test_trues.extend(test_batch_trues)
            if loss_name == 'BCE' or loss_name == 'BCEW':
                AUC, test_accuracy, threshold, Neg_rate, Pos_rate, All_num, Pos_num, Neg_num = get_cm(test_trues, test_preds)
            elif loss_name == 'CE':
                test_batch_outputs.extend(sm_outputs[:, 1].detach().cpu().numpy())
                test_accuracy = accuracy_score(test_trues, test_preds)
                visual_acc.append(test_accuracy)
            avg_loss = np.average(np.array(test_loss_list))

            test_bar.desc = "[{}__epoch__bar] Epoch:{} loss:{:.4f} accuracy:{:.4f} auc:{:.4f} threshold:{:.4f}".format(mode,
                epoch + 1, avg_loss, test_accuracy, AUC, threshold)

        if loss_name == 'BCE':
            print(f'Pos_Accuracy: {Pos_rate:.2f}, Pos_num: {Pos_num}')
            print(f'Neg_Accuracy: {Neg_rate:.2f}, Neg_num: {Neg_num}')
        elif loss_name == 'CE':
            class_counts = {0: {'correct': 0, 'incorrect': 0},
                            1: {'correct': 0, 'incorrect': 0}}

            for pred, true in zip(test_preds, test_trues):
                if pred == true:
                    class_counts[true]['correct'] += 1
                else:
                    class_counts[true]['incorrect'] += 1

            acc = []
            for class_label, counts in class_counts.items():
                correct_count = counts['correct']
                incorrect_count = counts['incorrect']
                total_count = correct_count + incorrect_count
                accuracy = correct_count / total_count if total_count > 0 else 0
                acc.append(accuracy)
                print(f'Class {class_label}: Correct: {correct_count}, Incorrect: {incorrect_count}, Accuracy: {accuracy:.2%}')

    return AUC, visual_acc


def read_and_merge_data(label_file, file_path, patient_id_col='', label_col='', img_path_col=''):
    label_data = pd.read_csv(label_file)[[patient_id_col, label_col]]
    label_data[img_path_col] = label_data[patient_id_col].apply(lambda x: f"{file_path + str(x)}.nii.gz")

    return label_data


def get_parser():
    parser = argparse.ArgumentParser(
        description='Model')
    parser.add_argument('--epochs', default=200,
                        type=int, help='number of epochs')
    parser.add_argument('--loss', type=str, default='BCEW')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', default=2e-6, type=float,
                        help='initial learning rate (default: 0.05)')
    parser.add_argument('--momentum', default=0.7, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--Tmax', default=60, type=float)
    parser.add_argument('--eta_min', default=1e-6, type=float)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--num_worker', default=0, type=int)
    parser.add_argument('--warmup_epoch', default=40, type=int)
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
                RandAxisFlip(prob=0.5),
                RandRotate90(prob=0.5),
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
    train_cohort_renew_name = ''
    label_path = ''
    datasets_val = ['']
    label_path_val = ['']

    test_dataloader = {}
    pid_col_name = ''
    label_col_name = ''
    img_path_col = ''

    data_dir_name = ''
    data_file_name = ''

    img_file_name = f''

    # 队列划分
    rt_renew = pd.read_csv(r'')
    rt_renew_cohort = {'RT': rt_renew}
    train_all = read_and_merge_data(f'', img_file_name)
    train_test = []
    if train_cohort_name == 'RT':
        train_test = pd.merge(train_all, rt_renew_cohort, how='inner', on=pid_col_name)
    elif train_cohort_name == 'EXV2':
        train_test = train_all
    X_train, X_val, y_train, y_val = train_test_split(train_test[img_path_col].values,
                                                      train_test[label_col_name].values, test_size=0.2,
                                                      stratify=train_test[label_col_name].values)


    train_dataset = MyDatasetAug(X_train, y_train, subtrahend=args.subtrahend, divisor=args.divisor, transform=transform, winMin=args.winMin, winMax=args.winMan)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.num_worker, drop_last=True)

    val_dataset = MyDatasetAug(X_val, y_val, subtrahend=args.subtrahend, divisor=args.divisor, winMin=args.winMin, winMax=args.winMan)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2,
                                                 shuffle=True, num_workers=args.num_worker, drop_last=True)

    for label_path, cohort_name in zip(label_path_val, datasets_val):
        test_all = read_and_merge_data(f'',f'')

        if cohort_name == ''
            for key in rt_renew_cohort.keys():
                print(key)
                if key != train_cohort_renew_name:
                    rt_test = pd.merge(test_all, rt_renew_cohort[key], how='inner', on=pid_col_name)
                    test_data = MyDatasetAug(rt_test[img_path_col], rt_test[label_col_name], subtrahend=args.subtrahend, divisor=args.divisor, winMin=args.winMin, winMax=args.winMan)
                    test_dataloader[key] = torch.utils.data.DataLoader(test_data, batch_size=2,
                                                                shuffle=True, num_workers=args.num_worker, drop_last=True)


    model = fmcib_model(eval_mode=False, heads=[4096, 2048, 256, 1], actFunction=args.act_function, device_num=args.device).to(device)
    new_parameters = []

    if args.finetune:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.heads.parameters():
            param.requires_grad = True

    for pname, p in model.named_parameters():
        for layer_name in ['heads']:
            if pname.find(layer_name) >= 0:
                new_parameters.append(p)
                break

    new_parameters_id = list(map(id, new_parameters))
    base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
    parameters = {'base_parameters': base_parameters,
                  'new_parameters': new_parameters}
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

    # Train
    hp = {''}

    scheduler1 = CosineAnnealingLR(
        optimizer, T_max=args.Tmax, eta_min=args.eta_min)

    scheduler2 = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=args.warmup_epoch, after_scheduler=scheduler1)

    all_epochs_auc_records = []
    for epoch in range(args.epochs):
        AUC_list = []
        epoch_record = {'Epoch': epoch}
        AUC_train, train_loss_list, t_visual_acc = train(train_dataloader, model, criterion, optimizer, device, epoch, args.loss)
        AUC, vv_visual_acc = validate(val_dataloader, model, criterion, device, epoch, 'val', hp, args.loss, 'internel_val')
        epoch_record['val_AUC'] = AUC
        AUC_list.append(AUC)

        scheduler2.step()
        if epoch < args.warmup_epoch:
            print('current learning rate: {}'.format(scheduler2.get_last_lr()))
        else:
            print('current learning rate: {}'.format(scheduler1.get_last_lr()))

        filename = f"./checkpoints/epoch_{epoch}_.pth"
        torch.save(model.state_dict(), filename)