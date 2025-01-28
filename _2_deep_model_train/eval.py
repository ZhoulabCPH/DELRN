import sys
from reproduction.deep_model_train.fmcib.datasets.dataSetAug import MyDatasetAug
from reproduction.deep_model_train.fmcib.models import fmcib_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    RandFlip,
)


def get_cm(AllLabels, AllValues):
    Auc = 0
    m = t = 0
    Precision = 0
    Recall = 0

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

        TP = cm[1][1]
        FP = cm[0][1]
        FN = cm[1][0]
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return Auc, Acc, t, Neg_rate, Pos_rate, len(AllLabels), Pos_num, Neg_num, Precision, Recall


def validate(dataloader, model, criterion, device, epoch, mode, hp, loss_name, key):
    # print('validate')
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
            threshold = 0

            if loss_name == 'BCE' or loss_name == 'BCEW':
                # BCE
                loss = criterion(outputs.squeeze(0), test_targets.float())  # 使用损失函数进行比对
                test_loss_list.append(loss.item())  # 以数组的形式，添加到训练损失总值中
                test_outputs = outputs.squeeze(0)  # 比较并输出一组元素中最大值所在的索引 argmax(1) 横向比较
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
                AUC, test_accuracy, threshold, Neg_rate, Pos_rate, All_num, Pos_num, Neg_num, Precision, Recall = get_cm(test_trues, test_preds)
            elif loss_name == 'CE':
                test_batch_outputs.extend(sm_outputs[:, 1].detach().cpu().numpy())
                test_accuracy = accuracy_score(test_trues, test_preds)
                visual_acc.append(test_accuracy)
            avg_loss = np.average(np.array(test_loss_list))
            test_bar.desc = "[{}__epoch__bar] Epoch:{} loss:{:.4f} accuracy:{:.4f} auc:{:.4f} Precision:{:.4f} Recall:{:.4f} threshold:{:.4f}".format(mode,
                epoch + 1, avg_loss, test_accuracy, AUC, Precision, Recall, threshold)

        if loss_name == 'BCE':
            print(f'Pos_Accuracy: {Pos_rate:.2f}, Pos_num: {Pos_num}')
            print(f'Neg_Accuracy: {Neg_rate:.2f}, Neg_num: {Neg_num}')
        elif loss_name == 'CE':
            class_counts = {0: {'correct': 0, 'incorrect': 0},
                            1: {'correct': 0, 'incorrect': 0}}

            # 假设 test_preds 和 test_trues 是两个包含预测和真实标签的列表
            for pred, true in zip(test_preds, test_trues):
                if pred == true:
                    class_counts[true]['correct'] += 1
                else:
                    class_counts[true]['incorrect'] += 1

            # 打印每个类别的统计信息
            acc = []
            for class_label, counts in class_counts.items():
                correct_count = counts['correct']
                incorrect_count = counts['incorrect']
                total_count = correct_count + incorrect_count
                accuracy = correct_count / total_count if total_count > 0 else 0
                acc.append(accuracy)
                print(f'Class {class_label}: Correct: {correct_count}, Incorrect: {incorrect_count}, Accuracy: {accuracy:.2%}')

        test_pred_hat = [int(i >= threshold) for i in test_preds]

    return AUC, visual_acc, test_trues, test_preds, threshold, test_pred_hat


def read_and_merge_data(label_file, file_path, patient_id_col='', label_col='', img_path_col=''):
    label_data = pd.read_csv(label_file)[[patient_id_col, label_col]]
    label_data[img_path_col] = label_data[patient_id_col].apply(lambda x: f"{file_path + str(x)}.nii.gz")

    return label_data


def get_parser():
    parser = argparse.ArgumentParser(
        description='Model')
    parser.add_argument('--epochs', default=1300,
                        type=int, help='number of epochs')
    parser.add_argument('--loss', type=str, default='BCEW')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', default=8e-7, type=float,
                        help='initial learning rate (default: 0.05)')
    parser.add_argument('--momentum', default=0.7, type=float)
    parser.add_argument('--weight_decay', default=1e-7, type=float)
    parser.add_argument('--Tmax', default=60, type=float)
    parser.add_argument('--eta_min', default=0, type=float)
    parser.add_argument('--num_worker', default=0, type=int)
    parser.add_argument('--warmup_epoch', default=10, type=int)
    parser.add_argument('--finetune', default=False, type=bool)
    parser.add_argument('--subtrahend', default=-125, type=int)
    parser.add_argument('--divisor', default=350, type=int)
    parser.add_argument('--winMin', default=-125, type=int)
    parser.add_argument('--winMan', default=225, type=int)
    parser.add_argument('--aug', default=False, type=bool)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--act_function', default='GELU', type=str)

    return parser.parse_args()


def draw_AUC_confusion_matrix(AllLabels, AllValues, threshold_train=None, cohort_name='', anno=[]):
    threshold = 0
    # 绘制 ROC 曲线
    plt.figure()
    colors = ['red', 'darkorange']
    for name, color, AllLabel, AllValue in zip(anno, colors, AllLabels, AllValues):
        fpr, tpr, thresholds = roc_curve(AllLabel, AllValue, pos_label=1)
        threshold = thresholds[np.argmax(tpr - fpr)]
        if threshold_train is not None:
            threshold = threshold_train
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=name + ' ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} Cohort'.format(cohort_name))
    plt.legend(loc="lower right")
    plt.show()
    for name, color, AllLabel, AllValue in zip(anno, colors, AllLabels, AllValues):
        # 使用最佳阈值确定预测
        predicted_labels = np.where(np.array(AllValue) >= threshold, 1, 0)
        cm = confusion_matrix(AllLabel, predicted_labels)

        # 绘制混淆矩阵的热图
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', square=True, cbar=False)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('{} {} Confusion Matrix'.format(cohort_name, name))
        plt.show()



if __name__ == '__main__':
    args = get_parser()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    if args.aug:
        transform = Compose(
            [
                RandFlip(prob=1, spatial_axis=0),
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
    rt_renew = pd.read_csv(r'')
    rt_renew_cohort = {'': rt_renew}
    train_all = read_and_merge_data(f'')

    train_test = []
    if train_cohort_name == '':
        train_test = pd.merge(train_all, rt_renew, how='inner', on=pid_col_name)

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
        test_all = read_and_merge_data(f'', f'')
        if cohort_name == '':
            for key in rt_renew_cohort.keys():
                print(key)
                rt_test = pd.merge(test_all, rt_renew_cohort[key], how='inner', on=pid_col_name)
                test_data = MyDatasetAug(rt_test[img_path_col], rt_test[label_col_name], subtrahend=args.subtrahend, divisor=args.divisor, winMin=args.winMin, winMax=args.winMan)
                test_dataloader[key] = torch.utils.data.DataLoader(test_data, batch_size=1,
                                                            shuffle=False, num_workers=args.num_worker, drop_last=False)

    # EVAL
    eval_path = ''
    model = fmcib_model(eval_mode=True, heads=[4096, 2048, 256, 1], actFunction=args.act_function, eval_path=eval_path, device_num=args.device).to(device)
    new_parameters = []
    if args.finetune:
        for param in model.parameters():
            param.requires_grad = False

        # Enable gradient computation for all parameters in model.conv_seg explicitly
        for param in model.heads.parameters():
            param.requires_grad = True

    for pname, p in model.named_parameters():  # 返回各层中参数名称和数据。
        for layer_name in ['heads']:
            if pname.find(layer_name) >= 0:
                new_parameters.append(p)
                break

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

    hp = {''}
    all_epochs_auc_records = []
    for epoch in range(args.epochs):
        AUC_list = []
        epoch_record = {'Epoch': epoch}

        AUC_train, v_visual_acc_train, train_AllLabels, train_AllValues, threshold, _ = validate(train_dataloader, model, criterion, device, epoch, 'val', hp, args.loss, 'train_val')
        AUC, vv_visual_acc, val_AllLabels, val_AllValues, _, _ = validate(val_dataloader, model, criterion, device, epoch, 'val', hp, args.loss, 'internel_val')
        epoch_record['val_AUC'] = AUC
        AUC_list.append(AUC)
        for key in test_dataloader.keys():
            AUC, v_visual_acc, test_AllLabels, test_AllValues, _, test_pred_hat = validate(test_dataloader[key], model, criterion, device, epoch, 'test', hp, args.loss, key)
            draw_AUC_confusion_matrix([test_AllLabels], [test_AllValues], threshold_train=threshold, cohort_name=key,
                                      anno=['External Validation'])
            epoch_record[f'{key}_AUC'] = AUC
            AUC_list.append(AUC)

