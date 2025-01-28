import sys
from reproduction._3_feature_fusion.datasets.dataSetFusion import MyDatasetFusion
import argparse
import time
import os
import numpy as np
import pandas as pd
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from reproduction._3_feature_fusion import tools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from lifelines.statistics import multivariate_logrank_test
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from _5_feature_fusion.model.model_mcvae import mcVAE_BRCA_mut

torch.backends.cuda.matmul.allow_tf32 = False



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
    # print("[AUC/{:.4f}] [Threshold/{:.4f}] [Acc/{:.4f}]".format(Auc, t, Acc))
    # print("{:.2f}% {:.2f}%".format(
    #     cm[0][0] / Neg_num * 100, cm[0][1] / Neg_num * 100))
    # print("{:.2f}% {:.2f}%".format(
    #     cm[1][0] / Pos_num * 100, cm[1][1] / Pos_num * 100))

    return Auc, Acc, t, Neg_rate, Pos_rate, len(AllLabels), Pos_num, Neg_num





def validate(dataloader, model, criterion, device, mode, epoch, key):
    print(mode+' cohort_'+key)
    model.eval()
    loss_list = []  # 每次训练的loss 用于展示数据
    avg_loss = 0
    pids = np.array([])  # 预测值 prediction -s 总数
    preds = np.array([])  # 预测值 prediction -s 总数
    trues = np.empty(shape=[0])  # 真值,总数
    AUC = 0
    threshold = 0

    bar = tqdm(dataloader, file=sys.stdout)  # 进度条显示数据
    with torch.no_grad():  # 这句话就将这里面的语句不去关注梯度信息
        for step, data in enumerate(bar):

            batch_pids = np.array([])
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

            loss = criterion(outputs, targets.float().squeeze(1))  # 使用损失函数进行比对
            outputs = torch.sigmoid(outputs)
            loss_list.append(loss.cpu().detach().numpy())
            avg_loss = np.average(np.array(loss_list))

            batch_pids = np.append(batch_pids, pid)  # 统计每个batch的值
            batch_preds = np.append(batch_preds, outputs.detach().cpu().numpy())  # 统计每个batch的值
            batch_trues = np.concatenate((batch_trues, targets.detach().cpu().numpy().squeeze(1)), axis=0)

            pids = np.append(pids, batch_pids)
            preds = np.append(preds, batch_preds)  # 转换为数组的形式，并统计总预测值
            trues = np.concatenate((trues, batch_trues), axis=0)  # 转换为数组的形式，并统计总真值
            # preds_sig = torch.sigmoid(preds)

            AUC, sklearn_accuracy, threshold, _, _, _, _, _ = get_cm(trues, preds)

            bar.desc = "[val__eppch__bar] Epoch:{} loss:{:.4f} accuracy:{:.4f} AUC:{:.4f} threshold:{:.4f}".format(
                epoch + 1, avg_loss, sklearn_accuracy, AUC, threshold)

    return avg_loss, AUC, pids, preds, trues, threshold


def read_and_merge_data(label_file, file_path, patient_id_col='患者序号', label_col='pCR', downSampling=False, img_path_col='img_path'):

    label_data = pd.read_csv(label_file)[[patient_id_col, label_col]]
    label_data[img_path_col] = label_data[patient_id_col].apply(lambda x: f"{file_path + str(x)}.nii.gz")

    # label_data[patient_id_col] = label_data[patient_id_col].apply(lambda x: f"{file_path + str(x).zfill(5)}.nii.gz")
    # label_data[patient_id_col] = label_data[patient_id_col].apply(
    #     lambda x: f"{file_path}0{x}.nii.gz" if len(str(x)) in [3, 4] else f"{file_path}{x}.nii.gz")

    if downSampling:
        # 分别筛选出标签为0和1的样本
        data_0 = label_data[label_data[label_col] == 0]
        data_1 = label_data[label_data[label_col] == 1]

        # 确定两个类别中较小的样本数量
        min_count = min(len(data_0), len(data_1))

        # 对较多的类别进行随机抽样
        data_0_sampled = data_0.sample(n=min_count, random_state=42)  # 使用 random_state 保证结果可重现

        # 合并两部分样本得到最终的平衡样本集
        label_data = pd.concat([data_0_sampled, data_1])
    # return label_data[patient_id_col].values, label_data[label_col].values
    return label_data


def feature_process(feature_path, pid_col_name):
    feature_ori = pd.read_csv(feature_path)
    # feature_min_max = tools.scale_on_min_max(data=feature_ori, pid_col_name=pid_col_name)
    return feature_ori


def KM(data):
    df=data
    # 计算生存曲线
    # 分离数据
    # 进行Logrank检验
    df['DFS_time'] = df['DFS_time'].astype(float)
    df['DFS_State'] = df['DFS_State'].astype(float)

    results_pre = multivariate_logrank_test(df['DFS_time'], df['pre_label'], df['DFS_State'])
    results_true = multivariate_logrank_test(df['DFS_time'], df['pCR'], df['DFS_State'])


    # 输出Logrank检验的P值
    print('Predicted Logrank test p-value:', results_pre.p_value)
    print('Ture Logrank test p-value:', results_true.p_value)

    # # Kaplan-Meier生存曲线
    kmf = KaplanMeierFitter()

    plt.figure(figsize=(10, 7))

    for label, grouped_df in df.groupby('pre_label'):
        kmf.fit(grouped_df['DFS_time'], grouped_df['DFS_State'], label=f'pre_label Group {label}')
        kmf.plot(ci_show=False, color='blue')

    # 绘制 pCR 的生存曲线，使用橙色
    for label, grouped_df in df.groupby('pCR'):
        kmf.fit(grouped_df['DFS_time'], grouped_df['DFS_State'], label=f'pCR Group {label}')
        kmf.plot(ci_show=False, color='orange')

    # 添加标题和标签
    plt.title('Kaplan-Meier Survival Curve')
    plt.xlabel('Time')
    plt.ylabel('Survival probability')

    # 在图上添加p-value
    plt.text(0.6, 0.6, f'pre_label p-value = {results_pre.p_value:.4f}', transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.6, 0.7, f'results_true p-value = {results_true.p_value:.4f}', transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

    return results_pre.p_value


def get_prelabel(pids, preds, threshold):
    binary_preds = np.where(preds >= threshold, 1, 0)
    pre_df = pd.DataFrame({
        '患者序号': pids,
        'pre_label': binary_preds,
        'pre_score': preds
    })
    pre_df['患者序号'] = pre_df['患者序号'].astype(int)

    return pre_df


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


def calculate_and_plot_metrics(y_true, y_pred, threshold):
    # 根据阈值生成预测标签
    y_pred_label = [1 if p >= threshold else 0 for p in y_pred]

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label).ravel()

    # 计算各种指标
    accuracy = accuracy_score(y_true, y_pred_label)
    precision = precision_score(y_true, y_pred_label)
    recall = recall_score(y_true, y_pred_label)  # 即敏感性
    specificity = tn / (tn + fp)  # 特异性
    f1 = f1_score(y_true, y_pred_label)

    # 指标名称和对应值
    metrics = ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1 Score']
    values = [accuracy, precision, recall, specificity, f1]

    # 设置图表风格
    sns.set(style="whitegrid")

    # 定义彩色调色板
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"]

    # 创建柱状图
    plt.figure(figsize=(8, 6))
    sns.barplot(x=metrics, y=values, palette=colors)

    # 在柱子顶部添加数值标签
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=12)

    # 添加图表标签和标题
    plt.ylim(0, 1)  # 设置 y 轴范围，确保数值标签可见
    plt.ylabel('Score')
    plt.title('Model Performance Metrics')

    # 显示图表
    plt.show()


def draw_score_violin(df, title):
    sns.set(style="whitegrid")

    # 创建图形和轴
    plt.figure(figsize=(12, 8))

    # 生成小提琴图
    sns.violinplot(x='pCR', y='pre_score', data=df, palette='muted', inner='quartile')

    # 添加标题和标签
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('pCR Label', fontsize=14)
    plt.ylabel('pre_score', fontsize=14)

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 显示图形
    plt.show()


def get_parser():
    parser = argparse.ArgumentParser(
        description='PyTorch Implementation of multiple Instance learning')
    parser.add_argument('--epochs', default=1,
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
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--checkpoint_dir', default='./checkpoint', type=str)
    parser.add_argument('--actf', default='ReLU', type=str)


    return parser.parse_args()



if __name__ == '__main__':
    args = get_parser()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    args.feature_fold = 'features_HUCUT_bW20_115'

    train_cohort_name = 'RT'
    train_cohort_renew_name = 'RT_noIMMU'
    label_path = 'RT_solved2.csv'
    datasets_val = ['RT', 'EXV1', 'EXV2', 'SHANGHAI']
    label_path_val = ['RT_solved2.csv', 'EXV1.csv', 'EXV2.csv', 'shuanghai_label_globe.csv']

    # 获取生存信息
    RT_survival = pd.read_csv(r'..\..\data_summary\label\OSDFS\RT_cli_osdfs.csv')
    EXV2_survival = pd.read_csv(r'..\..\data_summary\label\OSDFS\EXV2_cli_osdfs.csv')

    test_dataloader = {}
    pid_col_name = '患者序号'
    label_col_name = 'pCR'
    img_path_col = 'img_path'

    lrs = [5e-6]
    dims_1 = [256]
    dims_2 = [64]
    # actfs = ['ReLU', 'Tanh', 'LeakyReLU', 'GELU', 'SiLU']
    actfs = ['LeakyReLU']
    lays = [3]
    batch_sizes = [8]
    data_file_name_prefix = 'bounding_box_111_noW_toMax505050_111_noW_'

    img_file_name = f'../../dataset/bounding_box_img/{train_cohort_name}/IMG/{data_file_name_prefix}'

    rt_renew = pd.read_csv(r'..\..\data_summary\label\renew_patient\RT_pid_class.csv')
    # 0=IMMU,1=ORI
    rt_renew_class = {'IMMU': 0, 'noIMMU': 1}
    rt_IMMU_cohort = rt_renew[rt_renew['class'] == rt_renew_class['IMMU']]
    rt_noIMMU_cohort = rt_renew[rt_renew['class'] == rt_renew_class['noIMMU']]
    rt_renew_cohort = {'RT_IMMU': rt_IMMU_cohort, 'RT_noIMMU': rt_noIMMU_cohort}

    exv1_renew = pd.read_csv(r'..\..\data_summary\label\EXV1.csv')
    exv1_renew_cohort = {'EXV1_TUNE': exv1_renew[['患者序号']]}
    exv1_renew_c = pd.read_csv(
        r'..\..\data_summary\label\renew_patient\EXV1_pid_class.csv')
    exv1_renew_c['患者序号'] = exv1_renew_c['患者序号'].astype(str)
    exv1_renew_class = {'EXV1_500': 2, 'EXV1_150': 1, 'EXV1_038': 0}
    exv1_no150_cohort = exv1_renew_c[exv1_renew_c['class'] != exv1_renew_class['EXV1_150']]

    exv2_renew = pd.read_csv(r'..\..\data_summary\label\EXV2_solved1.csv')
    exv2_del = pd.read_csv(r'..\..\data_summary\肿瘤数量\EXV2.csv')
    exv2_renew = exv2_renew[~exv2_renew['患者序号'].isin(exv2_del['患者序号'])]
    exv2_renew_cohort = {'EXV2': exv2_renew[['患者序号']]}

    shanghai_renew = pd.read_csv(r'..\..\data_summary\label\renew_patient\SHANGHAI_pid_class.csv')
    shanghai_renew = shanghai_renew[['患者序号', 'class']]
    # 0=IMMU,1=ORI
    shanghai_renew_class = {'2.7': 0, '3': 1, '4.9': 2, '5': 3, '6': 4}
    shanghai_no6_cohort = shanghai_renew[shanghai_renew['class'] != shanghai_renew_class['6']]
    shanghai_renew_cohort = {'shanghai_no6': shanghai_no6_cohort}

    # tid_train_all, label_train_all = read_and_merge_data(f'../data_summary/ANO/{label_path}', img_file_name)
    train_all = read_and_merge_data(f'../../data_summary/label/{label_path}', img_file_name)

    train_test = None
    if train_cohort_name == 'RT':
        train_test = pd.merge(train_all, rt_noIMMU_cohort, how='inner', on=pid_col_name)
    # train_deep_feat_pd = pd.read_csv(r'D:\Suyang\python\escc_xinfuzhu\dataset\deep_features\RT_train_features_256.csv')
    train_deep_feat_pd = pd.read_csv(r'../../dataset/deep_feature/RT_train_features_256.csv')
    train_deep_feat_pd['患者序号'] = train_deep_feat_pd['CT_Name'].apply(lambda x: x.split('_')[-1].split('.')[0])

    val_deep_feat_pd = pd.read_csv(r'../../dataset/deep_feature/RT_val_features_256.csv')
    val_deep_feat_pd['患者序号'] = val_deep_feat_pd['CT_Name'].apply(lambda x: x.split('_')[-1].split('.')[0])

    # radio_intra_features = feature_process(
    #     'D:\Suyang\python\escc_xinfuzhu/model_radiomics_V2/intra_feats_fusion/{}/{}_features.csv'.format(args.feature_fold, 'RT_noIMMU'), pid_col_name)
    radio_intra_features = feature_process(
        rf'../../dataset/radiomic_feature/{args.feature_fold}/RT_noIMMU_features.csv', pid_col_name)
    radio_intra_features['患者序号'] = radio_intra_features['患者序号'].astype(str)
    radio_intra_features, scaler = tools.scale_on_min_max_train(radio_intra_features, pid_col_name)


    labels = train_test[[pid_col_name, label_col_name]]

    for batch_size in batch_sizes:
        args.batch_size = batch_size

        X_train, X_val, y_train, y_val = train_test_split(train_test['患者序号'].values,
                                                          train_test[label_col_name].values, test_size=0.2, random_state=7701,
                                                          stratify=train_test[label_col_name].values)

        train_dataset = MyDatasetFusion(pid=X_train, deep_features=train_deep_feat_pd, radio_features=radio_intra_features, labels=labels)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                                       shuffle=False, drop_last=False)

        val_dataset = MyDatasetFusion(pid=X_val, deep_features=val_deep_feat_pd, radio_features=radio_intra_features, labels=labels)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                     shuffle=False, drop_last=False)


        features = {}
        pt_features = {}
        labels = {}


        for label_path, cohort_name in zip(label_path_val, datasets_val):

            test_all = read_and_merge_data(f'../../data_summary/label/{label_path}',
                                           f'../../dataset/bounding_box_img/{cohort_name}/IMG/{data_file_name_prefix}')



            if cohort_name == 'EXV2':
                for key in exv2_renew_cohort.keys():
                    # deep_feat_pd = pd.read_csv(r'D:\Suyang\python\escc_xinfuzhu\dataset\deep_features\EXV2_ORI_test_features_256.csv')
                    deep_feat_pd = pd.read_csv(r'../../dataset/deep_feature/EXV2_test_features_256.csv')
                    # deep_feat_pd_tune = pd.read_csv(r'D:\Suyang\python\escc_xinfuzhu/dataset/deep_features/EXV2_TUNE2_test_features_256.csv')
                    deep_feat_pd['患者序号'] = deep_feat_pd['CT_Name'].apply(lambda x: x.split('_')[-1].split('.')[0])
                    # deep_feat_pd_tune['患者序号'] = deep_feat_pd_tune['CT_Name'].apply(lambda x: x.split('_')[-1].split('.')[0])
                    # deep_feat_pd[deep_feat_pd['患者序号'] == '3410'] = deep_feat_pd_tune.loc[deep_feat_pd_tune['患者序号'] == '3410'].values
                    # deep_feat_pd[deep_feat_pd['患者序号'] == '1847'] = deep_feat_pd_tune.loc[
                    #     deep_feat_pd_tune['患者序号'] == '1847'].values


                    radio_intra_features = feature_process(
                        rf'../../dataset/radiomic_feature/{args.feature_fold}/{cohort_name}_features.csv', pid_col_name)
                    radio_intra_features['患者序号'] = radio_intra_features['患者序号'].astype(str)
                    # radio_intra_features = feature_process(
                    #     'D:/Suyang/python/escc_xinfuzhu/model_radiomics_V2/intra_feats_fusion/{}/{}_features.csv'.format(args.feature_fold, 'EXV2'), pid_col_name)
                    # radio_intra_features['患者序号'] = radio_intra_features['患者序号'].astype(str)
                    # radio_intra_features_tune = feature_process(
                    #     'D:/Suyang/python/escc_xinfuzhu/model_radiomics_V2/intra_feats_fusion/{}/{}_features.csv'.format(args.feature_fold, 'EXV2_TUNE55'), pid_col_name)
                    # radio_intra_features_tune['患者序号'] = radio_intra_features_tune['患者序号'].astype(str)
                    # radio_intra_features[radio_intra_features['患者序号'] == '3410'] = radio_intra_features_tune.loc[radio_intra_features_tune['患者序号'] == '3410'].values
                    # radio_intra_features[radio_intra_features['患者序号'] == '1847'] = radio_intra_features_tune.loc[radio_intra_features_tune['患者序号'] == '1847'].values

                    radio_intra_features = tools.scale_on_min_max_test(radio_intra_features, pid_col_name, scaler)
                    labels = test_all[[pid_col_name, label_col_name]]

                    test_data = MyDatasetFusion(pid=deep_feat_pd['患者序号'], deep_features=deep_feat_pd, radio_features=radio_intra_features, labels=labels)
                    test_dataloader[key] = torch.utils.data.DataLoader(test_data, batch_size=1,
                                                                       shuffle=False,
                                                                       drop_last=False)

            if cohort_name == 'EXV1':
                test_all_add = read_and_merge_data(f'../../data_summary/label/OSDFS/EXV1ADD_cli_osdfs.csv',
                                                   f'../../dataset/bounding_box_img/{cohort_name}/IMG/{data_file_name_prefix}')

                for key in exv1_renew_cohort.keys():

                    deep_feat_pd = pd.read_csv(r'../../dataset/deep_feature/EXV1_test_features_256.csv')
                    deep_feat_pd['患者序号'] = deep_feat_pd['CT_Name'].apply(lambda x: x.split('_')[-1].split('.')[0])
                    exv1_renew['患者序号'] = exv1_renew['患者序号'].astype(str)
                    deep_feat_pd = pd.merge(deep_feat_pd, exv1_no150_cohort, on='患者序号', how='inner')
                    deep_feat_pd.drop(columns=['class'], inplace=True)

                    radio_intra_features = feature_process(
                        rf'../../dataset/radiomic_feature/{args.feature_fold}/{cohort_name}_features.csv', pid_col_name)
                    radio_intra_features['患者序号'] = radio_intra_features['患者序号'].astype(str)

                    radio_intra_features = pd.merge(radio_intra_features, exv1_no150_cohort, on='患者序号', how='inner')
                    radio_intra_features.drop(columns=['class'], inplace=True)

                    radio_intra_features = tools.scale_on_min_max_test(radio_intra_features, pid_col_name, scaler)
                    test_all['患者序号'] = test_all['患者序号'].astype(str)
                    test_all = pd.merge(test_all, exv1_no150_cohort, on='患者序号', how='inner')
                    labels = test_all[[pid_col_name, label_col_name]]

                    test_data = MyDatasetFusion(pid=deep_feat_pd['患者序号'], deep_features=deep_feat_pd, radio_features=radio_intra_features, labels=labels)
                    test_dataloader[key] = torch.utils.data.DataLoader(test_data, batch_size=1,
                                                                       shuffle=False,
                                                                       drop_last=False)


            if cohort_name == 'SHANGHAI':
                for key in shanghai_renew_cohort.keys():

                    deep_feat_pd = pd.read_csv(r'../../dataset/deep_feature/SHANGHAI_test_features_256.csv')
                    deep_feat_pd['患者序号'] = deep_feat_pd['CT_Name'].apply(lambda x: x.split('_')[-1].split('.')[0])
                    deep_feat_pd = pd.merge(deep_feat_pd, shanghai_renew_cohort[key]['患者序号'].astype(str), on='患者序号', how='inner')


                    radio_intra_features = feature_process(
                        rf'../../dataset/radiomic_feature/{args.feature_fold}/{cohort_name}_features.csv', pid_col_name)
                    radio_intra_features['患者序号'] = radio_intra_features['患者序号'].astype(str)


                    radio_intra_features = tools.scale_on_min_max_test(radio_intra_features, pid_col_name, scaler)
                    radio_intra_features = pd.merge(radio_intra_features, shanghai_renew_cohort[key]['患者序号'].astype(str), on='患者序号', how='inner')

                    test_all['患者序号'] = test_all['患者序号'].astype(str)
                    test = pd.merge(test_all, shanghai_renew_cohort[key]['患者序号'].astype(str), on='患者序号', how='inner')
                    labels = test[[pid_col_name, label_col_name]]

                    test_data = MyDatasetFusion(pid=deep_feat_pd['患者序号'], deep_features=deep_feat_pd, radio_features=radio_intra_features, labels=labels)
                    test_dataloader[key] = torch.utils.data.DataLoader(test_data, batch_size=1,
                                                                       shuffle=False,
                                                                       drop_last=False)


        # Calculate the starting time
        start_time = time.time()

        for lr in lrs:
            args.lr = lr
            for dim_1, dim_2 in zip(dims_1, dims_2):

                    for lay in lays:

                        for actf in actfs:
                            args.actf = actf

                            # Creating model
                            model = mcVAE_BRCA_mut(radio_f_dim=707, deep_f_dim=256, output_dim=128, head_hidden_1=dim_1,
                                                   head_hidden_2=dim_2, head_id=lay, actf=args.actf).cuda(args.device)

                            pretrained_dict = torch.load(
                                rf'D:\Suyang\python\escc_xinfuzhu\model_fusion\ct_mcvae\intra_tumoral\checkpoint\output1\layer_3_epoch_48_bs8_lr3e-05_Tm50_eta_min0_actf_LeakyReLU_dim1_256_dim2_64_train_0.782_val_0.796_test_0.757.pth',
                                map_location='cuda:'+str(args.device))['model']
                            model_dict = model.state_dict()
                            state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
                            model_dict.update(state_dict)
                            model.load_state_dict(model_dict)
                            # for k, v in model.named_parameters():
                            #     if 'head_mut' in k:
                            #         continue
                            #     else:
                            #         v.requires_grad = False

                            total_negative_samples = np.count_nonzero(train_test[label_col_name].values == 0)
                            total_positive_samples = np.count_nonzero(train_test[label_col_name].values == 1)
                            logit_pos_weight = torch.tensor([total_negative_samples / total_positive_samples]).to(device)

                            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                            # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=logit_pos_weight)
                            criterion = torch.nn.BCEWithLogitsLoss()



                            all_epochs_auc_records = []

                            print(rf"layer_{lay}_bs{args.batch_size}_lr{args.lr}_Tm{args.Tmax}_eta_min{args.eta_min}_actf_{args.actf}_dim1_{dim_1}_dim2_{dim_2}")

                            for epoch in range(args.epochs):
                                AUC_list = []
                                epoch_record = {'Epoch': epoch}

                                epoch_record = {'Epoch': epoch}
                                train_loss, train_AUC, train_pids, train_preds, train_trues, train_threshold = validate(train_dataloader, model, criterion, device, 'val', epoch, '')
                                AUC_list.append(train_AUC)
                                epoch_record['train_AUC'] = train_AUC
                                train_prelabel_df = get_prelabel(train_pids, train_preds, train_threshold)
                                train_osdfs_df = pd.merge(train_prelabel_df, RT_survival, on='患者序号', how='inner')
                                # print(KM36(train_osdfs_df))
                                draw_score_violin(train_osdfs_df, title='Distribution of pre_score in train')
                                draw_AUC_confusion_matrix([train_trues], [train_preds], threshold_train=train_threshold,
                                                          cohort_name='Xiehe',
                                                          anno=['Train'])
                                calculate_and_plot_metrics(train_trues, train_preds, train_threshold)


                                _, val_AUC, val_pids, val_preds, val_trues, val_threshold = validate(val_dataloader, model, criterion, device, 'val', epoch, '')
                                AllPred = [int(i >= train_threshold) for i in val_preds]
                                Acc = sum([val_trues[i] == AllPred[i]
                                           for i in range(len(AllPred))]) / len(AllPred)
                                print(rf'val train t acc: {Acc}')
                                AUC_list.append(val_AUC)
                                epoch_record['val_AUC'] = val_AUC
                                val_prelabel_df = get_prelabel(val_pids, val_preds, train_threshold)
                                val_osdfs_df = pd.merge(val_prelabel_df, RT_survival, on='患者序号', how='inner')
                                # print(rf'val_train_{KM36(val_osdfs_df)}')
                                draw_score_violin(val_osdfs_df, title='Distribution of pre_score in internal validation')
                                val_prelabel_df = get_prelabel(val_pids, val_preds, val_threshold)
                                val_osdfs_df = pd.merge(val_prelabel_df, RT_survival, on='患者序号', how='inner')
                                # print(rf'val_val_{KM36(val_osdfs_df)}')
                                draw_AUC_confusion_matrix([val_trues], [val_preds], threshold_train=train_threshold,
                                                          cohort_name='Xiehe',
                                                          anno=['Internal Validation'])
                                calculate_and_plot_metrics(val_trues, val_preds, train_threshold)


                                for key in test_dataloader.keys():
                                    _, test_AUC, test_pids, test_preds, test_trues, test_threshold = validate(test_dataloader[key], model, criterion, device, 'test', epoch, '')
                                    AllPred = [int(i >= train_threshold) for i in test_preds]
                                    Acc = sum([test_trues[i] == AllPred[i]
                                               for i in range(len(AllPred))]) / len(AllPred)
                                    print(rf'test train t acc: {Acc}')
                                    AUC_list.append(test_AUC)
                                    epoch_record['test_AUC'] = test_AUC
                                    test_prelabel_df = get_prelabel(test_pids, test_preds, train_threshold)

                                    if key == 'EXV2':
                                        test_osdfs_df = pd.merge(test_prelabel_df, EXV2_survival, on='患者序号', how='inner')

                                    # draw_score_violin(test_osdfs_df,
                                    #                   title='Distribution of pre_score in external validation')
                                    # test_prelabel_df = get_prelabel(test_pids, test_preds, test_threshold)
                                    # test_osdfs_df = pd.merge(test_prelabel_df, EXV2_survival, on='患者序号', how='inner')
                                    # # print(rf'test_test_{KM36(test_osdfs_df)}')
                                    # draw_AUC_confusion_matrix([test_trues], [test_preds], threshold_train=train_threshold,
                                    #                           cohort_name='Sichuan',
                                    #                           anno=['External Validation'])
                                    # calculate_and_plot_metrics(test_trues, test_preds, train_threshold)
