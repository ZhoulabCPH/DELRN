import SimpleITK as sitk
import radiomics.featureextractor as FEE
import os
import numpy as np
import pandas as pd

def feature_extract(img, mask, params_path, huCut):
    if huCut:
        print('HU CUT')
        img = sitk.IntensityWindowing(img, windowMinimum=-125, windowMaximum=225,
                                             outputMinimum=-125, outputMaximum=225)
    # 使用配置文件初始化特征抽取器
    extractor = FEE.RadiomicsFeatureExtractor(params_path)

    # 进行特征提取
    result = extractor.execute(img, mask)
    row = []
    row_next = []
    for idx, (key, val) in enumerate(result.items()):
        if idx<11:
            continue
        if not isinstance(val,(float,int,np.ndarray)):
            continue
        if np.isnan(val):
            val=0
        row.append(key)
        row_next.append(val)
    return row, row_next


def process_main(pids, img_path, img_root, mask_root, params_path, save_path, huCut, file_prefix):
    if not os.path.exists(save_path.rsplit('/', 1)[0]):
        # If it doesn't exist, create it
        os.makedirs(save_path.rsplit('/', 1)[0])
    error = []
    df = pd.DataFrame()
    for i, (pid, name) in enumerate(zip(pids, img_path)):
        patientNo = name.split('\\')[-1].split('.')[0].split('_')[-1]
        print(patientNo)
        if int(pid) != int(patientNo):
            print('pid error!')
            break

        print(i)
        img = sitk.ReadImage(img_root + '\\' + file_prefix + str(patientNo) + '.nii.gz')
        mask = sitk.ReadImage(mask_root + '\\' + file_prefix + str(patientNo) + '.nii.gz')
        try:
            row, row_next = feature_extract(img, mask, params_path, huCut)
            # 首先将'患者序号'添加到row列表的开始
            row.insert(0, '患者序号')
            row_next.insert(0, patientNo)  # 将索引i作为患者序号插入到row_next的开始

            # 如果DataFrame为空，则初始化它；否则添加新行
            if df.empty:
                df = pd.DataFrame([row_next], columns=row)  # 使用列表的列表来创建DataFrame
            else:
                temp_df = pd.DataFrame([row_next], columns=row)  # 创建一个临时DataFrame来存储新行
                df = pd.concat([df, temp_df], ignore_index=True)  # 将新行添加到df中
        except:
            error.append(patientNo)
    print('错误')
    print(error)
    df.to_csv(save_path, encoding='utf_8_sig', index=False)

def read_and_merge_data(label_file, patient_id_col='', label_col='', img_path_col=''):
    label_data = pd.read_csv(label_file)[[patient_id_col, label_col]]
    label_data[img_path_col] = label_data[patient_id_col].apply(lambda x: f"{str(x)}.nii.gz")
    return label_data



if __name__ == '__main__':
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    train_cohort_name = ''
    label_path_rt = ''
    datasets_val = ['']
    label_path_val = ['']
    img_dir = r''
    file_prefix = r''
    test_dataloader = {}
    pid_col_name = ''
    label_col_name = ''
    img_path_col = ''
    OS_time_col_name = ''
    OS_State_col_name = ''

    # 队列划分
    rt = pd.read_csv(r'')
    rt_cohort = {'RT': rt}

    for label_path, cohort_name in zip(label_path_val, datasets_val):
        print(label_path)
        test_all = read_and_merge_data(rf'')

        feat_save_path = rf''
        if not os.path.exists(os.path.join(feat_save_path)):
            os.makedirs(os.path.join(feat_save_path))

        if cohort_name == '':
            for key in rt_cohort.keys():
                print(key)
                exv2_test = pd.merge(test_all, rt_cohort[key], how='inner', on=pid_col_name)
                exv2_test = exv2_test.reset_index(drop=True)
                process_main(exv2_test[pid_col_name].values, exv2_test[img_path_col].values,
                             r'{}\{}\IMG'.format(img_dir, cohort_name),
                             r'{}\{}\MASK'.format(img_dir, cohort_name),
                             './extraction_params.yaml', '{}/{}_features.csv'.format(feat_save_path, key), huCut=True, file_prefix=file_prefix)








