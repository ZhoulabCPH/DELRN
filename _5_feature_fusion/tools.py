# -*- coding: UTF-8 -*-
import json
import os
import shutil
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.preprocessing import scale, MinMaxScaler


def display_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def makedir_ignore(p):
    if not os.path.exists(p):
        os.makedirs(p)


def makedir_delete(p):
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p)


def array2dict(arr1, arr2):
    d = {}
    if isinstance(arr1, str):
        for k, v in zip([arr1], arr2):
            d[k] = v
    else:
        for k, v in zip(arr1, arr2):
            d[k] = v
    return d


def dict_layer_switch(d):
    r = defaultdict(dict)
    for k1, v1 in d.items():
        for k2, v2 in v1.items():
            r[k2][k1] = v2
    return r


def preprocessing(df):
    """
    预处理，去除每一列的空值，并将非数值转化为数值型数据，分两步
    1. 如果本列含有null。
        - 如果是number类型
            如果全为空，则均置零；
            否则，空值的地方取全列平均值。
        - 如果不是number类型
            将空值置NA
    2. 如果本列不是数值型数据，则用label encoder转化为数值型
    :param df: dataframe
    :return: 处理后的dataframe
    """

    def process(c):
        if c.isnull().any().any():
            if np.issubdtype(c.dtype, np.number):
                new_c = c.fillna(c.mean())
                if new_c.isnull().any().any():
                    return pd.Series(np.zeros(c.size))
                return new_c
            else:
                return pd.Series(LabelEncoder().fit_transform(c.fillna("NA").values))
        else:
            if not np.issubdtype(c.dtype, np.number):
                return pd.Series(LabelEncoder().fit_transform(c.values))
        return c

    pre_df = df.copy()
    return pre_df.apply(lambda col: process(col))


def scale_on_feature(data):
    """
    对每一列feature进行归一化，使方差一样

    :param data: dataframe
    :return: 归一化后的dataframe
    """
    data_scale = data.copy()
    data_scale[data.columns] = scale(data_scale)
    return data_scale


def scale_on_min_max_train(data, pid_col_name, feature_range=(0, 1)):
    """
    对每一列feature进行相同区间归一化，使方差一样

    :param data:
    :param feature_range: dataframe
    :return: 归一化后的dataframe
    """

    scaler = MinMaxScaler(feature_range=feature_range)
    features_to_scale = data.columns.difference([pid_col_name])
    df_scaled = data.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(data[features_to_scale])

    return df_scaled, scaler


def scale_on_min_max_test(data, pid_col_name, scaler):
    """
    对每一列feature进行相同区间归一化，使方差一样

    :param data:
    :param feature_range: dataframe
    :return: 归一化后的dataframe
    """


    features_to_scale = data.columns.difference([pid_col_name])
    df_scaled = data.copy()
    df_scaled[features_to_scale] = scaler.transform(data[features_to_scale])

    return df_scaled


def prepare_target(target, nb=None, map_dict=None, method="map"):
    prepared = np.array(target).copy()

    is_numeric = np.issubdtype(prepared.dtype, np.number)

    if method == "map":
        # map labels to other labels
        prepared = np.array(map(lambda x: map_dict[x], list(prepared)))
    elif method == "size":
        try:
            cutted = pd.qcut(prepared, nb)
            prepared = cutted.code
        except Exception:
            return -1
    elif method == "range":
        cutted = pd.cut(prepared, nb)
    else:
        raise Exception

    return prepared


# prepare_target([1,1,1,11.1], map_dict={1: 111, 11.1:"p"})
# prepare_target([1,1,2,3,3,3], nb=2, method='size')

def encode_l(label):
    le = LabelEncoder().fit(label)
    el = le.transform(label)
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    return le, el, mapping


def encode_b(label):
    le = LabelBinarizer().fit(label)
    el = le.transform(label)
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    return le, el, mapping


def save_json(content, file_path):
    with open(file_path, 'w') as f:
        json.dump(content, f, sort_keys=True, indent=4)


def load_json(file_path):
    data = None
    with open(file_path) as f:
        data = json.load(f)
    return data


def reverse_dict(my_map):
    if sys.version_info[0] == 2:
        inv_map = {v: k for k, v in my_map.iteritems()}
    else:
        inv_map = {v: k for k, v in my_map.items()}
    return inv_map


def prepare_feature_n_label(df_feature, df_label, tags=None, key="mask"):
    # drop NA
    merged = pd.merge(df_feature, df_label.dropna()[['mask', 'label']], on=key)
    if tags is not None:
        merged = pd.merge(merged, tags)
        tv_label = list(set(merged[merged.dataset == 0].label.tolist()))
        merged = merged[merged.label.isin(tv_label)]
    feature = merged[[x for x in merged.columns if x not in ["label", "dataset"]]]
    label = merged[["label"]]
    if tags is not None:
        new_tags = merged[["dataset"]]
        return feature, label, new_tags
    return feature, label


def choose_feature(feature_file, use_pyradiomics=True):
    feature_classes = ['glcm',
                       'gldm',
                       'glrlm',
                       'glszm',
                       'ngtdm',
                       'shape',
                       'firstorder']

    if not use_pyradiomics:
        feature_classes = [
            "glcm",
            "glrlm",
            "shape",
            "firstorder"
        ]

    df = pd.read_csv(feature_file)
    columns = df.columns
    valid_columns = ['image', 'mask'] + [x for x in columns if len([y for y in feature_classes if y in x[:len(y)]]) > 0]
    return df[valid_columns]
