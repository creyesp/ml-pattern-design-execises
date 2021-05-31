import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

from config import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, COLUMNS, TARGET


def load_dataset(file_path):
    data = pd.read_csv(file_path)
    data.dropna(subset=COLUMNS, inplace=True)
    #pd.concat([pd.read_csv(f'/home/creyesp/tmp/tabular/data/{file_}') for file_ in os.listdir('/home/creyesp/tmp/tabular/data/') if file_.endswith('gz')]).reset_index(drop=True)

    dataset = data[COLUMNS]
    target_class = (data[TARGET]==0).astype(int)
    return dataset, target_class


def split_dataset(features, target):
    x_train, x_test, y_train, y_test = \
        train_test_split(features, target, test_size=0.1, stratify=target)
    
    print(f'number of clases: {y_train.value_counts()}')
    print(f'number of clases: {y_test.value_counts()}')
    return x_train, x_test, y_train, y_test


def initial_bias(target):
    pos = (target).sum()
    neg = (target==0).sum()
    initial_bias = np.log([pos/neg])[0]
    return initial_bias


def class_weight(target):
    pos = (target).sum()
    neg = (target==0).sum()
    total = pos + neg
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    return class_weight



def describe_dataset(features, target):
    num_desc = features[NUMERICAL_COLUMNS].apply(lambda x: {'mean': x.mean(), 'variance': x.var()})
    cat_desc = features[CATEGORICAL_COLUMNS].apply(lambda x: {'vocabulary': x.unique()})

    description = {
        'numerical': dict(num_desc),
        'categorical': dict(cat_desc),
        'bias': initial_bias(target),
        'class_weight': class_weight(target),
    }
    return description

def tf_dataset(features, target, train=False):
    ds_train = tf.data.Dataset.from_tensor_slices((dict(features), target))

    if train:
        ds_train = ds_train.shuffle(buffer_size=1_00_000)
    
    ds_train = ds_train.batch(2048, num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds_train

def get_dataset(file_path):
    features, target = load_dataset(file_path)
    x_train, x_test, y_train, y_test = split_dataset(features, target)
    data_description = describe_dataset(x_train, y_train)
    
    ds_train = tf_dataset(x_train, y_train, train=True)
    ds_test = tf_dataset(x_test, y_test, train=False)

    return ds_train, ds_test, data_description