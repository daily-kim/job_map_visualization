import pandas as pd
import numpy as np


def minmax_scaler(x, scale=(0, 1)):
    min, max = scale
    x = np.array(list(x))
    x_min = x.min()
    x_max = x.max()
    x = (x - x_min) / (x_max - x_min)
    x = x * (max-min) + min
    return x


def subject_mask_by_grade(df, subject_grade):

    if subject_grade == 0:  # all grade
        return df
    mask = (df.subject_grade == subject_grade)
    return df[mask]


def load_data():
    df_job_info = pd.read_csv(
        'data/job_data.csv', encoding='utf-8')
    df_job_major_subject = pd.read_csv(
        'data/job_major_subject.csv', encoding='utf-8')
    df_major_info = pd.read_csv(
        'data/major_info.csv', encoding='utf-8')
    df_subject_info = pd.read_csv('data/subject_info.csv', encoding='utf-8', header=0, names=(
        ['index', 'subject_name', 'description', 'subject_type', 'subject_grade']))
    return df_job_info, df_job_major_subject, df_major_info, df_subject_info


def normalize_dict(d, target=1.0):
    max_val = max(d.values())
    factor = target/max_val
    return {key: value*factor for key, value in d.items()}
