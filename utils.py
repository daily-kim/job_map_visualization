import pandas as pd
from gensim import models

ko_model = models.fasttext.load_facebook_model('data/cc.ko.300.bin.gz')

df_job_info = pd.read_csv(
    'data/job_data.csv', encoding='utf-8')
df_job_major_subject = pd.read_csv(
    'data/job_major_subject.csv', encoding='utf-8')
df_major_info = pd.read_csv(
    'data/major_info.csv', encoding='utf-8')
df_subject_info = pd.read_csv('data/subject_info.csv', encoding='utf-8', header=0, names=(
    ['index', 'subject_name', 'description', 'subject_type', 'subject_grade']))
# predefined common_subjects(1st grade)
COMMON_SUBJECTS = ", 국어, 통합과학, 과학탐구실험, 한국사, 수학, 영어, 통합사회"


def similarity_function(method, text1, text2):
    # sourcery skip: merge-comparisons, merge-duplicate-blocks, remove-redundant-if
    if method == 'fasttext':
        return ko_model.wv.n_similarity(text1.split(), text2.split())
    elif method == 'SBERT':
        return ko_model.wv.n_similarity(text1.split(), text2.split())
    else:
        print('No method')


def job_major_subject_matching(job_name):
    # job-major matching
    majorlist = list(
        df_job_major_subject[df_job_major_subject['job'] == job_name].major.unique())
    job_major = {job_name: majorlist}

    # major-subject matching
    subjects = df_job_major_subject[df_job_major_subject['job'] == job_name]
    subject_details = subjects.groupby(
        'major', group_keys=False).subject_details.apply(lambda x: ','.join(x))
    # add predefined common subjects(1st grade)
    for idx, line in enumerate(subject_details):
        subject_details[idx] = line + COMMON_SUBJECTS

    major_subject = dict()
    job_subject = dict()

    for idx, line in enumerate(subject_details):
        tmp = line.replace(':', ',')
        tmp = tmp.split(',')
        # '~교과'로 되어 있는 노이즈 데이터 제거
        for e in tmp:
            if '교과' in e:
                tmp.remove(e)
        tmp = [i.replace(' ', '') for i in tmp]

        # '수학II'가 아니라 그냥 'II'라고 되어 있는 과목명을 바로 직전의 과목명을 참고하여 '수학II'로 변경
        for e in tmp:
            if len(e) == 1:
                i = tmp.index(e)
                tmp[i] = tmp[i-1][:-1]+e

        major_subject[subject_details.keys()[idx]] = set(tmp)

        subject_set = set()
        for value in major_subject.values():
            for v in value:
                subject_set.add(v)

        job_subject = {job_name: list(subject_set)}

    return job_major, major_subject, job_subject
# from job_major, get major list and calculate similarity between job and major


def get_job_major_similarity(df_job_info, df_major_info, job_major, sim_method):
    job_major_sim = dict()
    for key, value in job_major.items():
        for v in value:
            # get major description
            major_desc = df_major_info[df_major_info['major']
                                       == v].major_summary.values[0]
        # get job description
            job_desc = df_job_info[df_job_info['job']
                                   == key].job_summary.values[0]
        # calculate similarity between job description and major description
            job_major_sim[v] = similarity_function(
                sim_method, major_desc, job_desc)
    return job_major_sim
# from major_subject, get subject list and calculate similarity between major description and subject description


def get_major_subject_similarity(df_major_info, df_subject_info, major_subject, sim_method):
    major_subject_sim = dict()

    for key, value in major_subject.items():
        one_major_sim = dict()
        for v in value:
            # get major description
            major_desc = df_major_info[df_major_info['major']
                                       == key].major_summary.values[0]
        # get subject description(try catch)
            try:
                tmp_df_subject_info = df_subject_info.replace(
                    ' ', '', regex=True)
                subject_desc = df_subject_info[tmp_df_subject_info['subject_name']
                                               == v].description.values[0]
            # calculate similarity between major description and subject description
                one_major_sim[v] = similarity_function(
                    sim_method, major_desc, subject_desc)
            except Exception:
                continue
        major_subject_sim[key] = one_major_sim
    return major_subject_sim


def get_job_subject_similarity_1(df_job_info, df_subject_info, job_subject, sim_method):
    job_subject_sim = dict()
    for key, value in job_subject.items():
        for v in value:
            # get major description
            job_desc = df_job_info[df_job_info['job']
                                   == key].job_summary.values[0]
        # get subject description(try catch)
            try:
                tmp_df_subject_info = df_subject_info.replace(
                    ' ', '', regex=True)
                subject_desc = df_subject_info[tmp_df_subject_info['subject_name']
                                               == v].description.values[0]
            # calculate similarity between major description and subject description
                job_subject_sim[v] = similarity_function(
                    sim_method, job_desc, subject_desc)
            except Exception:
                continue
    return job_subject_sim


def get_job_subject_similarity_2(job_major_sim, major_subject_sim):
    job_subject_sim_2 = dict()
    for major, majorsim in job_major_sim.items():
        for subject, subjectsim in major_subject_sim.get(major).items():
            try:
                job_subject_sim_2[subject] += majorsim*subjectsim
            except Exception:
                job_subject_sim_2[subject] = majorsim*subjectsim
    return job_subject_sim_2


def get_subject_subject_similarity(job_subject_sim, df_subject_info, threshold, sim_method):
    subject_list = list(job_subject_sim.keys())
    num_subject = len(subject_list)
    subject_dict = dict()

    tmp_df_subject_info = df_subject_info.replace(' ', '', regex=True)

    for idx1, subject1 in enumerate(subject_list):
        subject1_desc = df_subject_info[tmp_df_subject_info['subject_name']
                                        == subject1].description.values[0]
        for idx2, subject2 in enumerate(subject_list):
            if idx1 >= idx2:
                continue
            subject2_desc = df_subject_info[tmp_df_subject_info['subject_name']
                                            == subject2].description.values[0]
            similarity = similarity_function(
                sim_method, subject1_desc, subject2_desc)
            if similarity > threshold:
                subject_dict[(subject1, subject2)] = similarity
    return subject_dict


def subject_mask(df, subject_grade):

    if subject_grade is None:
        return df
    mask = (df.subject_grade == subject_grade)
    return df[mask]


def similarity_grade(job_name, sim_method='fasttext', threshold_subject=0.98, grade=3):
    # print(df_job_info)
    df_subject_info_masked = subject_mask(df_subject_info, grade)
    job_major, major_subject, job_subject = job_major_subject_matching(
        job_name)
    job_major_sim = get_job_major_similarity(
        df_job_info, df_major_info, job_major, sim_method)
    major_subject_sim = get_major_subject_similarity(
        df_major_info, df_subject_info_masked, major_subject, sim_method)
    job_subject_sim_1 = get_job_subject_similarity_1(
        df_job_info, df_subject_info_masked, job_subject, sim_method)
    job_subject_sim_2 = get_job_subject_similarity_2(
        job_major_sim, major_subject_sim)
    subject_subject_sim = get_subject_subject_similarity(
        job_subject_sim_1, df_subject_info_masked, threshold_subject, sim_method)

    return job_major_sim, major_subject_sim, job_subject_sim_1, job_subject_sim_2, subject_subject_sim


if __name__ == '__main__':
    job_major_sim, major_subject_sim, job_subject_sim_1, job_subject_sim_2, subject_subject_sim = similarity_grade(
        '통계학연구원', grade=1)

    print(job_subject_sim_2)
