from sklearn import model_selection
import h5py
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import shutil
import gc

# region selecting relevant patients from raw data
#############SKIP THIS BLOCK###########SKIP THIS BLOCK#############SKIP THIS BLOCK#############################################
data_info_path = '//nmbu.no/LargeFile/Project/REALTEK-HeadNeck-Project/DEPREDICT/AMC-data-sharing/AMC-data-sharing/EMBARC/'
anat_file = 'anat-dataframe.pkl'
dwi_file = 'dwi-dataframe.pkl'

dwi_df = pd.read_pickle(data_info_path + dwi_file)
anat_df = pd.read_pickle(data_info_path + anat_file)
include_patients = os.listdir('W:/embarc/resampled_nii')
# check missing data
np.all(dwi_df[dwi_df.index.isin(include_patients)][dwi_df.columns[[0,1,2,3,4,5,7]]].values == anat_df[anat_df.index.isin(include_patients)][anat_df.columns[[0,1,2,3,4,5,7]]].values)

dwi_df[dwi_df.index.isin(include_patients)][dwi_df.columns[[0,1,2,3,4,5,7]]].to_csv('W:/embarc/dataset/info.csv')
##################################################################################################################################
# endregion endraw2selected

# region process categorical columns
#############SKIP THIS BLOCK###########SKIP THIS BLOCK#############SKIP THIS BLOCK#############################################
# load data
patient_info = pd.read_csv('W:/embarc/dataset/info.csv', index_col=0)
# create new dataframe with w8_responder
new_df = pd.DataFrame(patient_info['w8_responder'].astype(float), index=patient_info.index)
# add stage1tx column
new_df['Stage1TX'] = np.where(patient_info['Stage1TX'] == 'PLA', 0., 1.)
# add gender column
new_df['gender'] = np.where(patient_info['gender'] == 'Male', 0., 1.)
# add age column
new_df['age'] = patient_info['age']
# add race column
dummy_coded_df = pd.get_dummies(patient_info, columns=['race'])
new_df['race_white'] = dummy_coded_df.race_White
new_df['race_black_or_african_ame'] = dummy_coded_df['race_Black or African Ame']
new_df['race_asian'] = dummy_coded_df.race_Asian
# add hispanic column
new_df['hispanic'] = np.where(patient_info['hispanic'] == 'Hisp-No', 0., 1.)

new_df['w0_score_17'] = patient_info['w0_score_17']

new_df.to_csv('W:/embarc/dataset/patient_info.csv')
###############################################################################################################################
# endregion

# region splitting data into 4 folds
#############SKIP THIS BLOCK###########SKIP THIS BLOCK#############SKIP THIS BLOCK#############################################
patient_info = pd.read_csv('W:/embarc/dataset/patient_info.csv', index_col=0)

ser_df = patient_info[patient_info.Stage1TX == 1].copy()
pla_df = patient_info[patient_info.Stage1TX == 0].copy()

skf = model_selection.StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
folds = []
for train_index, test_index in skf.split(ser_df.values[:, 1:], ser_df.values[:, 0]):
    folds.append(test_index)

folds

for fold in folds:
    print(ser_df.w8_responder[fold].value_counts())
    print(ser_df.index[fold])
    print(ser_df['fold'][fold])

ser_fold_index = np.zeros(117)
for i in range(4):
    ser_fold_index[folds[i]] = i
ser_df['fold'] = ser_fold_index
ser_df.to_csv('W:/embarc/dataset/ser_info.csv')


skf = model_selection.StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
folds = []
for train_index, test_index in skf.split(pla_df.values[:, 1:], pla_df.values[:, 0]):
    folds.append(test_index)

pla_fold_index = np.zeros(125)
for i in range(4):
    pla_fold_index[folds[i]] = i
pla_df['fold'] = pla_fold_index

for fold in folds:
    print(pla_df.w8_responder[fold].value_counts())
    print(pla_df.index[fold])
    print(pla_df['fold'][fold])

pla_df.to_csv('W:/embarc/dataset/placebo_info.csv')

#######################################################################################################################################
# endregion

# region creating h5py files

# load csv files
ser_df = pd.read_csv('W:/embarc/dataset/ser_info.csv', index_col=0)
# first create template
with h5py.File('W:/embarc/dataset/ser_data.h5', 'w') as f:
    f.attrs.create('target', ser_df.columns[0])
    f.attrs.create('demographic_info', ','.join(ser_df.columns[2:7]))
    f.attrs.create('clinical_info', ','.join(ser_df.columns[7:8]))
    f.attrs.create('score_info', ','.join(ser_df.columns[8:9]))
    for i in range(4):
        group = f.create_group(f'fold_{i}')
        selected_df = ser_df[ser_df.fold==i]
        group.create_dataset('patient_idx', data=selected_df.index.values)
        group.create_dataset('target', data=selected_df.values[:, 0], dtype='f4')
        group.create_dataset('demographic_info', data=selected_df.values[:, 2:7], dtype='f4')
        group.create_dataset('clinical_info', data=selected_df.values[:, 7:8], dtype='f4')
        group.create_dataset('score_info', data=selected_df.values[:, 8:9], dtype='f4')

# then save image
for i in range(4):
    print('fold', i)
    gc.collect()
    with h5py.File('W:/embarc/dataset/ser_data.h5', 'a') as f:
        group = f[f'fold_{i}']
        patient_idx = group['patient_idx'][:].astype(str)
        data1 = []
        data2 = []
        for pid in patient_idx:
            print(pid)
            img = np.load(f'W:/embarc/normalized_images/{str(pid)}.npy')
            shape = (224,224,160, 3)
            if img.shape != shape:
                pad_width = [(0, shape[i] - img.shape[i]) for i in range(4)]
                for i in range(4):
                    low, high = pad_width[i]
                    if high > 0:
                        pad_width[i] = (1, high-1)
                print('padding', pad_width)
                img = np.pad(img, pad_width)
            data1.append(img[..., :2])
            data2.append(img[..., [2]])
        print('saving...')
        group.create_dataset('ses_1', data = np.stack(data1), dtype='f4', chunks=(1, 224, 224, 160, 1), compression='lzf')
        group.create_dataset('ses_2', data = np.stack(data2), dtype='f4', chunks=(1, 224, 224, 160, 1), compression='lzf')
        

# do the same for 
# load csv files
pla_df = pd.read_csv('W:/embarc/dataset/placebo_info.csv', index_col=0)
# first create template
with h5py.File('W:/embarc/dataset/placebo_data.h5', 'w') as f:
    f.attrs.create('target', pla_df.columns[0])
    f.attrs.create('demographic_info', ','.join(pla_df.columns[2:7]))
    f.attrs.create('clinical_info', ','.join(pla_df.columns[7:8]))
    f.attrs.create('score_info', ','.join(pla_df.columns[8:9]))
    for i in range(4):
        group = f.create_group(f'fold_{i}')
        selected_df = pla_df[pla_df.fold==i]
        group.create_dataset('patient_idx', data=selected_df.index.values)
        group.create_dataset('target', data=selected_df.values[:, 0], dtype='f4')
        group.create_dataset('demographic_info', data=selected_df.values[:, 2:7], dtype='f4')
        group.create_dataset('clinical_info', data=selected_df.values[:, 7:8], dtype='f4')
        group.create_dataset('score_info', data=selected_df.values[:, 8:9], dtype='f4')

# then save image
for i in range(4):
    print('fold', i)
    gc.collect()
    with h5py.File('W:/embarc/dataset/placebo_data.h5', 'a') as f:
        group = f[f'fold_{i}']
        patient_idx = group['patient_idx'][:].astype(str)
        data1 = []
        data2 = []
        for pid in patient_idx:
            print(pid)
            img = np.load(f'W:/embarc/normalized_images/{str(pid)}.npy')
            shape = (224,224,160, 3)
            if img.shape != shape:
                pad_width = [(0, shape[i] - img.shape[i]) for i in range(4)]
                for i in range(4):
                    low, high = pad_width[i]
                    if high > 0:
                        pad_width[i] = (1, high-1)
                print('padding', pad_width)
                img = np.pad(img, pad_width)
            data1.append(img[..., :2])
            data2.append(img[..., [2]])
        print('saving...')
        group.create_dataset('ses_1', data = np.stack(data1), dtype='f4', chunks=(1, 224, 224, 160, 1), compression='lzf')
        group.create_dataset('ses_2', data = np.stack(data2), dtype='f4', chunks=(1, 224, 224, 160, 1), compression='lzf')

# endregion
