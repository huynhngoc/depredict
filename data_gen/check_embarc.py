import h5py
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import shutil

data_info_path = '//nmbu.no/LargeFile/Project/REALTEK-HeadNeck-Project/DEPREDICT/AMC-data-sharing/AMC-data-sharing/EMBARC/'
anat_file = 'anat-dataframe.pkl'
dwi_file = 'dwi-dataframe.pkl'

dwi_df = pd.read_pickle(data_info_path + dwi_file)

dwi_df.columns[:10]
dwi_df.columns[10:20]

dwi_df[dwi_df.columns[:10]]

dwi_patient = dwi_df[dwi_df.columns[:10]][dwi_df.Stage1TX=='SER'].index
dwi_patient

anat_df = pd.read_pickle(data_info_path + anat_file)

anat_df.columns[:10]
anat_df.columns[10:20]

anat_df[anat_df.columns[:10]]

anat_patient = anat_df[anat_df.columns[:10]][anat_df.Stage1TX=='SER'].index
anat_patient

# TODO: RESAMPLE all DWI and anat images --> same origin (voxel spacing and size are the same)

img_folder = "W:/data_bids4/derivatives/sub-UM0080/ses-1/"
dwi_img_file = 'dwi/sub-UM0080_ses-1_FAWarped.nii.gz'
brain_seg_file = 'anat/sub-UM0080_ses-1_aparc+aseg.nii.gz'

dwi_img = sitk.ReadImage(img_folder + dwi_img_file)
brain_img = sitk.ReadImage(img_folder + brain_seg_file)


dwi_img_data = sitk.GetArrayFromImage(dwi_img)
brain_img_data = sitk.GetArrayFromImage(brain_img)
brain_img_data = brain_img_data.astype(float)
brain_img_data[brain_img_data == 0] = np.nan
plt.imshow(dwi_img_data[127], 'gray')
plt.imshow(brain_img_data[127], 'tab20', alpha=0.2)
plt.show()
# does not seems right
# try some resampling --> MUST HAVE
new_dwi_img = sitk.Resample(dwi_img, brain_img, sitk.Transform(), sitk.sitkLinear, 0, dwi_img.GetPixelID())

new_dwi_img_data = sitk.GetArrayFromImage(new_dwi_img)

plt.subplot(1,2,1)
plt.imshow(dwi_img_data[127], 'gray')
plt.imshow(brain_img_data[127], 'tab20', alpha=0.2)
plt.subplot(1,2,2)
plt.imshow(new_dwi_img_data[127], 'gray')
plt.imshow(brain_img_data[127], 'tab20', alpha=0.2)
plt.show()


data_folder = "W:/data_bids4/derivatives/"
dwi_img_filename = '{pid}/ses-1/dwi/{pid}_ses-1_FAWarped.nii.gz'
anat_img_filename = '{pid}/ses-1/anat/{pid}_ses-1_aparc+aseg.nii.gz'

np.all(dwi_patient == anat_patient) # True
for pid in dwi_patient:
    print(pid)
    if not os.path.exists(data_folder + dwi_img_filename.format(pid=pid)):
        print(pid, 'missing dwi')
        continue
    if not os.path.exists(data_folder + anat_img_filename.format(pid=pid)):
        print(pid, 'missing anat')
        continue
    dwi_image = sitk.ReadImage(data_folder + dwi_img_filename.format(pid=pid))
    anat_image = sitk.ReadImage(data_folder + anat_img_filename.format(pid=pid))

    if not np.all(anat_image.GetSpacing() == dwi_image.GetSpacing()):
        print(pid, 'Needs spacing resample')
    if not np.all(anat_image.GetSize() == dwi_image.GetSize()):
        print(pid, 'Needs shape resample')
    if np.all(anat_image.GetOrigin() == dwi_image.GetOrigin()):
        print(pid, 'does not need origin resample')

ser_exclude_patients = []
for pid in dwi_patient:
    print(pid)
    if not os.path.exists(data_folder + dwi_img_filename.format(pid=pid)):
        print(pid, 'missing dwi')
        ser_exclude_patients.append(pid)
    if not os.path.exists(data_folder + anat_img_filename.format(pid=pid)):
        print(pid, 'missing anat')
        ser_exclude_patients.append(pid)
    

ser_patients = [pid for pid in dwi_patient if pid not in ser_exclude_patients]


exclude_patients = []
dwi_img_filename = '{pid}/ses-{ses}/dwi/{pid}_ses-{ses}_FAWarped.nii.gz'
anat_img_filename = '{pid}/ses-{ses}/anat/{pid}_ses-{ses}_brain.nii.gz'
seg_img_filename = '{pid}/ses-{ses}/anat/{pid}_ses-{ses}_aparc+aseg.nii.gz'
for pid in dwi_df.index:
    print(pid)
    if not os.path.exists(data_folder + dwi_img_filename.format(pid=pid, ses=1)):
        print(pid, 'missing dwi ses 1')
        exclude_patients.append(pid)
    if not os.path.exists(data_folder + anat_img_filename.format(pid=pid, ses=1)):
        print(pid, 'missing anat ses 1')
        exclude_patients.append(pid)
    if not os.path.exists(data_folder + anat_img_filename.format(pid=pid, ses=2)):
        print(pid, 'missing anat ses 2')
        exclude_patients.append(pid)

include_patients = [pid for pid in dwi_df.index if pid not in exclude_patients]
len(include_patients) # 242
for pid in include_patients:
    print(pid)
    dwi_image = sitk.ReadImage(data_folder + dwi_img_filename.format(pid=pid, ses=1))
    anat_image1 = sitk.ReadImage(data_folder + anat_img_filename.format(pid=pid, ses=1))
    anat_image2 = sitk.ReadImage(data_folder + anat_img_filename.format(pid=pid, ses=2))

    if not np.all(anat_image1.GetSpacing() == dwi_image.GetSpacing()):
        print(pid, 'Needs spacing resample')
    if not np.all(anat_image1.GetSize() == dwi_image.GetSize()):
        print(pid, 'Needs shape resample')
    if np.all(anat_image1.GetOrigin() == dwi_image.GetOrigin()):
        print(pid, 'does not need origin resample')

    if not np.all(anat_image1.GetSpacing() == anat_image2.GetSpacing()):
        print(pid, 'Needs spacing resample  anat2')
    if not np.all(anat_image1.GetSize() == anat_image2.GetSize()):
        print(pid, 'Needs shape resample anat2')
    if np.all(anat_image1.GetOrigin() == anat_image2.GetOrigin()):
        print(pid, ' does not need origin resample anat2')

saved_image_folder = 'W:/embarc/resampled_images/'
for pid in include_patients:
    print(pid)
    dwi_image = sitk.ReadImage(data_folder + dwi_img_filename.format(pid=pid, ses=1))
    anat_image1 = sitk.ReadImage(data_folder + anat_img_filename.format(pid=pid, ses=1))
    anat_image2 = sitk.ReadImage(data_folder + anat_img_filename.format(pid=pid, ses=2))

    print('resampling...')
    new_dwi_image = sitk.Resample(dwi_img, anat_image1, sitk.Transform(), sitk.sitkLinear, 0, dwi_image.GetPixelID())
    new_anat_image2 = sitk.Resample(anat_image2, anat_image1, sitk.Transform(), sitk.sitkLinear, 0, anat_image2.GetPixelID())

    print('saving...')
    data = np.stack([
        sitk.GetArrayFromImage(new_dwi_image),
        sitk.GetArrayFromImage(anat_image1),
        sitk.GetArrayFromImage(new_anat_image2),
        ], axis=-1)

    np.save(saved_image_folder + pid + '.npy', data)

nii_image_folder = 'W:/embarc/resampled_nii/'
dwi_img_filename = '{pid}/ses-{ses}/dwi/{pid}_ses-{ses}_FAWarped.nii.gz'
anat_img_filename = '{pid}/ses-{ses}/anat/{pid}_ses-{ses}_brain.nii.gz'
seg_img_filename = '{pid}/ses-{ses}/anat/{pid}_ses-{ses}_aparc+aseg.nii.gz'
for pid in include_patients[-1:]:
    print(pid)
    ref_image = sitk.ReadImage(data_folder + seg_img_filename.format(pid=pid, ses=1))
    dwi_image = sitk.ReadImage(data_folder + dwi_img_filename.format(pid=pid, ses=1))
    anat_image1 = sitk.ReadImage(data_folder + anat_img_filename.format(pid=pid, ses=1))
    anat_image2 = sitk.ReadImage(data_folder + anat_img_filename.format(pid=pid, ses=2))

    print('resampling...')
    new_dwi_image = sitk.Resample(dwi_img, ref_image, sitk.Transform(), sitk.sitkBSpline, 0, dwi_image.GetPixelID())
    new_anat_image1 = sitk.Resample(anat_image1, ref_image, sitk.Transform(), sitk.sitkBSpline, 0, anat_image1.GetPixelID())
    new_anat_image2 = sitk.Resample(anat_image2, ref_image, sitk.Transform(), sitk.sitkBSpline, 0, anat_image2.GetPixelID())

    print('saving...')
    if not os.path.exists(nii_image_folder + pid):
        os.makedirs(nii_image_folder + pid)

    sitk.WriteImage(new_dwi_image, nii_image_folder + pid + '/dwi.nii.gz')
    sitk.WriteImage(anat_image1, nii_image_folder + pid + '/anat1.nii.gz')
    sitk.WriteImage(new_anat_image2, nii_image_folder + pid + '/anat2.nii.gz')


sitk.sitkBSpline

data = np.load(saved_image_folder + include_patients[0] + '.npy')


(data[...,1] - data[...,2]).max()

plt.imshow(data[120][...,1]);plt.show()
plt.imshow(data[120][...,2]);plt.show()
import gc

big_pid = []
saved_sample_images = 'W:/embarc/check_direction/'
for pid in include_patients[147:]:
    gc.collect()
    print(pid)
    ref_image = sitk.ReadImage(data_folder + seg_img_filename.format(pid=pid, ses=1))
    data = sitk.GetArrayFromImage(ref_image)

    color_map = list(np.unique(data.flatten()))
    new_data = np.zeros((256,256,256))
    for color in color_map:
        # print(color)
        new_data[data==color] = color_map.index(color)

    bool_data = (data > 0).astype(int)
    ax0 = bool_data.sum(axis=(1,2))
    ax1 = bool_data.sum(axis=(0,2))
    ax2 = bool_data.sum(axis=(0,1))

    ax0_min, ax0_max = np.argwhere(ax0 > 0).flatten().min(), np.argwhere(ax0 > 0).flatten().max()
    ax0_start, ax0_end = (ax0_min + ax0_max) // 2 - 112, (ax0_min + ax0_max) // 2 + 112
    if ax0_start < 0:
        ax0_start, ax0_end = 0, 224

    ax1_min, ax1_max = np.argwhere(ax1 > 0).flatten().min(), np.argwhere(ax1 > 0).flatten().max()
    ax1_start, ax1_end = (ax1_min + ax1_max) // 2 - 112, (ax1_min + ax1_max) // 2 + 112
    if ax1_start < 0:
        ax1_start, ax1_end = 0, 224

    ax2_min, ax2_max = np.argwhere(ax2 > 0).flatten().min(), np.argwhere(ax2 > 0).flatten().max()
    ax2_start, ax2_end = (ax2_min + ax2_max) // 2 - 80, (ax2_min + ax2_max) // 2 + 80

    final_data = new_data[ax0_start:ax0_end, ax1_start:ax1_end, ax2_start:ax2_end]
    if ax0_min <= ax0_start or ax0_max >= ax0_end or ax1_min <= ax1_start or ax1_max >= ax1_end or ax1_min <= ax1_start or ax1_max >= ax1_end:
        print(ax0_min, ax0_max, ax0_start, ax0_end)
        print(ax1_min, ax1_max, ax1_start, ax1_end)
        print(ax2_min, ax2_max, ax2_start, ax2_end)
        big_pid.append(pid)
    if final_data[0].sum() > 0 or final_data[-1].sum() > 0 or final_data[:, 0].sum() > 0 or final_data[:,-1].sum() > 0 or final_data[...,0].sum() > 0 or final_data[...,-1].sum() > 0:
        print(ax0_min, ax0_max, ax0_start, ax0_end)
        print(ax1_min, ax1_max, ax1_start, ax1_end)
        print(ax2_min, ax2_max, ax2_start, ax2_end)
        print(final_data[0].sum() > 0, final_data[-1].sum() > 0, final_data[:, 0].sum() > 0, final_data[:,-1].sum() > 0, final_data[...,0].sum() > 0, final_data[...,-1].sum() > 0)
        big_pid.append(pid)

    # plt.imshow(final_data[88], 'tab20c');plt.savefig(saved_sample_images + 'ax0/' + pid + '.png');
    # plt.imshow(final_data[:,80], 'tab20');plt.savefig(saved_sample_images + 'ax1/' + pid + '.png');
    # plt.imshow(final_data[:,:,80], 'tab20');plt.savefig(saved_sample_images + 'ax2/' + pid + '.png');
    # plt.close('all');


include_patients = os.listdir(nii_image_folder)

bool_data = (data > 0).astype(int)
ax0 = bool_data.sum(axis=(1,2))
np.argwhere(ax0 > 0).flatten()[0]
np.argwhere(ax0 > 0).flatten()[-1]


# wrong_spacing = []
# for pid in include_patients:
#     gc.collect()
#     print(pid)
#     ref_image = sitk.ReadImage(data_folder + seg_img_filename.format(pid=pid, ses=1))
#     spacing = ref_image.GetSpacing()
#     if not np.all((1.0, 1.0, 1.0) == spacing):
#         print(pid, spacing)
#         wrong_spacing.append(pid)


nii_folder = 'W:/embarc/resampled_nii/'
saved_folder = 'W:/embarc/resampled_images/'

include_patients = os.listdir(nii_folder)

for pid in include_patients[120:]:
    print(pid)
    dwi = sitk.GetArrayFromImage(sitk.ReadImage(nii_folder+pid+'/dwi.nii.gz'))
    anat1 = sitk.GetArrayFromImage(sitk.ReadImage(nii_folder+pid+'/anat1.nii.gz'))
    anat2 = sitk.GetArrayFromImage(sitk.ReadImage(nii_folder+pid+'/anat2.nii.gz'))
    data = np.stack([dwi, anat1, anat2], axis=-1)
    np.save(saved_folder + pid + '.npy', data)

double_check_pid = []
check_pid = []
for pid in include_patients[120:]:
    gc.collect()
    print(pid)
    ref_image = sitk.ReadImage(f'W:/data_bids4/derivatives/{pid}/ses-1/anat/{pid}_ses-1_aparc+aseg.nii.gz')
    data = sitk.GetArrayFromImage(ref_image)
    bool_data = (data > 0).astype(int)
    ax0 = bool_data.sum(axis=(1,2))
    ax1 = bool_data.sum(axis=(0,2))
    ax2 = bool_data.sum(axis=(0,1))
    ax0_min, ax0_max = np.argwhere(ax0 > 0).flatten().min(), np.argwhere(ax0 > 0).flatten().max()
    ax0_start, ax0_end = (ax0_min + ax0_max) // 2 - 112, (ax0_min + ax0_max) // 2 + 112
    ax1_min, ax1_max = np.argwhere(ax1 > 0).flatten().min(), np.argwhere(ax1 > 0).flatten().max()
    ax1_start, ax1_end = (ax1_min + ax1_max) // 2 - 112, (ax1_min + ax1_max) // 2 + 112
    ax2_min, ax2_max = np.argwhere(ax2 > 0).flatten().min(), np.argwhere(ax2 > 0).flatten().max()
    ax2_start, ax2_end = (ax2_min + ax2_max) // 2 - 80, (ax2_min + ax2_max) // 2 + 80
    image_data = np.load(saved_folder + pid + '.npy')
    # shift images with start < 0
    shift_0 = 0
    if ax0_start < 0:
        shift_0 = -ax0_start
        ax0_start, ax0_end = 0, 224
    shift_1 = 0
    if ax1_start < 0:
        shift_1 = -ax1_start
        ax1_start, ax1_end = 0, 224
    final_data = image_data[ax0_start:ax0_end, ax1_start:ax1_end, ax2_start:ax2_end]
    if shift_0 > 0:
        double_check_pid.append(pid)
        print('shifting axis 0', shift_0)
        final_data = np.roll(final_data, shift_0, axis=0)
    if shift_1 > 0:
        double_check_pid.append(pid)
        print('shifting axis 1', shift_1)
        final_data = np.roll(final_data, shift_1, axis=1)
    print('saving...')
    np.save('W:/embarc/cropped_images/' + pid + '.npy', final_data)

    # if final_data[0].sum() > 0 or final_data[-1].sum() > 0 or final_data[:, 0].sum() > 0 or final_data[:,-1].sum() > 0 or final_data[:, :,0].sum() > 0 or final_data[:, :,-1].sum() > 0:
    #     print(ax0_min, ax0_max, ax0_start, ax0_end)
    #     print(ax1_min, ax1_max, ax1_start, ax1_end)
    #     print(ax2_min, ax2_max, ax2_start, ax2_end)
    #     print(final_data[0].sum() > 0, final_data[-1].sum() > 0, final_data[:, 0].sum() > 0, final_data[:,-1].sum() > 0, final_data[:, :,0].sum() > 0, final_data[:, :,-1].sum() > 0)
    #     check_pid.append(pid)


ax0 = []
ax1 = []
ax2 = []
for pid in include_patients:
    data = np.load('W:/embarc/cropped_images/' + pid + '.npy')
    if not ax0:
        i = 1
        plt.subplot(3, 3, i)
        plt.axis('off')
        ax0.append(plt.imshow(data[112][..., 0], 'gray'))
        i+=1
        plt.subplot(3, 3, i)
        plt.axis('off')
        ax0.append(plt.imshow(data[112][..., 1], 'gray'))
        i+=1
        plt.subplot(3, 3, i)
        plt.axis('off')
        ax0.append(plt.imshow(data[112][..., 2], 'gray'))
        i+=1
        plt.subplot(3, 3, i)
        plt.axis('off')
        ax1.append(plt.imshow(data[:, 112][..., 0], 'gray'))
        i+=1
        plt.subplot(3, 3, i)
        plt.axis('off')
        ax1.append(plt.imshow(data[:, 112][..., 1], 'gray'))
        i+=1
        plt.subplot(3, 3, i)
        plt.axis('off')
        ax1.append(plt.imshow(data[:, 112][..., 2], 'gray'))
        i+=1
        plt.subplot(3, 3, i)
        plt.axis('off')
        ax2.append(plt.imshow(data[:, :, 80][..., 0], 'gray'))
        i+=1
        plt.subplot(3, 3, i)
        plt.axis('off')
        ax2.append(plt.imshow(data[:, :, 80][..., 1], 'gray'))
        i+=1
        plt.subplot(3, 3, i)
        plt.axis('off')
        ax2.append(plt.imshow(data[:, :, 80][..., 2], 'gray'))
    else:
        ax0[0].set_data(data[112][..., 0])
        ax0[1].set_data(data[112][..., 1])
        ax0[2].set_data(data[112][..., 2])
        ax1[0].set_data(data[:, 112][..., 0])
        ax1[1].set_data(data[:, 112][..., 1])
        ax1[2].set_data(data[:, 112][..., 2])
        ax2[0].set_data(data[:, :, 80][..., 0])
        ax2[1].set_data(data[:, :, 80][..., 1])
        ax2[2].set_data(data[:, :, 80][..., 2])
    plt.pause(1e-3)
    plt.savefig(f'W:/embarc/check_direction/all/{pid}.png')

plt.show()


ax0 = []
ax1 = []
ax2 = []
for pid in include_patients:
    # dwi = sitk.GetArrayFromImage(sitk.ReadImage(f'{nii_folder}{pid}/dwi.nii.gz'))
    # anat1 = sitk.GetArrayFromImage(sitk.ReadImage(f'{nii_folder}{pid}/anat1.nii.gz'))
    # anat2 = sitk.GetArrayFromImage(sitk.ReadImage(f'{nii_folder}{pid}/anat2.nii.gz'))

    ref_image = sitk.ReadImage(f'W:/data_bids4/derivatives/{pid}/ses-1/anat/{pid}_ses-1_aparc+aseg.nii.gz', sitk.sitkFloat64)

    dwi = sitk.ReadImage(f'W:/data_bids4/derivatives/{pid}/ses-1/dwi/{pid}_ses-1_FAWarped.nii.gz', sitk.sitkFloat64)
    anat1 = sitk.ReadImage(f'W:/data_bids4/derivatives/{pid}/ses-1/anat/{pid}_ses-1_brain.nii.gz', sitk.sitkFloat64)
    anat2 = sitk.ReadImage(f'W:/data_bids4/derivatives/{pid}/ses-2/anat/{pid}_ses-2_brain.nii.gz', sitk.sitkFloat64)


    dwi = sitk.Resample(dwi, ref_image, sitk.CenteredTransformInitializer(
            ref_image,
            dwi,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        ), sitk.sitkBSpline, 0, dwi.GetPixelID())

    anat1 = sitk.Resample(anat1, ref_image, sitk.CenteredTransformInitializer(
            ref_image,
            anat1,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        ), sitk.sitkBSpline, 0, anat1.GetPixelID())
    
    anat2 = sitk.Resample(anat2, ref_image, sitk.CenteredTransformInitializer(
            ref_image,
            anat2,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        ), sitk.sitkBSpline, 0, anat2.GetPixelID())

    dwi = sitk.GetArrayFromImage(dwi)
    anat1 = sitk.GetArrayFromImage(anat1)
    anat2 = sitk.GetArrayFromImage(anat2)

    if not ax0:
        i = 1
        plt.subplot(3, 3, i)
        ax0.append(plt.imshow(dwi[127], 'gray'))
        i+=1
        plt.subplot(3, 3, i)
        ax0.append(plt.imshow(anat1[127], 'gray'))
        i+=1
        plt.subplot(3, 3, i)
        ax0.append(plt.imshow(anat2[127], 'gray'))

        i+=1
        plt.subplot(3, 3, i)
        ax1.append(plt.imshow(dwi[:, 127], 'gray'))
        i+=1
        plt.subplot(3, 3, i)
        ax1.append(plt.imshow(anat1[:, 127], 'gray'))
        i+=1
        plt.subplot(3, 3, i)
        ax1.append(plt.imshow(anat2[:, 127], 'gray'))

        i+=1
        plt.subplot(3, 3, i)
        ax2.append(plt.imshow(dwi[:, :,  127], 'gray'))
        i+=1
        plt.subplot(3, 3, i)
        ax2.append(plt.imshow(anat1[:, :,  127], 'gray'))
        i+=1
        plt.subplot(3, 3, i)
        ax2.append(plt.imshow(anat2[:, :,  127], 'gray'))
    
    else:
        ax0[0].set_data(dwi[127])
        ax0[1].set_data(anat1[127])
        ax0[2].set_data(anat2[127])

        ax1[0].set_data(dwi[:, 127])
        ax1[1].set_data(anat1[:, 127])
        ax1[2].set_data(anat2[:, 127])

        ax2[0].set_data(dwi[:, :,  127])
        ax2[1].set_data(anat1[:, :,  127])
        ax2[2].set_data(anat2[:, :,  127])
    
    plt.pause(1e-3)

plt.show()


ax0 = []
ax1 = []
ax2 = []
for pid in include_patients:
    print(pid)
    # dwi = sitk.GetArrayFromImage(sitk.ReadImage(f'{nii_folder}{pid}/dwi.nii.gz'))
    # anat1 = sitk.GetArrayFromImage(sitk.ReadImage(f'{nii_folder}{pid}/anat1.nii.gz'))
    # anat2 = sitk.GetArrayFromImage(sitk.ReadImage(f'{nii_folder}{pid}/anat2.nii.gz'))

    ref_image = sitk.ReadImage(f'W:/data_bids4/derivatives/{pid}/ses-1/anat/{pid}_ses-1_aparc+aseg.nii.gz', sitk.sitkFloat64)

    dwi = sitk.ReadImage(f'W:/data_bids4/derivatives/{pid}/ses-1/dwi/{pid}_ses-1_FAWarped.nii.gz', sitk.sitkFloat64)
    anat1 = sitk.ReadImage(f'W:/data_bids4/derivatives/{pid}/ses-1/anat/{pid}_ses-1_brain.nii.gz', sitk.sitkFloat64)
    anat2 = sitk.ReadImage(f'W:/data_bids4/derivatives/{pid}/ses-2/anat/{pid}_ses-2_brain.nii.gz', sitk.sitkFloat64)

    print('resampling...')
    dwi = sitk.Resample(dwi, ref_image, sitk.CenteredTransformInitializer(
            ref_image,
            dwi,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        ), sitk.sitkBSpline, 0, dwi.GetPixelID())

    anat1 = sitk.Resample(anat1, ref_image, sitk.CenteredTransformInitializer(
            ref_image,
            anat1,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        ), sitk.sitkBSpline, 0, anat1.GetPixelID())
    
    anat2 = sitk.Resample(anat2, ref_image, sitk.CenteredTransformInitializer(
            ref_image,
            anat2,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        ), sitk.sitkBSpline, 0, anat2.GetPixelID())

    print('saving...')
    sitk.WriteImage(dwi, f'{nii_folder}{pid}/dwi.nii.gz')
    sitk.WriteImage(anat1, f'{nii_folder}{pid}/anat1.nii.gz')
    sitk.WriteImage(anat2, f'{nii_folder}{pid}/anat2.nii.gz')


# normalize images
for pid in os.listdir('W:/embarc/cropped_images'):
    print(pid)
    data = np.load(f'W:/embarc/cropped_images/{pid}')
    print('normlizing...')
    anat1 = data[..., 1]
    m1 = anat1[anat1>0.001].mean()
    s1 = anat1[anat1>0.001].std()
    low1, high1 = m1-3*s1, m1+3*s1
    anat2 = data[..., 2]
    m2 = anat2[anat2>0.001].mean()
    s2 = anat2[anat2>0.001].std()
    low2, high2 = m2-3*s2, m2+3*s2
    new_data = np.zeros(data.shape)
    new_data[..., 0] = data[..., 0].clip(0)
    new_data[..., 1] = (anat1.clip(low1, high1) - low1) / (6*s1)
    new_data[..., 2] = (anat2.clip(low2, high2) - low2) / (6*s2)
    print('saving...')
    np.save(f'W:/embarc/normalized_images/{pid}', new_data)