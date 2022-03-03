import chunk
import SimpleITK as sitk
import numpy as np
import h5py
all_folds = []
all_labels = []
for __ in range(2):
    placebo = [3, 7, 8, 17, 18, 20, 21, 29, 
            32, 33, 37, 39, 42, 45, 47, 50, 51, 56, 
            62, 64, 68, 69, 72, 74]

    mph = [1, 4, 6, 10, 11, 22, 24, 27, 34, 36, 38, 
        41, 44, 49, 52, 54, 59, 63, 67, 70, 71, 75]


    np.random.shuffle(placebo)
    np.random.shuffle(mph)

    folds = []
    labels = []

    for i in range(3):
        fold = []
        label = []
        fold.append(placebo.pop())
        label.append(0)
        for _ in range(7):
            fold.append(mph.pop())
            label.append(1)
            fold.append(placebo.pop())
            label.append(0)
        if i==2:
            fold.append(mph.pop())
            label.append(1)
        folds.append(fold)
        labels.append(label)

    assert len(placebo) == 0
    assert len(mph) == 0
    print(folds, labels)
    all_folds.extend(folds)
    all_labels.extend(labels)

print(all_folds, all_labels)

bl_image = '../../depredict_data/bl/{pid:03d}_bl.nii'
pt_image = '../../depredict_data/pt/{pid:03d}_pt.nii'

def get_image(name):
    image = sitk.ReadImage(name)
    return sitk.GetArrayFromImage(image)


with h5py.File('../../depredict_data/depredict_v2.h5', 'w') as f:
    for i in range(6):
        f.create_group(f'fold_{i}')

for i in range(6):
    images = []
    for pid in all_folds[i]:
        bl = get_image(bl_image.format(pid=pid))
        pt = get_image(bl_image.format(pid=pid))
        images.append(np.stack([bl, pt], axis=-1))
    data = np.stack(images)
    with h5py.File('../../depredict_data/depredict_v2.h5', 'a') as f:
        f[f'fold_{i}'].create_dataset('input', data=data, dtype='f4', chunks=(1, 120, 256,256, 1), compression='lzf')
        f[f'fold_{i}'].create_dataset('target', data=all_labels[i], dtype='f4', compression='lzf')
        f[f'fold_{i}'].create_dataset('patient_idx', data=all_folds[i])

for i in range(6):
    with h5py.File('../../depredict_data/depredict_v2.h5', 'r') as f:
        print(f[f'fold_{i}']['input'])
        print(f[f'fold_{i}']['target'])
        print(f[f'fold_{i}']['patient_idx'])