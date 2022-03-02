import SimpleITK as sitk
import numpy as np

seg_filename = '//nmbu.no/LargeFile/Project/REALTEK-HeadNeck-Project/DEPREDICT/ADHD_seg_motion/ADHD_seg_motion/rois_output_firstseg/{treatment_type}/image/{img_type}/nifti/{pid:03d}_{img_type}_first_all_fast_firstseg.nii.gz'
seg_filename2 = '//nmbu.no/LargeFile/Project/REALTEK-HeadNeck-Project/DEPREDICT/ADHD_seg_motion/ADHD_seg_motion/rois_output_firstseg/{treatment_type}/image/{img_type}/nifti/{pid:03d}_{img_type}_first_all_none_firstseg.nii.gz'
img_filename = '//nmbu.no/LargeFile/Project/REALTEK-HeadNeck-Project/DEPREDICT/ADHD_seg_motion/ADHD_seg_motion/children/{treatment_type}/image/{img_type}/nifti/{pid:03d}_{img_type}.nii'


treatment_type = 'MPH'
img_type = 'bl'
pid = 10

contour_bl_1 = sitk.ReadImage(seg_filename.format(
    treatment_type=treatment_type, img_type=img_type, pid=pid))
img_bl_1 = sitk.ReadImage(img_filename.format(
    treatment_type=treatment_type, img_type=img_type, pid=pid))

img_type = 'pt'
contour_pt_1 = sitk.ReadImage(seg_filename.format(
    treatment_type=treatment_type, img_type=img_type, pid=pid))
contour_pt_1_alt = sitk.ReadImage(seg_filename2.format(
    treatment_type=treatment_type, img_type=img_type, pid=pid))
img_pt_1 = sitk.ReadImage(img_filename.format(
    treatment_type=treatment_type, img_type=img_type, pid=pid))


def compare(img1, img2):
    data1 = sitk.GetArrayFromImage(img1)
    data2 = sitk.GetArrayFromImage(img2)
    cond1 = np.allclose(img1.GetSpacing(), img2.GetSpacing())
    cond2 = np.all(img1.GetSize() == img2.GetSize())
    cond3 = np.allclose(data1, data2)

    if not cond3:
        print(np.unique(data1), np.unique(data2))
    return cond1 and cond2 and cond3


compare(contour_bl_1, contour_pt_1)

compare(contour_pt_1_alt, contour_pt_1)

# \\nmbu.no\LargeFile\Project\REALTEK-HeadNeck-Project\DEPREDICT\ADHD_seg_motion\ADHD_seg_motion\rois_output_firstseg\MPH\image\pt\nifti

img_pt_1.GetSize()
img_pt_1.GetSpacing()


pid = 1
contour_pt_test = sitk.ReadImage(seg_filename2.format(
    treatment_type=treatment_type, img_type=img_type, pid=pid))
data_test = sitk.GetArrayFromImage(contour_pt_test)
np.unique(data_test)
