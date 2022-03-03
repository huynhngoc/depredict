import SimpleITK as sitk
import os

raw_pt_folder = '../../depredict_data/pt_raw'
raw_bl_folder = '../../depredict_data/bl_raw'

roi_pt_folder = '../../depredict_data/roi_pt'
roi_bl_folder = '../../depredict_data/roi_bl'

pt_folder = '../../depredict_data/pt'
bl_folder = '../../depredict_data/bl'


def normalize(source, mask, target):
    for img in os.listdir(source):
        image_name = source + '/' + img
        roi_name = mask + '/' + img
        target_name = target + '/' + img
        image = sitk.ReadImage(image_name)
        image_data = sitk.GetArrayFromImage(image)
        roi = sitk.ReadImage(roi_name)
        roi_data = sitk.GetArrayFromImage(roi)

        mean = image_data[roi_data > 0].mean()
        std = image_data[roi_data > 0].std()
        # data is normalized between mean - 3std, and mean + 3std
        lower = mean - 3*std
        upper = mean + 3*std

        # then scaled back to 1024 (10 bits)
        new_image_data = 1024 * ((image_data - lower) / (upper - lower)).clip(0, 1)
        new_image = sitk.GetImageFromArray(new_image_data)
        new_image.CopyInformation(image)

        sitk.WriteImage(new_image, target_name)


normalize(raw_bl_folder, roi_bl_folder, bl_folder)
normalize(raw_pt_folder, roi_pt_folder, pt_folder)