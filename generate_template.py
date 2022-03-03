import pandas as pd

labels = [10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]
bins = [4, 8, 16, 32, 48, 64]

roi_pt_folder = '../../depredict_data/roi_pt/'
roi_bl_folder = '../../depredict_data/roi_bl/'

pt_folder = '../../depredict_data/pt/'
bl_folder = '../../depredict_data/bl/'

image_folders = [pt_folder, bl_folder]
roi_folders = [roi_pt_folder, roi_bl_folder]
periods = ['pt', 'bl']

# columns names
# image_dir,mask_dir,output_file_name,bin_count,label,shape,first_order,glszm,glrlm,ngtdm,gldm,glcm,LBP

for bin in bins:
    for image_folder, roi_folder, period in zip(image_folders, roi_folders, periods):
        templates = {
            'image_dir': [image_folder] * 14,
            'mask_dir': [roi_folder] * 14,
            'output_file_name': [f'radiomics_outputs/{period}_bin{bin:02d}_label{label:02d}.csv' for label in labels],
            'bin_count': [bin] * 14,
            'label': labels,
            'shape': [1]*14,
            'first_order': [1]*14,
            'glszm': [1]*14,
            'glrlm': [1]*14,
            'ngtdm': [1]*14,
            'gldm': [1]*14,
            'glcm': [1]*14,
            'LBP': [1]*14
        }

        pd.DataFrame(templates).to_csv(f'radiomics_templates/{period}_bin{bin:02d}.csv', index=False)