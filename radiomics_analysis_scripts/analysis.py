import pandas as pd
import numpy as np

labels = [10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]
bins = [4, 8, 16, 32, 48, 64]
periods = ['bl', 'pt']

analysis_data = pd.read_csv('radiomics_outputs_analysis/corr_info.csv')
# column names
columns = np.unique(analysis_data.column.values)
# column types
col_types = np.unique([col.split('_')[0] for col in columns], return_counts=True)
# 'LBP', 'first', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm', 'shape'
# 10, 18, 24, 14, 16, 16,  5, 14

[col for col in columns if col.split('_')[0] == 'first']
analysis_data.shape
# 3276 rows

# constant_col = analysis_data['pearson_r_std'] == 0
# analysis_data[constant_col]

constant_col = analysis_data['pearson_r_mean'] == 1
constant_data = analysis_data[constant_col]
# 1120 rows

# are they same period, labels
constant_data.groupby('column').agg({'period': 'count', 'label':'count'})
# YES! all 28 - max number possible for 2 periods and 14 labels
# what are they?
# LBP features (10, ofc, did not depend on bins), shape features (14), 1st order feature (16)
# the names are:
const_col_names = constant_data.groupby('column').agg({'period': 'count', 'label':'count'}).index.values
# all shape features, all LBP features, all 1st order features except for
# first_order_Entropy and first_order_Uniformity


high_corr = analysis_data['pearson_r_mean'] > 0.9
t = analysis_data[high_corr].groupby('column').agg({'period': 'count', 'label':'count'})
t[t.period < 28]