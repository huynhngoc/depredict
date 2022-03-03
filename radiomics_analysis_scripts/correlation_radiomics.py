import hoggorm as ho
import pandas as pd
import numpy as np
import scipy.stats as st

labels = [10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]
bins = [4, 8, 16, 32, 48, 64]
periods = ['bl', 'pt']

filename = 'radiomics_outputs/{period}_bin{bin:02d}_label{label:02d}.csv'

# comparing different bins, per labels
for period in periods:
    for label in labels:
        name_list = []
        data_list_centered = []
        data_list_standardized = []
        for bin in bins:
            print(period, label, bin)
            data = pd.read_csv(filename.format(period=period, label=label, bin=bin), index_col=0)
            data_list_centered.append(ho.center(data.values, axis=0))
            data_list_standardized.append(ho.standardise(data.values, mode=0))
            name_list.extend([f'bin{bin:02d}_{col}' for col in data.columns])

        print('Calculating pearson r centered')
        pearson_centered = pd.DataFrame(np.concatenate(data_list_centered, axis=1), columns=name_list).corr()

        print('Calculating pearson standardized')
        pearson_standardized = pd.DataFrame(np.concatenate(data_list_standardized, axis=1), columns=name_list).corr()

        print('saving to files')
        pearson_centered.to_csv(f'radiomics_outputs_analysis/pearson_centered_{period}_label{label:02d}.csv')
        pearson_standardized.to_csv(f'radiomics_outputs_analysis/pearson_standardized_{period}_label{label:02d}.csv')