import pandas as pd
import numpy as np

labels = [10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]
bins = [4, 8, 16, 32, 48, 64]
periods = ['bl', 'pt']

columns = pd.read_csv('radiomics_outputs/bl_bin04_label10.csv', index_col=0).columns
result_data = []
for period in periods:
    for label in labels:
        pearson_centered = pd.read_csv(f'radiomics_outputs_analysis/pearson_centered_{period}_label{label:02d}.csv', index_col=0)
        pearson_standardized = pd.read_csv(f'radiomics_outputs_analysis/pearson_standardized_{period}_label{label:02d}.csv', index_col=0)

        for col in columns:
            pearson_r = []
            for i, bin1 in enumerate(bins[:-1]):
                for bin2 in bins[i:]:
                    bin1_col = f'bin{bin1:02d}_{col}'
                    bin2_col = f'bin{bin2:02d}_{col}'

                    r = pearson_centered[bin1_col][pearson_centered.index==bin2_col].values[0]
                    pearson_r.append(r)
            result_data.append([period, label, col, np.mean(pearson_r), np.std(pearson_r)])

pd.DataFrame(result_data, columns=['period', 'label', 'column', 'pearson_r_mean', 'pearson_r_std']).to_csv('radiomics_outputs_analysis/corr_info.csv', index=False)