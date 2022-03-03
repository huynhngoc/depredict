# DEPREDICT analysis

## radiomics feature extraction
```
imskaper_feature_extraction -file radiomics_templates/bl_bin04.csv
imskaper_feature_extraction -file radiomics_templates/bl_bin08.csv
imskaper_feature_extraction -file radiomics_templates/bl_bin16.csv
imskaper_feature_extraction -file radiomics_templates/bl_bin32.csv
imskaper_feature_extraction -file radiomics_templates/bl_bin48.csv
imskaper_feature_extraction -file radiomics_templates/bl_bin64.csv


imskaper_feature_extraction -file radiomics_templates/pt_bin04.csv
imskaper_feature_extraction -file radiomics_templates/pt_bin08.csv
imskaper_feature_extraction -file radiomics_templates/pt_bin16.csv
imskaper_feature_extraction -file radiomics_templates/pt_bin32.csv
imskaper_feature_extraction -file radiomics_templates/pt_bin48.csv
imskaper_feature_extraction -file radiomics_templates/pt_bin64.csv
```

## deep learning
```
sbatch slurm.sh config/effb0_012.json depredict_effb0_012 20
```