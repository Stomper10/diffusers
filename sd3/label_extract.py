import re
import os
import math
import json
import numpy as np
import pandas as pd



# Raw phenotype data from DNAnexus
MRI_pheno_1_base = pd.read_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/MRI_pheno_1_base.csv", low_memory=False) # 502140
MRI_pheno_2_T1image = pd.read_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/MRI_pheno_2_T1image.csv", low_memory=False) # 502140
MRI_pheno_3_ICD10_1 = pd.read_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/MRI_pheno_3_ICD10_1.csv", low_memory=False) # 502140
MRI_pheno_4_ICD10_2 = pd.read_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/MRI_pheno_4_ICD10_2.csv", low_memory=False) # 502140
MRI_pheno_5_mainICD10 = pd.read_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/MRI_pheno_5_mainICD10.csv", low_memory=False) # 502140

# T1 unzip dir
MRI_ids = os.listdir("/leelabsg/data/20252_unzip")
len(MRI_ids) # 48473 (removed 8 unzip failed dirs: 1198204_20252_2_0, 1746620_20252_2_0, 2062434_20252_2_0, 2250047_20252_2_0, 3378932_20252_2_0, 4563493_20252_2_0, 5362138_20252_2_0, 5405738_20252_2_0)
# extract instance 2 ids
MRI_ids_i2 = [int(id[:7]) for id in MRI_ids if id[-3] == "2"]
len(MRI_ids_i2) # 43554
# extract instance 3 ids
MRI_ids_i3 = [int(id[:7]) for id in MRI_ids if id[-3] == "3"]
len(MRI_ids_i3) # 4919



# instance 2 -> filtered labels only for MRI participants
MRI_pheno_1_base_i2 = MRI_pheno_1_base[MRI_pheno_1_base['eid'].isin(MRI_ids_i2)].loc[:, ~MRI_pheno_1_base.columns.str.contains("i3")] # 43531
MRI_pheno_2_T1image_i2 = MRI_pheno_2_T1image[MRI_pheno_2_T1image['eid'].isin(MRI_ids_i2)].loc[:, ~MRI_pheno_2_T1image.columns.str.contains("i3")] # 43531
MRI_pheno_3_ICD10_1_i2 = MRI_pheno_3_ICD10_1[MRI_pheno_3_ICD10_1['eid'].isin(MRI_ids_i2)]
MRI_pheno_4_ICD10_2_i2 = MRI_pheno_4_ICD10_2[MRI_pheno_4_ICD10_2['eid'].isin(MRI_ids_i2)]
MRI_pheno_5_mainICD10_i2 = MRI_pheno_5_mainICD10[MRI_pheno_5_mainICD10['eid'].isin(MRI_ids_i2)]

dfs_2 = [MRI_pheno_1_base_i2, MRI_pheno_2_T1image_i2, MRI_pheno_3_ICD10_1_i2, MRI_pheno_4_ICD10_2_i2, MRI_pheno_5_mainICD10_i2]
merged_df_i2 = dfs_2[0]
for df in dfs_2[1:]:
    merged_df_i2 = pd.merge(merged_df_i2, df, on='eid', how='inner')

# instance 3 -> filtered labels only for MRI participants
MRI_pheno_1_base_i3 = MRI_pheno_1_base[MRI_pheno_1_base['eid'].isin(MRI_ids_i3)].loc[:, ~MRI_pheno_1_base.columns.str.contains("i2")] # 4918
MRI_pheno_2_T1image_i3 = MRI_pheno_2_T1image[MRI_pheno_2_T1image['eid'].isin(MRI_ids_i3)].loc[:, ~MRI_pheno_2_T1image.columns.str.contains("i2")] # 43531
MRI_pheno_3_ICD10_1_i3 = MRI_pheno_3_ICD10_1[MRI_pheno_3_ICD10_1['eid'].isin(MRI_ids_i3)]
MRI_pheno_4_ICD10_2_i3 = MRI_pheno_4_ICD10_2[MRI_pheno_4_ICD10_2['eid'].isin(MRI_ids_i3)]
MRI_pheno_5_mainICD10_i3 = MRI_pheno_5_mainICD10[MRI_pheno_5_mainICD10['eid'].isin(MRI_ids_i3)]

dfs_3 = [MRI_pheno_1_base_i3, MRI_pheno_2_T1image_i3, MRI_pheno_3_ICD10_1_i3, MRI_pheno_4_ICD10_2_i3, MRI_pheno_5_mainICD10_i3]
merged_df_i3 = dfs_3[0]
for df in dfs_3[1:]:
    merged_df_i3 = pd.merge(merged_df_i3, df, on='eid', how='inner')

# which images don't have label info?
absent_image_i2 = set(list(MRI_pheno_1_base_i2['eid'])) - set(MRI_ids_i2)
absent_label_i2 = set(MRI_ids_i2) - set(list(MRI_pheno_1_base_i2['eid']))
len(absent_label_i2)

absent_image_i3 = set(list(MRI_pheno_1_base_i3['eid'])) - set(MRI_ids_i3)
absent_label_i3 = set(MRI_ids_i3) - set(list(MRI_pheno_1_base_i3['eid']))
len(absent_label_i3)


# check if mainICD10 is the perfect subset of ICD10
def is_subset(row):
    # Handle NaN values by replacing them with an empty set
    diagnoses_41202 = set(row['p41202'].split('|')) if pd.notna(row['p41202']) else set()
    diagnoses_41270 = set(row['p41270'].split('|')) if pd.notna(row['p41270']) else set()
    
    # Check if p41202 diagnoses are a subset of p41270 diagnoses
    return diagnoses_41202.issubset(diagnoses_41270)

merged_df_i2['is_p41202_subset_of_p41270'] = merged_df_i2.apply(is_subset, axis=1)
merged_df_i2['is_p41202_subset_of_p41270'].sum() # perfect subset
merged_df_i3['is_p41202_subset_of_p41270'] = merged_df_i3.apply(is_subset, axis=1)
merged_df_i3['is_p41202_subset_of_p41270'].sum() # perfect subset

# drop mainICD10
columns_to_drop_mainICD10_i2 = [col for col in merged_df_i2.columns if "p41202" in col or "p41262" in col]
merged_df_i2_filtered = merged_df_i2.drop(columns=columns_to_drop_mainICD10_i2).sort_values('eid').reset_index(drop=True)

columns_to_drop_mainICD10_i3 = [col for col in merged_df_i3.columns if "p41202" in col or "p41262" in col]
merged_df_i3_filtered = merged_df_i3.drop(columns=columns_to_drop_mainICD10_i3).sort_values('eid').reset_index(drop=True)



# Flatten the DataFrame
def flatten_diagnoses(df, i): # can handle patient that has 0 dianosis
    # Split the diagnoses column into a list
    if i == 2:
        retain_list = ['eid', 'p52', 'p34', 'p31', 'p54_i2', 'p53_i2', 'p21003_i2', 'p21000_i2', 'p20016_i2', 'p21001_i2', 'p23104_i2', 'p25009_i2', 'p25007_i2', 'p25005_i2', 'p25001_i2', 'p25003_i2', 'p25025_i2']
    else:
        retain_list = ['eid', 'p52', 'p34', 'p31', 'p54_i3', 'p53_i3', 'p21003_i3', 'p21000_i3', 'p20016_i3', 'p21001_i3', 'p23104_i3', 'p25009_i3', 'p25007_i3', 'p25005_i3', 'p25001_i3', 'p25003_i3', 'p25025_i3']
    df['diagnoses_list'] = df['p41270'].str.split('|')

    # Identify diagnosis date columns based on the naming pattern
    date_columns = [col for col in df.columns if col.startswith('p41280_a')]
    
    # Melt the date columns to align them into a single column
    df_dates = df.melt(
        id_vars=retain_list+['diagnoses_list'],  # Include unrelated columns
        value_vars=date_columns,
        var_name='diagnosis_number',
        value_name='diagnosis_date'
    )
    
    # Extract the index part after "a" in column names
    df_dates['diagnosis_index'] = df_dates['diagnosis_number'].str.extract(r'p41280_a(\d+)').astype(int)

    # Align each diagnosis with the corresponding date
    df_exploded = df.explode('diagnoses_list').reset_index(drop=True)
    df_exploded['diagnosis_match_index'] = df_exploded.groupby('eid').cumcount()
    
    df_aligned = pd.merge(
        df_dates,
        df_exploded[['eid', 'diagnoses_list', 'diagnosis_match_index']],
        left_on=['eid', 'diagnosis_index'],
        right_on=['eid', 'diagnosis_match_index']
    )

    df_aligned = df_aligned[retain_list + ['diagnoses_list_y', 'diagnosis_date']]  # Retain unrelated columns
    df_aligned = df_aligned.rename(columns={'diagnoses_list_y': 'diagnosis'})
    
    # Extract only the ICD code (first part of each diagnosis string)
    df_aligned['diagnosis'] = df_aligned['diagnosis'].str.split(' ').str[0]
    
    return df_aligned

# flattened_i2 = flatten_diagnoses(merged_df_i2_filtered, i=2).sort_values('eid').reset_index(drop=True)
# process in chunks due to large dataframe for i2
def process_in_chunks(df, chunk_size, flatten_func, **kwargs):
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
    processed_chunks = [flatten_func(chunk, **kwargs) for chunk in chunks]
    return pd.concat(processed_chunks, ignore_index=True)

flattened_i2 = process_in_chunks(merged_df_i2_filtered, chunk_size=10, flatten_func=flatten_diagnoses, i=2)
flattened_i3 = flatten_diagnoses(merged_df_i3_filtered, i=3).sort_values('eid').reset_index(drop=True)

# save intermediate df
flattened_i2.sort_values('eid').to_csv('/shared/s1/lab06/wonyoung/diffusers/sd3/data/flattened_i2.csv', index=False)
flattened_i3.sort_values('eid').to_csv('/shared/s1/lab06/wonyoung/diffusers/sd3/data/flattened_i3.csv', index=False)



# drop ICDs if diagnosis date > attending date
flattened_i2 = pd.read_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/flattened_i2.csv") # 398663
flattened_i3 = pd.read_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/flattened_i3.csv") # 37212

flattened_i2['p53_i2'] = pd.to_datetime(flattened_i2['p53_i2'])
flattened_i2['diagnosis_date'] = pd.to_datetime(flattened_i2['diagnosis_date'])
condition = flattened_i2['p53_i2'] < flattened_i2['diagnosis_date']
flattened_i2.loc[condition, ['diagnosis', 'diagnosis_date']] = np.nan # just filtering rows also filter NaN rows
flattened_i2_date_filtered = flattened_i2.drop_duplicates() # 261856
flattened_i2_date_filtered_clean = flattened_i2_date_filtered[~(flattened_i2_date_filtered['diagnosis'].isna() & flattened_i2_date_filtered.duplicated(subset=['eid'], keep=False))] # 244077

flattened_i3['p53_i3'] = pd.to_datetime(flattened_i3['p53_i3'])
flattened_i3['diagnosis_date'] = pd.to_datetime(flattened_i3['diagnosis_date'])
condition = flattened_i3['p53_i3'] < flattened_i3['diagnosis_date']
flattened_i3.loc[condition, ['diagnosis', 'diagnosis_date']] = np.nan
flattened_i3_date_filtered = flattened_i3.drop_duplicates() # 31620
flattened_i3_date_filtered_clean = flattened_i3_date_filtered[~(flattened_i3_date_filtered['diagnosis'].isna() & flattened_i3_date_filtered.duplicated(subset=['eid'], keep=False))] # 30442



# load phecode mapping file
phecode = pd.read_csv('/shared/s1/lab06/wonyoung/diffusers/sd3/data/icd10-phecode-map-jbn-Nov2017.txt', sep='\t', header=0, dtype={'icd10': str, 'phecode': str})
phecode['phecode'] = phecode['phecode'].str.split('.').str[0]
phecode_int = phecode.drop_duplicates()

phecode_int[phecode_int['icd10'].duplicated()]


flatdatefilt_i2_phecode = pd.merge(
    flattened_i2_date_filtered_clean,
    phecode_int,
    left_on='diagnosis',  # Column in flattened_i2_date_filtered_clean
    right_on='icd10',     # Column in phecode
    how='left'            # Use 'left' join to keep all rows in flattened_i2_date_filtered_clean
)

unmappe_icd10s = list(set(flatdatefilt_i2_phecode[flatdatefilt_i2_phecode['phecode'].isna()]['diagnosis']))
unmappe_icd10s_cleaned = list(filter(lambda x: not (isinstance(x, float) and math.isnan(x)), unmappe_icd10s))
len(unmappe_icd10s_cleaned)

# Custom sorting key to split alphabet and number
def sorting_key(value):
    match = re.match(r"([A-Za-z]+)([0-9.]+)", value)
    if match:
        return match.group(1), float(match.group(2))  # Alphabet, then numeric value
    return value

unmappe_icd10s_sorted = sorted(unmappe_icd10s_cleaned, key=sorting_key)
# Function to divide list into chunks of length 50
def divide_list(lst, chunk_size=50):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
# Divide the list
chunks = divide_list(unmappe_icd10s_sorted, chunk_size=50)
# Print the result
list_of_chunks = []
for i, chunk in enumerate(chunks):
    list_of_chunks.append(chunk)
list_of_chunks[0]

# brain structure related phecode list
T1_related_phecode_list = [330]


# 
unmappe_icd10s = set(flatdatefilt_i2_phecode[flatdatefilt_i2_phecode['phecode'].isna()]['diagnosis'])

phecode.iloc[72,:]
flattened_i2_date_filtered[flattened_i2_date_filtered['eid']==1000800]['p53_i2']
flattened_i2_date_filtered_clean[flattened_i2_date_filtered_clean['eid']==1000800]
flattened_i2_date_filtered[flattened_i2_date_filtered['diagnosis'].isna()]
flattened_i2_date_filtered_clean[flattened_i2_date_filtered_clean['diagnosis'].isna()]

not_nan_rows = flattened_i2[flattened_i2['diagnosis'].isna()]

len(set(list(flattened_i2['eid'])))
len(set(list(flattened_i2_date_filtered['eid'])))

type(flattened_i2['p53_i2'].iloc[0])

flattened_i2['diagnosis_date'].iloc[0]
flattened_i2[flattened_i2['eid']==1000638]['p53_i2']
merged_df_i2_filtered[merged_df_i2_filtered['eid']==1000638]['p41270']














merged_df_i3_filtered[merged_df_i3_filtered['eid']==1003365]['p41280_a3']
merged_df_i3_filtered[merged_df_i3_filtered['p41280_a0'].isna()]
flattened_i3[flattened_i3['diagnosis'].isna()]


merged_df_i3_filtered['diagnoses_list'] = merged_df_i3_filtered['p41270'].str.split('|')
merged_df_i3_filtered[merged_df_i3_filtered['eid']==1003365]['diagnoses_list'][0]

date_columns = [col for col in merged_df_i3_filtered.columns if col.startswith('p41280_a')]
len(date_columns)
df_dates = merged_df_i3_filtered.melt(
        id_vars=['eid', 'diagnoses_list'],
        value_vars=date_columns,
        var_name='diagnosis_number',
        value_name='diagnosis_date'
    )
df_dates[df_dates['eid'] == 1003365].iloc[1,:]
df_dates['diagnosis_index'] = df_dates['diagnosis_number'].str.extract(r'p41280_a(\d+)').astype(int)
df_dates[df_dates['eid'] == 1003365].iloc[4,:]

df_exploded = merged_df_i3_filtered.explode('diagnoses_list').reset_index(drop=True)
df_exploded['diagnosis_match_index'] = df_exploded.groupby('eid').cumcount()
df_aligned = pd.merge(
        df_dates,
        df_exploded[['eid', 'diagnoses_list', 'diagnosis_match_index']],
        left_on=['eid', 'diagnosis_index'],
        right_on=['eid', 'diagnosis_match_index']
    )

df_dates_ex = df_dates.explode('diagnoses_list').reset_index(drop=True)
df_dates_ex[df_dates_ex['eid'] == 1003365].iloc[4,:]

df_dates['diagnosis_match_index'] = df_dates.groupby('eid').cumcount()  # Align using `cumcount()`


flattened_i2 = flatten_diagnoses(merged_df_i2_filtered)
flattened_i3 = flatten_diagnoses(merged_df_i3_filtered)


volume_csv = pd.read_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/MRI_pheno_2_image_wy_exported.csv")
volume_csv = volume_csv.dropna(how='all', subset=volume_csv.columns.difference(['eid']))
colnames = ['eid', "volume_grey_and_white_i2", "volume_grey_and_white_i3", 
            "volume_white_i2", "volume_white_i3",
            "volume_grey_i2", "volume_grey_i3",
            "volume_peripheral_i2", "volume_peripheral_i3",
            "volume_ventricular_i2", "volume_ventricular_i3",
            "volume_brainstem_i2", "volume_brainstem_i3"]
volume_csv.rename(columns={col: f"{colnames[i]}" for i, col in enumerate(volume_csv.columns)}, inplace=True)
prev_meta_csv = pd.read_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/ukbb_cn_train.csv")



matching_rows = volume_csv[volume_csv['eid'] == 1000502]

# number of 20252_new files = 48481
zip_filenames_20252 = os.listdir("/media/leelabsg-storage1/DATA/UKBB/bulk/20252_new") # 48481 zip files list
zip_filenames_20252[:10] # print samples of zip files list
unzip_filenames_20252 = [i[:-4] for i in zip_filenames_20252] # unzip files list (not remove repeat visit samples )
len(unzip_filenames_20252) # 48481

T1_MASTER_202212 = pd.DataFrame(columns=["id", "usable", "instance", "MRI_acq_date", "dir_name", "abs_path"])
T1_MASTER_202212["id"] = [i[:7] for i in zip_filenames_20252]
T1_MASTER_202212["instance"] = [i[-3] for i in unzip_filenames_20252] # firt visit = 2 / repeat visit = 3
T1_MASTER_202212["dir_name"] = unzip_filenames_20252
T1_MASTER_202212["abs_path"] = [f"/media/leelabsg-storage1/DATA/UKBB/bulk/20252_unzip/{i}" for i in unzip_filenames_20252]

MRI_acq_date_list = []
MRI_sanity_list = []
for name in unzip_filenames_20252:
    dir = "/media/leelabsg-storage1/DATA/UKBB/bulk/20252_unzip/" + f"{name}" + "/T1/T1.json"
    try:
        with open(dir, 'r') as file:
            data = json.load(file)
        acq_date = data["AcquisitionDate"]
        sanity = "Yes"
    except:
        acq_date = "Not available"
        sanity = "No"
        print(f"{name} seems compromised.")
    MRI_acq_date_list.append(acq_date)
    MRI_sanity_list.append(sanity)


T1_MASTER_202212["MRI_acq_date"] = MRI_acq_date_list
T1_MASTER_202212["usable"] = MRI_sanity_list




remove = ["1198204", "1746620", "2062434", "2250047", # unzip error file ids
          "3378932", "4563493", "5362138", "5405738"]
id_list_fil = [x for x in id_list if x not in remove] # 8 removed
len(id_list_fil) # 43554
id_list_fil[:10]



# Load MASTER file and mapping file
disease = pd.read_csv('/shared/s1/lab06/wonyoung/diffusers/sd3/data/PEDMASTER_ALL_20180514.txt', sep='\t')
mapping = pd.read_csv('/shared/s1/lab06/wonyoung/diffusers/sd3/data/mapping.csv')


# ID file -> not necessary
ii = np.load('/media/leelabsg-storage1/yein/research/data/csv/jangho/ID_128_full.npy', allow_pickle=True)
ii.shape
a = ii.tolist()
ii_pd = pd.DataFrame(a)
cap = ii_pd.rename(columns={0:'id', 1:'age'})
#cap = cap.loc[~(cap == 0.0).all(axis=1)]




mapping_mri = mapping[mapping['Shawn'].isin(cap['id'].copy())] # 37792
print(disease.shape)
print(cap.shape)
print(mapping_mri.shape)

mapping_mri.columns = ['FID', 'Shawn'] 
mri_disease = pd.merge(disease, mapping_mri, on='FID') #37724
print(disease.shape)



# Explore what diseases the individuals have
mri_disease_count= np.sum(mri_disease.iloc[:, 70:1756], axis = 0)
# top ten disease
mri_disease_count[mri_disease_count.isin(np.sort(mri_disease_count)[::-1][:10])]
# exploration diagnosed
np.sum(mri_disease.iloc[:, 70:1756], axis = 1).value_counts()[:20]

#Filtering out individuals with diseases we chose
def preprocess_filt_disease(data, disease):
    hi_disease = data[np.sum(data[disease] == 1, axis = 1) >= 1].index
    data.drop(hi_disease, axis = 0, inplace = True)
    
    print(len(hi_disease), 'dropped')


#processed = preprocess_make_train(mri_disease, 9)
disease_list = ['X191','X198','X225','X249', 'X250', 'X290','X290.11', 'X291', 'X292', 'X303', 'X306', 'X315', 'X430','X433', 'X752',
                'X753']+list(mri_disease.columns[128:260])+list(mri_disease.columns[547:550])
# 191 Manlignant and unknown neoplasms of brain and nervous system
# 198 Secondary malignant neoplasm
# 225 Benign neoplasm of brain and other parts of nervous system
# 249, 250 diabetes
# 290~292 dimensia
# 303 Psychogenic and somatoform disorders
# 306 Other mental disorder
# 315 Develomental delays and disorders
# 331 Other cerebral degenerations
# 346 Abnormal findings on study of brain and/or nervous system
# 348 Other conditions of brain
# 430~433 cerebrovascular disease
# 752 congenital anomalies of nervous system, spine
# 753 congenital anomalies of nervous system, spine
# 781 Symptoms involving nervous and musculoskeletal systems
# -ing

# Cancers
# 145, 149, 150, 151 153, 155, 157, 158, 159, 164, 165, 170, 172, 173, 174, 175, 180, 182, 184, 185, 187, 189, 190, 191
# 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 204, 208, 210, 211, 212, 213, 214, 215, 216, 217, 218, 220, 221, 222, 
# 223, 224, 225, 226, 227, 228, 229, 230
processed=copy.deepcopy(mri_disease)
preprocess_filt_disease(processed, disease_list)
processed

seeya = mri_disease.drop(processed.index, axis = 0)
print(seeya.shape)


import torch
import torch.nn as nn

# Assuming you have lists of conditioning variables for each sample in the batch
ages = [age1, age2, ..., ageN]
genders = [gender1, gender2, ..., genderN]
bvv_values = [bvv1, bvv2, ..., bvvN]
patch_positions = [pos1, pos2, ..., posN]  # Only in Stage 2
patch_position_embedding = nn.Embedding(num_positions, embedding_dim)

cond_list = []

for age, gender, bvv, patch_position in zip(ages, genders, bvv_values, patch_positions):
    # Normalize age and BVV
    normalized_age = (age - min_age) / (max_age - min_age)
    normalized_bvv = (bvv - min_bvv) / (max_bvv - min_bvv)

    # Encode gender
    if gender == 'Male':
        gender_encoded = torch.tensor([1.0, 0.0])
    else:
        gender_encoded = torch.tensor([0.0, 1.0])

    # Encode patch position (choose one method)
    # Option A: Using embeddings (assuming patch_position_embedding is defined)
    patch_pos_encoded = patch_position_embedding(torch.tensor(patch_position))
    # Option B: Normalized coordinates
    # patch_pos_encoded = torch.tensor([x_idx / max_x, y_idx / max_y, z_idx / max_z])

    # Concatenate conditioning variables
    cond_sample = torch.cat([torch.tensor([age]), gender_encoded, torch.tensor([bvv]), patch_pos_encoded], dim=-1)
    cond_list.append(cond_sample)

# Stack all cond samples into a batch tensor
cond = torch.stack(cond_list, dim=0)  # Shape: [batch_size, cond_dim]
