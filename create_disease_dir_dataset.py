import os
import pandas as pd
import shutil
#load data
csv_file = '**'
df = pd.read_csv(csv_file)
base_path = "**"
base_path1 = "**"
# label
unique_dx = df['DX'].unique()
num_unique_dx = len(unique_dx)
print(f"Total number of unique diagnoses: {num_unique_dx}")
print("Unique diagnoses:", unique_dx)

ptid_to_dx = {}
for index, row in df.iterrows():
    ptid = row['PTID']
    dx = row['DX']
    ptid_to_dx[ptid] = dx

source_folder = base_path + '/' + 'new_tau_163842' #tau_2562_new_test
# source_folder = base_path + '/' + 'tau_2562_new'
# disease_folders = 'disease_folders'
for folder_name in os.listdir(source_folder):
    for ptid, dx in ptid_to_dx.items():
        if ptid in folder_name:
            disease_folders = ptid_to_dx[ptid]
            dx_folder = os.path.join(base_path1 + '/' + disease_folders, dx)
            if not os.path.exists(base_path1 + '/' + disease_folders):
                os.makedirs(base_path1 + '/' + disease_folders)
            if not os.path.exists(dx_folder):
                os.makedirs(dx_folder)
            if not os.path.exists(os.path.join(dx_folder, folder_name)):
                shutil.copytree(os.path.join(source_folder, folder_name), os.path.join(dx_folder, folder_name))

print("Classification differenet Labels.")