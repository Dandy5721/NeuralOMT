import nibabel as nb
import numpy as np
import torch
import os
# from torch_geometric.data import Data
from collections import defaultdict
from nibabel.freesurfer.mghformat import load
import vtk

class My_dHCP_Data_tau(torch.utils.data.Dataset):
    def __init__(self, input_path, normalisation='std', *args):
        self.input_path = input_path
        obain_target_file = os.listdir(self.input_path)
        all_feature_gii_path = defaultdict()
        all_thick_mgh_path = defaultdict()
        all_dir = []
        dir_set = set()

        for dir in obain_target_file:
            if dir[-5] == "l":
                dir_set.add(dir[:-2] + 'll')
            if dir[-5] == "r":
                dir_set.add(dir[:-2] + 'rr')
        for file in dir_set:
            if file[-2:] == "ll":
                all_feature_gii_path[file[:-2] + 'll0'] = []
                all_feature_gii_path[file[:-2] + 'll1'] = []
                all_thick_mgh_path[file[:-2] + 'll0'] = []
                all_thick_mgh_path[file[:-2] + 'll1'] = []
            if file[-2:] == "rr":
                all_feature_gii_path[file[:-2] + 'rr0'] = []
                all_feature_gii_path[file[:-2] + 'rr1'] = []
                all_thick_mgh_path[file[:-2] + 'rr0'] = []
                all_thick_mgh_path[file[:-2] + 'rr1'] = []

        for file in dir_set:
            all_dir.append(file)
            if file[-2:] == "ll":
                sub_file_path = os.path.join(self.input_path, file[:-2] + "_0")
                for sub_file in os.listdir(sub_file_path):
                    if "lh.fsaverage.pial.gii" == sub_file:
                        # print("-----", sub_file)
                        target_path = os.path.join(sub_file_path, sub_file)
                        all_feature_gii_path[file[:-2] + 'll0'].append(target_path)

                    if "csv" in sub_file:
                        # print("-----", sub_file)
                        target_path = os.path.join(sub_file_path, sub_file)
                        all_thick_mgh_path[file[:-2] + 'll0'].append(target_path)

                sub_file_path = os.path.join(self.input_path, file[:-2] + "_1")
                for sub_file in os.listdir(sub_file_path):
                    if "lh.fsaverage.pial.gii" == sub_file:
                        # print("-----", sub_file)
                        target_path = os.path.join(sub_file_path, sub_file)
                        all_feature_gii_path[file[:-2] + 'll1'].append(target_path)

                    if "csv" in sub_file:
                        # print("-----", sub_file)
                        target_path = os.path.join(sub_file_path, sub_file)
                        all_thick_mgh_path[file[:-2] + 'll1'].append(target_path)

            elif file[-2:] == "rr":
                sub_file_path = os.path.join(self.input_path, file[:-2] + "_0")
                for sub_file in os.listdir(sub_file_path):
                    if "rh.fsaverage.pial.gii" == sub_file:
                        # print("-----", sub_file)
                        target_path = os.path.join(sub_file_path, sub_file)
                        all_feature_gii_path[file[:-2] + 'rr0'].append(target_path)

                    if "csv" in sub_file:
                        # print("-----", sub_file)
                        target_path = os.path.join(sub_file_path, sub_file)
                        all_thick_mgh_path[file[:-2] + 'rr0'].append(target_path)

                sub_file_path = os.path.join(self.input_path, file[:-2] + "_1")
                for sub_file in os.listdir(sub_file_path):
                    if "rh.fsaverage.pial.gii" == sub_file:
                        # print("-----", sub_file)
                        target_path = os.path.join(sub_file_path, sub_file)
                        all_feature_gii_path[file[:-2] + 'rr1'].append(target_path)
                    # print("..........", sub_file)
                    if "csv" in sub_file:
                        print("-----", sub_file)
                        target_path = os.path.join(sub_file_path, sub_file)
                        all_thick_mgh_path[file[:-2] + 'rr1'].append(target_path)

        self.gii_files = all_feature_gii_path
        self.thick_mgh_path = all_thick_mgh_path
        self.normalisation = normalisation
        self.all_dir = all_dir

    def __len__(self):
        return len(self.all_dir)

    def __get_dir_name__(self, idx):
        return self.all_dir[idx]

    def __gengiifilename__(self, dir):
        # gii_filename = self.gii_files[idx]
        # label_filename = self.label_path[idx]
        return self.gii_files[dir]

    def __genmghfilename__(self, dir):
        # gii_filename = self.gii_files[idx]
        # label_filename = self.label_path[idx]
        return self.thick_mgh_path[dir]

    def extract_label(self, label_file):
        label_data = nb.load(label_file).darrays[0].data
        return label_data

    def extract_feature(self, feature_file, mgh_file):
        feature_data = []
        # print('>>>>>>', feature_file)mgh_file
        # print('>>>>>>', mgh_file)

        feature_data.append(nb.load(feature_file[0]).darrays[0].data)
        feature_data.append(nb.load(feature_file[0]).darrays[1].data)
        values = []
        with open(mgh_file[0], "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                if line:
                    values.append(float(line))
        mgh_data = np.array(values)
        return feature_data, mgh_data

    def __max_min_standard__(self, features):
        x_min = min(-0.0001, features.min())
        x_max = max(features.max(), 0.0001)
        big_x_ind = features >= 0
        small_x_ind = features < 0
        features[big_x_ind] = features[big_x_ind] / x_max
        features[small_x_ind] = features[small_x_ind] / x_min
        return features
    def __getcurdirection__(self, idx):
        dir_name = self.__get_dir_name__(idx)
        orient = dir_name[-2:]
        return  orient
    def __getitem__(self, idx):
        dir_name = self.__get_dir_name__(idx)

        # print(dir_name)
        # print()

        orient = dir_name[-2:]
        if orient == "ll":
            gii_filename1 = self.__gengiifilename__(dir_name[:-2] + 'll0')
            gii_filename2 = self.__gengiifilename__(dir_name[:-2] + 'll1')
            mgh_filename1 = self.__genmghfilename__(dir_name[:-2] + 'll0')
            mgh_filename2 = self.__genmghfilename__(dir_name[:-2] + 'll1')
        else:
            gii_filename1 = self.__gengiifilename__(dir_name[:-2] + 'rr0')
            gii_filename2 = self.__gengiifilename__(dir_name[:-2] + 'rr1')
            mgh_filename1 = self.__genmghfilename__(dir_name[:-2] + 'rr0')
            mgh_filename2 = self.__genmghfilename__(dir_name[:-2] + 'rr1')

        features0, mgh_features0 = self.extract_feature(gii_filename1, mgh_filename1)
        features1, mgh_features1 = self.extract_feature(gii_filename2, mgh_filename2)

        mgh_array0 = mgh_features0
        # mgh_array0 = mgh_features0.get_fdata()
        # mgh_array0 = np.array(mgh_array0,dtype=np.float32)

        mgh_array1 = mgh_features1
        # mgh_array1 = mgh_features1.get_fdata()
        # mgh_array1 = np.array(mgh_array1, dtype=np.float32)
        # mgh_array11 = torch.Tensor(mgh_array1)
        # print('111111', mgh_array11.shape)
        # print('222222', mgh_array11.squeeze(1).shape)
        # if torch.Tensor(mgh_array0).shape[0] == 0 or torch.Tensor(mgh_array1).shape[0] == 0:
        # print("............", mgh_filename1)
        # print("-------------", torch.tensor(mgh_array0).max())
        # print(",,,,,,,,,,,,,,", torch.tensor(mgh_array1).max())
        sample = { 'v0': torch.Tensor(features0[0]), 'f0': torch.Tensor(features0[1]), 'mgh0' : torch.Tensor(mgh_array0).unsqueeze(1),
        'v1': torch.Tensor(features1[0]), 'f1': torch.Tensor(features1[1]), 'mgh1' : torch.Tensor(mgh_array1).unsqueeze(1) }  # 这里假设 v 和 f 与 features 的结构相同

        return sample


base_path = '**'

