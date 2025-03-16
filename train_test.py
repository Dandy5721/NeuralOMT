import os
import random
import string
import shutil

base_path = "**/tau_163842_new"
dst_path = '**/tau_163842_new_test'

all_folders = [f.path for f in os.scandir(base_path) if f.is_dir()]
test_len = int(len(all_folders) / 2 * 0.2)
print(len(all_folders), test_len)

count = 0
fold_list = set()
for folder in all_folders:
    name = folder.split("/")[-1][:-1]
    if name not in fold_list:
        fold_list.add(name)
        shutil.move(folder[:-1] + '0', dst_path + '/' + name + '0')
        shutil.move(folder[:-1] + '1', dst_path + '/' + name + '1')
        count += 1
        print(count , name)
        if count >= test_len:
            break
    else:
        continue




