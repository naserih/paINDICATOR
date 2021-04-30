#split_train_test.py
import os
import shutil
import random
import numpy as np

n_of_sets = 4
set_size = 54
folder_index_start = 2
# src_folder_path = '../DICOMfortable/static/data/MET_CTS/'
# src_folder_path = '/mnt/iDiveShare/hossein/1_RAWDATA/R2_1620_TSP_M2N_400/CTRL2/'
src_folder_path = '../DICOMfortable/static/data/CTRL_CTS/'
out_src_path = '../DICOMfortable/static/data/CTRL'
out_sets = [out_src_path+str(i+folder_index_start)+'/' for i in range(n_of_sets)]
for out_set in out_sets:
    if not os.path.exists(out_set):
        os.mkdir(out_set)

src_files = os.listdir(src_folder_path)
random.shuffle(src_files)


# print (src_files)
cnt = 0
set_num = 0
for src_file in src_files:
    if cnt == set_size:
        cnt = 0
        set_num += 1
    print(cnt, src_folder_path+src_file, out_sets[set_num])
    cnt+=1
    shutil.copytree(src_folder_path+src_file, out_sets[set_num]+src_file)
# print(src_file_names)
# patient_ids = [f.split("_")[0] for f in src_file_names]
# note_date = [int((f.split("_")[1])[:8]) for f in src_file_names]
# print(note_date)
# cnt = 0
# set0_patients = []
# for i in range(len(src_file_names)):
#     if cnt < 50 and note_date[i] > 20180000 and patient_ids[i] not in set0_patients:
#         print (cnt)
#         cnt += 1
#         set0_patients.append(patient_ids[i])
#         shutil.copy(src_folder_path+src_file_names[i], out_sets[0] +src_file_names[i])
#     elif cnt < 100 and note_date[i] > 20180000 and patient_ids[i] not in set0_patients:
#         print (cnt)
#         cnt += 1
#         set0_patients.append(patient_ids[i])
#         shutil.copy(src_folder_path+src_file_names[i], out_sets[1] +src_file_names[i])
#     elif cnt < 150 and note_date[i] > 20180000 and patient_ids[i] not in set0_patients:
#         print (cnt)
#         cnt += 1
#         set0_patients.append(patient_ids[i])
#         shutil.copy(src_folder_path+src_file_names[i], out_sets[2] +src_file_names[i])

#     elif cnt < 300 and note_date[i] > 20180000 and patient_ids[i] not in set0_patients:
#         print (cnt)
#         cnt += 1
#         set0_patients.append(patient_ids[i])
#         shutil.copy(src_folder_path+src_file_names[i], out_sets[3] +src_file_names[i])
#     else:
#         # print (cnt)
#         # set0_patients.append(patient_ids[i])
#         shutil.copy(src_folder_path+src_file_names[i], out_sets[4] +src_file_names[i])



# # for shuffle_file_name in shuffle_file_names[:50]:
# #     shutil.move(src_folder_path+shuffle_file_name, out_0_folder_path +shuffle_file_name)

# # for shuffle_file_name in shuffle_file_names[50:100]:
# #     shutil.move(src_folder_path+shuffle_file_name, out_1_folder_path +shuffle_file_name)
