#ct2note_map
import os 
import datetime
import numpy as np
import shutil
import csv

CT_DIRECTORY = '../DICOMfortable/static/data/MET'
METSETS = 6
ct_folders = ['%s%s'%(CT_DIRECTORY,i+1) for i in range(METSETS)]

NOTE_DIRECTORY = '../data/notes/TS_MET_notes'
note_folders = os.listdir(NOTE_DIRECTORY)
note_files = {}
for note_folder in note_folders:
    note_files[note_folder] = {}
    files = [f for f in os.listdir(os.path.join(NOTE_DIRECTORY,note_folder)) if 'ConsultNote' in f] 
    for f in files:
        note_files[note_folder][int(f[:8])] = f 
# print(ct_files)
# print(note_files)
mapped_file = [['ct_pid', 'ct', 'match_note', 'batch']]
for i in range(len(ct_folders)):
    ct_folder = ct_folders[i]
    note_set = '../data/notes/MET%i/'%i
    if not os.path.exists(note_set):
        os.mkdir(note_set)
    
    ct_files = os.listdir(ct_folder) 
    for ct in ct_files:
        ct_pid = ct.split('_')[0]
        ct_date = int(ct.split('_')[1])
        ct_lag = ct.split('_')[2]
        # print ct_date
        note_dates = np.array(note_files[ct_pid].keys())
        match_date = note_dates[np.argmin(abs(ct_date-note_dates))]
        # print(ct_date-match_date, ct_lag)
        match_note = note_files[ct_pid][match_date]
        print(match_date, ct_date, ct_date-match_date, ct_lag, ct_pid)
        shutil.copyfile(os.path.join(NOTE_DIRECTORY,ct_pid,match_note), os.path.join(note_set,ct_pid+'_'+match_note))
        mapped_file.append([ct_pid, ct, match_note, 'MET%i'%(i+1)])

matching_files =  '../data/matchedfile_200_splitted.csv'
with open(matching_files, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(mapped_file)
