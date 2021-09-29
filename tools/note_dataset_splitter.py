#split_train_test.py
import os
import shutil
import random
import csv
import numpy as np
from dotenv import load_dotenv
load_dotenv()

MET_NOTES_WITH_CT = os.getenv("MET_NOTES_WITH_CT")

n_of_sets = 1
set_size = 50
folder_index_start = 0

src_folder_path = '../../DATA/patients_documents/SPINE_MET_pdfs/ALL/'
out_src_path = '../../DATA/patients_documents/SPINE_MET_pdfs/SET'
out_map_path = '../../DATA/patients_documents/SPINE_MET_pdfs/MAP/'
out_sets = [out_src_path+str(i+folder_index_start)+'/' for i in range(n_of_sets)]
old_scores_path = '../../../Galenus/TEXTractor/static/data/pain_scores/'

mapped_notes = []
with open(MET_NOTES_WITH_CT, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        # print header
        for row in csvreader:
            mapped_notes.append(row[1])

for out_set in out_sets:
    if not os.path.exists(out_set):
        os.mkdir(out_set)

# print mapped_notes
src_files = os.listdir(src_folder_path)
missing_files = [f for f in mapped_notes if f not in src_files]

if len(missing_files) > 0 :
    print 'Error: missing files', missing_files

random.shuffle(mapped_notes)

print 'mapped_notes: ',len(mapped_notes)
if not os.path.exists(out_map_path):
    os.mkdir(out_map_path)

if len(os.listdir(out_map_path)) == len(mapped_notes):
    print 'Already exists. Delete the folder to continue.'
else:   
    for mapped_note in mapped_notes:
        shutil.copy(src_folder_path+mapped_note, out_map_path+mapped_note)
    print 'DONE! copy to MAP'
# print mapped_notes
# 

### compare to the previosly scored notes

old_scores_files = [os.path.join(old_scores_path, f, os.listdir(os.path.join(old_scores_path,f))[0], os.listdir(os.path.join(old_scores_path, f,os.listdir(os.path.join(old_scores_path,f))[0]))[0]) for f in os.listdir(old_scores_path) if '_R' in f]

scored_notes = []
vs = ['na', 'none', 'mild', 'moderate', 'severe']
for old_scores_file in old_scores_files:
    with open(old_scores_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if row[1] in vs:
                scored_notes.append(row[0])
            else:
                print 'missing: ', row
print len(scored_notes), len(list(set(scored_notes)))

scored_notes = sorted(scored_notes)
with_mlt_scores = []
for i in range(1,len(scored_notes)):
    if scored_notes[i] == scored_notes[i-1]:
        if scored_notes[i] not in with_mlt_scores:
            with_mlt_scores.append(scored_notes[i])

# print 'with_mlt_scores: ', len(with_mlt_scores)

with_one_score = [f for f in scored_notes if f not in with_mlt_scores]

# print 'with_one_score: ', len(with_one_score)

valid_no_score = []
valid_one_score = []
valid_mlt_score = []
invalid_scored = []
for note in mapped_notes:
    if note not in scored_notes:
        valid_no_score.append(note)
    elif note in with_one_score:
        valid_one_score.append(note)
    elif note in with_mlt_scores:
        valid_mlt_score.append(note)

for note in scored_notes:
    if note not in mapped_notes:
        if note not in invalid_scored:
            invalid_scored.append(note)

print 'valid_no_score: ', len(valid_no_score)
print 'valid_one_score: ', len(valid_one_score)
print 'valid_mlt_score: ', len(valid_mlt_score)
print 'invalid_scored: ', len(invalid_scored)


random.shuffle(valid_no_score)


tcnt = 0
cnt = 0
set_num = 0

for note in valid_one_score:
    tcnt+=1
    if cnt == set_size:
        cnt = 0
        set_num += 1
    # print(tcnt, cnt, src_folder_path+note, out_sets[set_num])
    cnt+=1
    if cnt ==1 and len(os.listdir(out_sets[set_num])) > 0:
        print 'folder is not emplty: ', out_sets[set_num]
        break
    else:
        shutil.copy(src_folder_path+note, out_sets[set_num]+note)
