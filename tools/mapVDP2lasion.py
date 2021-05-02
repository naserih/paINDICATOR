#collect_labels.py
import os
import csv 
import datetime
from dotenv import load_dotenv
load_dotenv()

APPS_ROOT = os.environ.get("APPS_ROOT")
LESION_CENTERS_PATH =  os.environ.get("LESION_CENTERS_PATH")
NLP_SCORES_LIST = os.environ.get("NLP_SCORES_LIST")
CTS_WITH_NOTE_LIST = os.environ.get('CTS_WITH_NOTE_LIST')
CT_PAIN_LABELS = os.environ.get('CT_PAIN_LABELS')
LESION_CENTERS_FULLPATH = os.path.join(APPS_ROOT, LESION_CENTERS_PATH)
NLP_SCORES_FILE = os.path.join(APPS_ROOT, NLP_SCORES_LIST)
CTS_WITH_NOTE_FILE = os.path.join(APPS_ROOT, CTS_WITH_NOTE_LIST)
CT_PAIN_LABELS_FILE = os.path.join(APPS_ROOT, CT_PAIN_LABELS)
# print(NLP_SCORES_FILE)

lc_folders = [os.path.join(LESION_CENTERS_FULLPATH,f) for f in os.listdir(LESION_CENTERS_FULLPATH)]
for lc_folder in lc_folders:
    lc_set = [os.path.join(lc_folder,f) for f in os.listdir(lc_folder)]
    if len(lc_set) != 1:
        print('FILT TYPE ERROR!')
    lc_files = [os.path.join(lc_set[0],f) for f in os.listdir(lc_set[0])]
    # print (lc_files)

nlp_scores = {}
with open(NLP_SCORES_FILE, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    for row in csvreader:
        note_id = row[0].split('.')[0]
        # print(note_id)
        if note_id in nlp_scores:
            print('ERROR: note_id is not unique', note_id)
        nlp_scores[note_id] = {
        'note_id' : row[0],
        'n_pains' : row[1],
        'n_nopains' : row[2],
        'api' : row[4],
        }

metadata = {}
with open(CTS_WITH_NOTE_FILE, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    for row in csvreader:
        notekey = row[2].split("_")[0]
        if notekey in metadata:
            print('ERROR: notekey is not unique', notekey)
            break
        if notekey not in nlp_scores:
            # print('ERROR: notekey not in NLP score', notekey, row[2])
            print ( row[2])
        metadata[notekey] = {
            'pid':row[0],
            'ct_name':row[1],
            'note_name': row[2],
            'ct_batch': row[3],
            'note_id' : nlp_scores[notekey]['note_id'],
            'n_pains' : nlp_scores[notekey]['n_pains'],
            'n_nopains' : nlp_scores[notekey]['n_nopains'],
            'api' : nlp_scores[notekey]['api'],
        }
        if metadata[notekey]['api'] == "":
            metadata[notekey]['vdp'] = 'undocumented'
        elif float(metadata[notekey]['api']) >= 0:
            metadata[notekey]['vdp'] = 'pain'
        elif float(metadata[notekey]['api']) < 0:
            metadata[notekey]['vdp'] = 'no pain'
        else:
            print ('VPD ERROR')




with open(CT_PAIN_LABELS_FILE, 'w', newline = '') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['pid', 'ct_name', 'note_name', 'ct_batch', 
        'note_id', 'n_pains', 'n_nopains', 'api', 'vdp'])
    for notekey in metadata:
        csvwriter.writerow([
            metadata[notekey]['pid'],
            metadata[notekey]['ct_name'],
            metadata[notekey]['note_name'],
            metadata[notekey]['ct_batch'],
            metadata[notekey]['note_id'],
            metadata[notekey]['n_pains'],
            metadata[notekey]['n_nopains'],
            metadata[notekey]['api'],
            metadata[notekey]['vdp']
            ])

