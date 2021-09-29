import os
import csv 
from dotenv import load_dotenv
load_dotenv()

'''
This routine is an extension for NLP pipeline to extract VDP from individual
pain scores for each note.
pain_score_file: INPUT: list of pain scores output of the pain NLP pipeline
note_vdp_score: OUTPUT: csv file to store VDP list

'''
pain_score_file = os.environ.get("PAIN_SCORE_FILE")
note_vdp_score = os.environ.get("NOTE_VPD_SCORE")

metadata = {}
with open(pain_score_file, 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    for row in csvreader:
        file_name = row[0].split("_")[1]
        score = row[1]
        if file_name not in metadata:
            metadata[file_name] = {
            'pain' : 0,
            'nopain': 0,
            'irrelevant' : 0,
            'api' : '',
            }
        if score == "" or score =='-':
            metadata[file_name]['irrelevant'] += 1
        elif score == '0':
            metadata[file_name]['nopain'] += 1
        elif score == '1':
            metadata[file_name]['pain'] += 1
        elif score == '10':
            metadata[file_name]['pain'] += 1
            metadata[file_name]['nopain'] += 1
        else:
            print(score)
        if (metadata[file_name]['pain']+metadata[file_name]['nopain']) != 0:
            metadata[file_name]['api'] = float(metadata[file_name]['pain']-metadata[file_name]['nopain'])/float(metadata[file_name]['pain']+metadata[file_name]['nopain'])

with open(note_vdp_score , 'wb') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['note', 'pain', 'nopain', 'irrelevant','api'])
    for key in metadata:
        # print (key)
        csvwriter.writerow([key, metadata[key]['pain'], metadata[key]['nopain'],metadata[key]['irrelevant'],metadata[key]['api']]) 
        # print(file_name)