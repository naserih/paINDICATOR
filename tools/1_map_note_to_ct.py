import csv
import os
from dotenv import load_dotenv
import numpy as np
import datetime

load_dotenv()

'''
this routine is mapping note date to ct accusition date.
ct accusition datetime is output of the 0_get_ct_accusition_date.py
CT should be within one week of the consultation note.
'''
# print appointmet_types

CT_ACC_DATE = os.getenv('CT_ACC_DATE')
MET_NOTES_ROOT =  os.getenv("MET_NOTES_ROOT")
MET_NOTES_WITH_CT_OUT = os.getenv("MET_NOTES_WITH_CT")

patient_pdfs = os.listdir(MET_NOTES_ROOT)

# print patient_pdfs


ct_database = {}
with open(CT_ACC_DATE, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    # print header
    for row in csvreader:
        # print row
        ct_filepath = row[0]
        ct_filename = ct_filepath.split('/')[-2]
        patient_id = ct_filename.split("_")[0]
        ct_batch = ct_filepath.split('/')[-3]
        # print ct_filename, ct_batch
        ct_date = datetime.datetime.strptime(row[5][:14],"%Y%m%d%H%M%S")
        # print ct_date
        if int(row[5][:8])!= int(row[6][:8]):
            print 'Error in ct_date'
        
        while len(patient_id)<7:
            patient_id = '0'+patient_id

        if patient_id not in ct_database:
            ct_database[patient_id] = {ct_date:[ct_batch, ct_filename, ct_filepath],
            # 'ct_name':ct_filename
            }
        else:
            ct_database[patient_id][ct_date] = [ct_batch, ct_filename, ct_filepath]
            # ct_database[patient_id]['ct_name'] = ct_filename
        # print(patient_id)
        # print(row)

# print(ct_database)


map_file = [['patient_id', 'note_name', 'note_date', 'note_path',  'timelag_day',  'ct_batch', 'ct_name',  'ct_date','ct_path']]
for note in patient_pdfs:
    patient_id = note.split("_")[0] 
    note_date = datetime.datetime.strptime(note.split("_")[1],"%Y%m%dT%H%M%S")
    note_type = note.split("_")[2]
    # print note_type
    if note_type != 'ConsultNote':
        print 'FILE TYPE ERROR'

    if patient_id in ct_database:
        # print patient_id
        for ct_date in ct_database[patient_id]:
            timelag = ((ct_date-note_date).total_seconds())/(3600*24)
            # print  timelag
            if timelag > -2 and timelag < 10:

                # print '>> ', timelag
                map_file.append([patient_id, 
                    note, 
                    note_date.strftime("%Y%m%dT%H%M%S"), 
                    os.path.join(MET_NOTES_ROOT,note),
                    int(timelag), 
                    ct_database[patient_id][ct_date][0],
                    ct_database[patient_id][ct_date][1],
                    ct_date.strftime("%Y%m%dT%H%M%S"), 
                    ct_database[patient_id][ct_date][2]])




with open(MET_NOTES_WITH_CT_OUT, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for row in map_file:
        csvwriter.writerow(row)

print 'GENERATED!', MET_NOTES_WITH_CT_OUT