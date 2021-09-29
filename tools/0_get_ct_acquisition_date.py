import pydicom
import csv
import numpy as np
import os
import csv
from dotenv import load_dotenv

load_dotenv()


patient_files_path = os.getenv("CT_DATABASE_PATH")
ct_acc_date_out = os.getenv("CT_ACC_DATE")


data_batches = [os.path.join(patient_files_path, f) for f in os.listdir(patient_files_path)]

patient_folders = []
for datapath in data_batches:
    patient_folders += [os.path.join(datapath, f, 'XY') for f in os.listdir(datapath)]

metadate = {}
cnt = 0
for patient_folder in patient_folders:
    patient_cts = [os.path.join(patient_folder, f) for f in os.listdir(patient_folder) if "CT" in f]
    print (cnt, len(patient_cts), patient_folder)
    for ct_file in patient_cts:
        ds = pydicom.read_file(ct_file)
        
        creationDate = ds.InstanceCreationDate 
        creationTime = ds.InstanceCreationTime
        studyDate = ds.StudyDate 
        studyTime = ds.StudyTime
        try:
            seriesDate = ds.SeriesDate
            seriesTime = ds.SeriesTime
        except:
            seriesDate = ""
            seriesTime = ""

        contentDate = ds.ContentDate
        contentTime = ds.ContentTime

        if cnt not in metadate:
            
            metadate[cnt] = {
            'patient_folder': patient_folder,
            'creationDateTime' : [creationDate+creationTime,creationDate+creationTime],
            'studyDateTime' : [studyDate+studyTime,studyDate+studyTime],
            'seriesDateTime' : [seriesDate+seriesTime,seriesDate+seriesTime],
            'contentDateTime' : [contentDate+contentTime,contentDate+contentTime],
            }
        else:
            if creationDate+creationTime != "" and metadate[cnt]['creationDateTime'][0] == '':
                metadate[cnt]['creationDateTime'][0] = creationDate+creationTime
            if creationDate+creationTime != "" and metadate[cnt]['creationDateTime'][1] == '':
                metadate[cnt]['creationDateTime'][1] = creationDate+creationTime
            if creationDate+creationTime != "" and metadate[cnt]['creationDateTime'][0] != '':
                if float(creationDate+creationTime) < float(metadate[cnt]['creationDateTime'][0]):
                    metadate[cnt]['creationDateTime'][0] = creationDate+creationTime
            if creationDate+creationTime != "" and metadate[cnt]['creationDateTime'][0] != '':
                if float(creationDate+creationTime) > float(metadate[cnt]['creationDateTime'][1]):
                    metadate[cnt]['creationDateTime'][1] = creationDate+creationTime
                metadate[cnt]['creationDateTime'][0] = creationDate+creationTime
           
            if studyDate+studyTime != "" and metadate[cnt]['studyDateTime'][0] == '':
                metadate[cnt]['studyDateTime'][0] = studyDate+studyTime
            if studyDate+studyTime != "" and metadate[cnt]['studyDateTime'][1] == '':
                metadate[cnt]['studyDateTime'][1] = studyDate+studyTime
            if studyDate+studyTime != "" and metadate[cnt]['studyDateTime'][0] != '':
                if float(studyDate+studyTime) < float(metadate[cnt]['studyDateTime'][0]):
                    metadate[cnt]['studyDateTime'][0] = studyDate+studyTime
            if studyDate+studyTime != "" and metadate[cnt]['studyDateTime'][0] != '':
                if float(studyDate+studyTime) > float(metadate[cnt]['studyDateTime'][1]):
                    metadate[cnt]['studyDateTime'][1] = studyDate+studyTime
                metadate[cnt]['studyDateTime'][0] = studyDate+studyTime
           
            if contentDate+contentTime != "" and metadate[cnt]['contentDateTime'][0] == '':
                metadate[cnt]['contentDateTime'][0] = contentDate+contentTime
            if contentDate+contentTime != "" and metadate[cnt]['contentDateTime'][1] == '':
                metadate[cnt]['contentDateTime'][1] = contentDate+contentTime
            if contentDate+contentTime != "" and metadate[cnt]['contentDateTime'][0] != '':
                if float(contentDate+contentTime) < float(metadate[cnt]['contentDateTime'][0]):
                    metadate[cnt]['contentDateTime'][0] = contentDate+contentTime
            if contentDate+contentTime != "" and metadate[cnt]['contentDateTime'][0] != '':
                if float(contentDate+contentTime) > float(metadate[cnt]['contentDateTime'][1]):
                    metadate[cnt]['contentDateTime'][1] = contentDate+contentTime
                metadate[cnt]['contentDateTime'][0] = contentDate+contentTime
            
            if seriesDate+seriesTime != "" and metadate[cnt]['seriesDateTime'][0] == '':
                metadate[cnt]['seriesDateTime'][0] = seriesDate+seriesTime
            if seriesDate+seriesTime != "" and metadate[cnt]['seriesDateTime'][1] == '':
                metadate[cnt]['seriesDateTime'][1] = seriesDate+seriesTime
            if seriesDate+seriesTime != "" and metadate[cnt]['seriesDateTime'][0] != '':
                if float(seriesDate+seriesTime) < float(metadate[cnt]['seriesDateTime'][0]):
                    metadate[cnt]['seriesDateTime'][0] = seriesDate+seriesTime
            if seriesDate+seriesTime != "" and metadate[cnt]['seriesDateTime'][0] != '':
                if float(seriesDate+seriesTime) > float(metadate[cnt]['seriesDateTime'][1]):
                    metadate[cnt]['seriesDateTime'][1] = seriesDate+seriesTime
                metadate[cnt]['seriesDateTime'][0] = seriesDate+seriesTime
           
            # if metadate[cnt]['studyDateTime'] != studyDate+'_'+studyTime:
                # print('Error')
            # if metadate[cnt]['contentDate'] != contentDate+'_'+contentTime:
                # print ('Error content')
    cnt += 1

            
        # print(creationDate, creationTime)

# print(metadate)
# print(contentDate,contentTime)

# ct_times = 'ct_times.csv'%cnt
with open(ct_acc_date_out, 'wb') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['patient_folder',
        'MinCreationDateTime', 'MaxCreationDateTime', 
        'MinSeriesDateTime', 'MaxSeriesDateTime',
        'MinStudyDateTime', 'MaxStudyDateTime',
        'MinContentDateTime', 'MaxContentDateTime'])
    for key in metadate:
        csvwriter.writerow([metadate[key]['patient_folder']]+
            metadate[key]['creationDateTime']+
            metadate[key]['seriesDateTime']+
            metadate[key]['studyDateTime']+
            metadate[key]['contentDateTime'])