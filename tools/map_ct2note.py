import os
import csv 
import datetime

notes_path = '../data/notes/TS_MET_notes/'
downloaded_notes_path = '../data/pdfs/TS_MET_notes/ALL/'
downloaded_images_path = '/mnt/iDriveShare/hossein/1_RAWDATA/2016_2020_TSPINE_M2N/'
ts_images_list = '/mnt/iDriveShare/hossein/1_RAWDATA/2018-2019_TSPINE/TSPINE_patientList_2016_2020.csv'
ct2note_map = '../data/ct2note_map_200.csv'
patient_folders = os.listdir(notes_path)
downloaded_images_folders = os.listdir(downloaded_images_path)
downloaded_notes_folders = os.listdir(downloaded_images_path)


downloaded_note_dates = [datetime.datetime.strptime(f.split('_')[1],"%Y%m%dT%H%M%S") for f in os.listdir(downloaded_notes_path)]
downloaded_note_ids = [f.split('_')[0] for f in os.listdir(downloaded_notes_path)]

print downloaded_note_dates
downloaded_images = [f.split('_')[0] for f in os.listdir(downloaded_images_path)]
print downloaded_images

ct_database = {}
with open(ts_images_list, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    for row in csvreader:
        patient_id = row[0]
        # ct_date = row[2]
        # print(ct_date)
        ct_date = datetime.datetime.strptime(row[2],"%b %d %Y %H:%M%p")
        while len(patient_id)<7:
            patient_id = '0'+patient_id

        if patient_id not in ct_database:
            ct_database[patient_id] = {ct_date:row}
        else:
            ct_database[patient_id][ct_date] = row
        # print(patient_id)
        # print(row)

# print(ct_database.keys())

with open(ct2note_map, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['patient_id', 'note_date', 'ct_date', 'time_lag_day', 'note_in_downloads', 'image_in_downloads', 'note_path', 'ct_path'])
    for patient_folder in patient_folders:
        patient_id = patient_folder
        # print(patient_id)
        patient_notes = os.listdir(notes_path+patient_folder)
        for note in patient_notes:
            note_type = note.split("_")[1]
            note_date = datetime.datetime.strptime(note.split("_")[0],"%Y%m%dT%H%M%S")
            # print note_type
            if note_type == 'ConsultNote':
                if patient_id in ct_database:
                    prevDA = 1000
                    prevDB = -1000
                    for ct_date in ct_database[patient_id]:
                        time_lag = (ct_date-note_date).total_seconds()
                        if timelag > 0 and timelag < prevDA:
                            prevDA = time_lag
                        if timelag < 0 and timelag > prevDB:
                            prevDB = time_lag

                        if note_date in downloaded_note_dates:
                            note_in_downloads = 1
                        else:
                            note_in_downloads = 0

                        if patient_id in downloaded_images:
                            image_in_downloads = 1
                        else:
                            image_in_downloads = 0
                        csvwriter.writerow([patient_id, note_date.strftime("%Y%m%d"), ct_date.strftime("%Y%m%d"), int(time_lag/3600), note_in_downloads, image_in_downloads,notes_path+patient_folder+'/'+note]+ct_database[patient_id][ct_date])

                    # print patient_id, note_date, note
                    # print ct_database[patient_id].keys()
