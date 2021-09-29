import os
import csv 
from dotenv import load_dotenv
load_dotenv()

'''
This routine is an extension for texTRACTOR to extract manually assigned VDPs.
textractor_pain_score_file: INPUT: path to textractor output 
note_manual_vdp_score: OUTPUT: csv file to store VDP list

'''
textractor_pain_score_file = os.environ.get("TEXTRACTOR_RESULTS")
note_manual_vdp_score = os.environ.get("NOTE_M_VPD_SCORE")

tt_users = [os.path.join(textractor_pain_score_file, f) for f in os.listdir(textractor_pain_score_file)]
tt_folders = [os.path.join(f,os.listdir(f)[0]) for f in tt_users]
tt_pain_score_files = [os.path.join(f,os.listdir(f)[0]) for f in tt_folders]

metadata = {}
users = []
valid_users = ['1415S0', '9265S1', '7932S2', '2643S3', '3832S4', '3589S5',
                '1415R0', '9265R1', '7932R2', '2643R3', '3832R4']
for tt_pain_score_file in tt_pain_score_files:    
    with open(tt_pain_score_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        # header = next(csvreader)
        user = tt_pain_score_file.split('/')[-3].split('_')[0]
        # print user
        if user in valid_users:
            if user not in users:
                users.append(user)
            for row in csvreader:
                # print row
                file_name = "_".join(row[0].split("_")[:2])
                # print file_name
                score = row[1]
                if len(row) > 2:
                    comment = row[2]
                else:
                    comment = ''
                # print score
                if file_name not in metadata:
                    metadata[file_name] = {}
                if user not in  metadata[file_name]:
                    metadata[file_name][user] = {
                                }
                else:
                    user += '1'
                    metadata[file_name][user] = {
                                }
                    users.append(user)
                    print 'ERROR: doublicate user'
                    # print metadata[file_name]

                        
                metadata[file_name][user]['vdp'] = score
                metadata[file_name][user]['comment'] = comment

with open(note_manual_vdp_score , 'wb') as csvfile:
    csvwriter = csv.writer(csvfile)
    header = ['note']+[j for sub in zip(users,users) for j in sub]
    csvwriter.writerow(header)
    for file_name in metadata:
        datarow = []
        for user in users:
            if user in metadata[file_name]:
                datarow.append(metadata[file_name][user]['vdp'])
                datarow.append(metadata[file_name][user]['comment'])
            else:
                datarow.append('')
                datarow.append('')
        # print (key)
        csvwriter.writerow([file_name]+datarow)

print 'DONE!', note_manual_vdp_score