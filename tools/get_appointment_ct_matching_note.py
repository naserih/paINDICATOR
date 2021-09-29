import csv
import os
from dotenv import load_dotenv
import requests
import lxml.html as lh
from datetime import datetime
import dictionaries as dc
import numpy as np

load_dotenv()


# PatientId = '0007680'
patientIds = []
# print appointmet_types
MEDPHYS_API = os.getenv("MEDPHYS_API")
APPOINTMENT_URI = os.getenv("APPOINTMENT_URI")
MET_ALL_SPINE_IDS = os.getenv("MET_ALL_SPINE_IDS")
MET_ALL_SPINE_IDS_CT_NOTE_OUT = os.getenv("MET_ALL_SPINE_IDS_CT_NOTE")
MET_ALLSPINE_IDS_CT_NOTE_PATH = os.getenv("MET_ALLSPINE_IDS_CT_NOTE_PATH")
NOTES_ROOTS = [os.getenv("MET_NOTES_ROOT"), os.getenv("CTRL_NOTES_ROOT")]
NOTES_ROOTS_index = ['MET', 'CTRL']
CT_DATABASE_PATH = os.getenv('CT_DATABASE_PATH')
NOTE_VPD_SCORE =  os.getenv("NOTE_VPD_SCORE")
def getAppointments(patientId):
    # print '>>>>>>>>>>>>>>>>>>>>>>>>>>  Downloading: %s'%(patientId)
    appointments_url = "%s/%s?PatientId=%s"%(MEDPHYS_API, APPOINTMENT_URI, patientId)
    # print appointments_url
    response  = requests.get(appointments_url)
    # print requests

    if response.status_code == 200:
            #Store the contents of the website under doc
            doc = lh.fromstring(response.content)
            #Parse data that are stored between <tr>..</tr> of HTML
            tr_elements = doc.xpath('//tr')
            # print tr_elements
            table = {}
            header = []
            i=0
            #For each row, store each first element (header) and an empty list
            for t in tr_elements[0]:
                i+=1
                name=t.text_content().strip()
                header.append(name)
                # print '%d:"%s"'%(i,name)
                # table[name] = []

            # print len(tr_elements)   
            #Since out first row is the header, data is stored on the second row onwards
            for j in range(1,len(tr_elements)):
                #T is our j'th row
                T=tr_elements[j]
                
                #If row is not of size i, the //tr data is not from our table 
                if len(T)!= len(header):
                    print 'Row does not have %d columns' %len(header)
                
                #i is the index of our column
                i=0
                
                #Iterate through each element of the row
                for t in T.iterchildren():
                    data=t.text_content() 
                    data = data.strip().lower()
                    data = data.strip()
                    # print data
                    #Append the data to the empty list of the i'th column
                    if 'Scheduled Date' in header[i]:
                        if data != '':
                            data = datetime.strptime(data.strip(), '%b %d %Y %I:%M%p')
                            # data = data.strftime("%Y%m%dT%H%M%S")
                    elif 'Actual Date' in header[i]:
                        if data != '':
                            data = datetime.strptime(data.strip(), '%b %d %Y %I:%M%p')
                            # data = data.strftime("%Y%m%dT%H%M%S")
                    elif 'Appointment' in header[i]:
                        if data not in dc.appointment_types:
                            # appointment_types.append(data)
                            print 'UNKNOWN APPOINTMENT TYPE: ', data
                        # print data
                        # else:
                        #     print data
                    elif 'Status' in header[i]:
                        if data not in dc.appointment_status_types:
                           print 'UNKNOWN APPOINTMENT STATUS: ', data 
                        # pass

                    if j not in table:
                        table[j] = {header[i]:data}
                    else:
                        table[j][header[i]]=data
                    #Increment i for the next column
                    i+=1
            # print [len(C) for (title,C) in table]
    else:
        print "ERROR getting: %s" %patient

    return table

def export_matching_rows(patientIds):
    matching_rows = [['p_id', 'consult_date', 'ct_date']]
    for patientId in patientIds:
        table = getAppointments(patientId)
        consult_dates = []
        ct_dates = []
        for row in table:
        # print table[row]
            if table[row]['Status'] in dc.valid_status:
                if table[row]['Appointment'] in dc.valid_cosult_types:
                    # print 'consult: ', table[row]['Scheduled Date']
                    consult_date = table[row]['Scheduled Date']
                    consult_dates.append(consult_date)
                elif table[row]['Appointment'] in dc.valid_ct_types:
                    # print 'ct: ', table[row]['Scheduled Date']
                    ct_date = table[row]['Scheduled Date']
                    ct_dates.append(ct_date)
                elif table[row]['Appointment'] in dc.ivalid_appointment_types:
                    pass
                else:
                    # print table[row]
                    pass
        cnt = 0
        for consult_date in consult_dates:
            # print consult_date
            for ct_date in ct_dates:
                # print 
                t_delta = (ct_date-consult_date).total_seconds()/(3600*24) 
                # print t_delta, 'DAYS'
                if t_delta > -1 and t_delta < 10:
                    cnt += 1
                    matching_rows.append([patientId, consult_date.strftime("%Y%m%dT%H%M%S"), ct_date.strftime("%Y%m%dT%H%M%S")])
                    # print consult_date, ct_date
        if cnt == 0:
            for row in table:
                if table[row]['Status'] in dc.valid_status:
                        # print table[row]
                        pass
                        # print table[row]['Scheduled Date'], table[row]['Appointment']
                        
        else:
            # print cnt
            pass 
    
    with open(MET_ALL_SPINE_IDS_CT_NOTE_OUT, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(matching_rows)
     
    # print len(sorted(appointment_types))
    # print sorted(appointment_types)
    return matching_rows

with open(MET_ALL_SPINE_IDS, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        patientIds.append(row[0])


# patientIds = ['0035549']
# matching_rows = export_matching_rows(patientIds)
# matching_rows = []
vdp_metadata = {}
with open(NOTE_VPD_SCORE, 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    for row in csvreader:
        note_id = row[0].split('.')[0]
        if note_id not in vdp_metadata:
            vdp_metadata[note_id] = row[4]
        else:
            print 'ERROR'
# print vdp_metadata
matching_metadata = {}
with open(MET_ALL_SPINE_IDS_CT_NOTE_OUT, 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    # print header
    for row in csvreader:
        # matching_rows.append(row)
        note_date = row[1][:8]
        if row[0] not in matching_metadata:
            matching_metadata[row[0]] = {
            note_date : {'note_date' : [row[1]],
                                'ct_date' : [row[2]]}}
        else:
            if note_date not in matching_metadata[row[0]]:
                matching_metadata[row[0]][note_date] = {'note_date' : [row[1]],
                                'ct_date' : [row[2]]}
            else:
                matching_metadata[row[0]][note_date]['note_date'].append(row[1])
                matching_metadata[row[0]][note_date]['ct_date'].append(row[2])




# print matching_rows

matching_rows_with_path = [['p_id', 'ct_date', 'note_date',  'note_path', 'ct_path', 'api', 'vdp']]
note_metadata = {}
 
for k in range(len(NOTES_ROOTS)):
    INDEX = NOTES_ROOTS_index[k]
    NOTES_ROOT = NOTES_ROOTS[k]
    notes_folders = os.listdir(NOTES_ROOT)
    for folder in notes_folders:
        note_paths = [os.path.join(NOTES_ROOT, folder,f) for f in os.listdir(os.path.join(NOTES_ROOT, folder)) if 'ConsultNote' in f]
        note_dates = [f[:8] for f in os.listdir(os.path.join(NOTES_ROOT, folder)) if 'ConsultNote' in f]
        for i in range(len(note_dates)):
            note_path = note_paths[i]
            note_date = note_dates[i]
            if folder not in note_metadata:
                note_metadata[folder] = {note_date:{'INDEX':[INDEX],
                                                'PATH':[note_path]}}
            else:
                if note_date not in note_metadata[folder]:
                    note_metadata[folder][note_date] = {'INDEX':[INDEX],
                                                'PATH':[note_path]}
                else:
                    note_metadata[folder][note_date]['INDEX'].append(INDEX)
                    note_metadata[folder][note_date]['PATH'].append(note_path)
                    # print '============'

downloaded_met_sets = [f for f in os.listdir(CT_DATABASE_PATH) if 'MET' in f]

downloaded_ct_metadata = {}
for met_set in downloaded_met_sets:
    met_list = os.listdir(os.path.join(CT_DATABASE_PATH, met_set))
    for file_name in met_list:
        p_id = file_name.split('_')[0]
        ct_date = file_name.split('_')[1]
        file_path = os.path.join(CT_DATABASE_PATH, met_set,file_name)
    # for 
        if p_id not in downloaded_ct_metadata:
            downloaded_ct_metadata[p_id] = {'ct_date':[ct_date], 
                                            'file_path':[file_path]}
        else:
            downloaded_ct_metadata[p_id]['ct_date'].append(ct_date)
            downloaded_ct_metadata[p_id]['file_path'].append(file_path)
    
    # print p_id, ct_date 
# print len(downloaded_ct_metadata.keys())
mapped_downloads = []
for p_id in matching_metadata:
    if p_id in note_metadata:
        for n_date in matching_metadata[p_id]:
            if n_date in note_metadata[p_id]:
                # print note_metadata[p_id][n_date]
                if 'CTRL' in note_metadata[p_id][n_date]['INDEX']:
                    print 'ERROR'
                for index in range(len(matching_metadata[p_id][n_date]['note_date'])):
                    note_date = matching_metadata[p_id][n_date]['note_date'][index]
                    ct_date = matching_metadata[p_id][n_date]['ct_date'][index]
                    if p_id in downloaded_ct_metadata:
                        # print ct_date[:8]
                        dl_ct_dates = downloaded_ct_metadata[p_id]['ct_date']
                        dl_ct_paths = downloaded_ct_metadata[p_id]['file_path']
                        if ct_date[:8] in dl_ct_dates:
                            dl_path = dl_ct_paths[dl_ct_dates.index(ct_date[:8])]
                            # print p_id,  ct_date[:8], '<<<'
                        else:
                            ct_datetime = datetime.strptime(ct_date[:8].strip(), '%Y%m%d')
                            dl_ct_delta = [abs(ct_datetime-datetime.strptime(f.strip(), '%Y%m%d')).total_seconds()/(3600*24) for f in dl_ct_dates]
                            dl_ct_argmin = np.argmin(dl_ct_delta)
                            dl_ct_min = np.min(dl_ct_delta)
                            if dl_ct_min < 10:
                                dl_path = dl_ct_paths[dl_ct_argmin]
                                # print p_id,  ct_date[:8], dl_ct_dates[dl_ct_argmin], dl_ct_min
                            else:
                                dl_path = ''
                        # for l in range(len(dl_ct_dates)):
                        #     dl_ct_date = dl_ct_dates[l]
                        #     dl_ct_path = dl_ct_paths[l]
                        #     if dl_ct_date == ct_date[:8]:
                        #         mapped_downloads.append(dl_ct_path)
                        #         print p_id,  ct_date[:8], '<<<-'
                        #     elif dl_ct_path not in mapped_downloads:
                        #         print p_id,  ct_date[:8], dl_ct_date
                    else:
                        dl_path = ''
                        # pass

                    # print vdp_metadata.keys()
                    for note_path in note_metadata[p_id][n_date]['PATH']:
                        note_id = (note_path.split('/')[-1]).split('_')[0]
                        if note_id in vdp_metadata:
                            # print p_id, note_id
                            api = vdp_metadata[note_id]
                            if api == '':
                                vdp = 'undocumented'

                            elif float(api)>= 0:
                                vdp = 'pain'
                            elif float(api) <0:
                                vdp = 'no pain'
                            # pass
                        else:
                            print p_id, note_id, '<<<'
                            vdp = 'UNK'
                            api = 'UNK'
                            # pass
                        # if 'MET7' in dl_path:
                        matching_rows_with_path.append([p_id, ct_date, note_date, note_path, dl_path, api, vdp]) 

                            # matching_rows_with_path.append([p_id, dl_path.split('/')[-1], note_path.split('/')[-1], 'MET7']) 

with open(MET_ALLSPINE_IDS_CT_NOTE_PATH, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(matching_rows_with_path)
     
            # note_metadata[folder] = '1'
        # print note_dates
# print note_metadata
