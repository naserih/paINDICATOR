import os
import csv

match_list = r'C:\Users\FEPC-L389\Google Drive\1_PhDProject\Galenus\data\find_matching_ct4note_all.csv'
map_out_path =  r'C:\Users\FEPC-L389\Google Drive\1_PhDProject\Galenus\data\best_matching_ct4note.csv'
with open(match_list, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    print (header)
    best_match = {}
    for row in csvreader:
        pid = row[0]
        ct_date = row[2]
        nt_date = row[1]
        h_lag = float(row[3])
        d_lag = float(row[4])
        if pid not in best_match:
            if h_lag >= 0:
                best_match[pid] = {
                                ct_date:{
                                    'ct_in_dl' : row[6],
                                    'after_in_dl': row[5],
                                    'before_in_dl': '',
                                    'after_path':row[7],
                                    'before_path':'',
                                    'ct_after_note': nt_date,
                                    'after_lag': d_lag,
                                    'ct_before_note': '',
                                    'before_lag': '',
                                        }
                                }
            else:
                best_match[pid] = {
                                ct_date:{
                                    'ct_in_dl' : row[6],
                                    'after_in_dl': '',
                                    'before_in_dl': row[5],
                                    'after_path':'',
                                    'before_path':row[7],
                                    'ct_after_note': '',
                                    'after_lag': '',
                                    'ct_before_note': nt_date,
                                    'before_lag': d_lag,
                                        }
                                    }
        else:
            if ct_date not in best_match[pid]:
                if h_lag >= 0:
                    best_match[pid][ct_date] = {
                                    'ct_in_dl' : row[6],
                                    'after_in_dl': row[5],
                                    'before_in_dl': '',
                                    'after_path':row[7],
                                    'before_path':'',
                                    'ct_after_note': nt_date,
                                    'after_lag': d_lag,
                                    'ct_before_note': '',
                                    'before_lag': '',
                                            }
                else:
                    best_match[pid][ct_date] = {
                                    'ct_in_dl' : row[6],
                                    'after_in_dl': '',
                                    'before_in_dl': row[5],
                                    'after_path':'',
                                    'before_path':row[7],
                                    'ct_after_note': '',
                                    'after_lag': '',
                                    'ct_before_note': nt_date,
                                    'before_lag': d_lag,
                                        }
            else:
                if h_lag >= 0 and (best_match[pid][ct_date]['before_lag'] == "" or best_match[pid][ct_date]['after_lag'] > d_lag):
                    best_match[pid][ct_date]['ct_after_note'] = nt_date
                    best_match[pid][ct_date]['after_lag'] = d_lag
                    best_match[pid][ct_date]['after_in_dl'] = row[5]
                    best_match[pid][ct_date]['after_path'] = row[7]
                if h_lag < 0 and (best_match[pid][ct_date]['before_lag'] == "" or best_match[pid][ct_date]['before_lag'] > d_lag):
                    best_match[pid][ct_date]['ct_before_note'] = nt_date
                    best_match[pid][ct_date]['before_lag'] = d_lag
                    best_match[pid][ct_date]['before_in_dl'] = row[5]
                    best_match[pid][ct_date]['before_path'] = row[7]

# print (best_match)

with open(map_out_path, 'wb') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['pid', 'ct_date', 'ct_in_dl', 'ct_after_note', 'after_lag', 'after_in_dl','after_path', 'ct_before_note', 'before_lag', 'before_in_dl', 'before_path'])
    for pid in best_match:
        for ct_date in best_match[pid]:
            info = best_match[pid][ct_date]
            csvwriter.writerow([pid, ct_date, info['ct_in_dl'], info['ct_after_note'], info['after_lag'], info['after_in_dl'],info['after_path'],info['ct_before_note'], info['before_lag'], info['before_in_dl'],info['before_path']])
        
        # print (row)
