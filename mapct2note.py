import csv
from datetime  import datetime
cts_dates = 'cts_dates.csv'
notes_dates = 'notes_dates.csv'


# date_time_obj = datetime.strptime('20190202', '%Y%m%d')

notes = {}
with open(notes_dates, 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    note_header = next(csvreader)
    for row in csvreader:
        if row[0] not in notes:
            notes[row[0]] = [[datetime.strptime(row[1], '%Y%m%d')]] # , [datetime.strptime(row[2], '%Y%m%d')]]
        else:
            notes[row[0]][0].append(datetime.strptime(row[1], '%Y%m%d'))
            # notes[row[0]][1].append(datetime.strptime(row[2], '%Y%m%d'))
cts = {}
with open(cts_dates, 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    ct_header = next(csvreader)
    for row in csvreader:
        if row[0] not in cts:
            cts[row[0]] = [[datetime.strptime(row[1], '%Y%m%d')], [datetime.strptime(row[2], '%Y%m%d')]]
        else:
            cts[row[0]][0].append(datetime.strptime(row[1], '%Y%m%d'))
            cts[row[0]][1].append(datetime.strptime(row[2], '%Y%m%d'))


print note_header
print ct_header

for p in notes:
    if p not in cts:
        print ([p])
    else:
        for note_d in notes[p][0]:
            prev_after = 1000
            prev_before = -1000
            after_ct = datetime.strptime("20000101", '%Y%m%d')
            before_ct =datetime.strptime("20000101", '%Y%m%d')
            for ct_d in cts[p][1]:
                delta = (ct_d-note_d).days
                if delta >= 0 and delta < prev_after:
                    prev_after = delta
                    after_ct = ct_d
                if delta < 0 and delta > prev_before:
                    prev_before = delta
                    before_ct = ct_d
            print ([p, note_d.strftime('%Y%m%d'), before_ct.strftime('%Y%m%d'), after_ct.strftime('%Y%m%d'), prev_before, prev_after])
