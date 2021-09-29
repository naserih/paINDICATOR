#collect_labels.py
import os
import csv 
import datetime
from dotenv import load_dotenv
load_dotenv()


'''
This routine is mapping bone lesions to the NLP extracted note.
For healthy bones VDP is assigned to No Pain.
LESION_CENTERS_PATH: INPUT:  path to the folder containing lesion centers (extracted by diCOMBINE)
NLP_VPD_SCORE_LIST: INPUT:  path to the csv file of NLP extracted VDPs (out put of the script "2_extract_vdp_from_score.py")
MANUAL_VPD_SCORE_LIST: INPUT:  path to the csv file of manually extracted VDPs (out put of the script "2_extract_manual_vdp.py")
MET_NOTES_WITH_CT: INPUT:  path to the CSV file with list of notes with mapping CTs (out put of 1_map_note_to_ct.py)
CT_DATABASE_PATH : INPUT:  path to the directory containing CT files
LASION_CENTERS_WITH_LABEL_OUT : OUTPUT: path for the csv file to write VDP and file info for lasion centers
'''

# APPS_ROOT = os.environ.get("APPS_ROOT")
LESION_CENTERS_PATH =  os.environ.get("LESION_CENTERS_PATH")
NLP_VPD_SCORE_LIST = os.environ.get("NOTE_VPD_SCORE")
MANUAL_VALIDATED_VPD_SCORE_LIST = os.environ.get("NOTE_M_V_VPD_SCORE")
MET_NOTES_WITH_CT = os.environ.get('MET_NOTES_WITH_CT')
LESION_CENTERS_WITH_LABEL_OUT = os.environ.get('LESION_CENTERS_WITH_LABEL')
CT_DATABASE_PATH = os.environ.get('CT_DATABASE_PATH')

lc_folders = [os.path.join(LESION_CENTERS_PATH,f) for f in os.listdir(LESION_CENTERS_PATH)]


def get_original_cts():
    original_cts = []
    original_ct_folders = [os.path.join(CT_DATABASE_PATH,f) for f in os.listdir(CT_DATABASE_PATH)]
    for original_ct_folder in original_ct_folders:
        original_cts += [os.path.join(original_ct_folder,f) for f in os.listdir(original_ct_folder)]
    return original_cts      


def get_lc_metadata():

    # print original_cts

    # os.list
    # print original_ct_folders
    lc_metadata = {}
    for lc_folder in lc_folders:
        lc_set = os.listdir(lc_folder)
        # print lc_set
        if len(lc_set) != 1:
            print('FILE TYPE ERROR!')
        lc_files = os.listdir(os.path.join(lc_folder,lc_set[0]))
        for lc_file in lc_files:
            lc_file_path = os.path.join(lc_folder,lc_set[0],lc_file)
            lc_rows = []
            with open(lc_file_path , 'r')  as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    lc_rows.append(row)
                if len(lc_rows) == 0:
                    lc_rows = [['']]
            ct_name = lc_file.split('.')[0]
            # print ct_name
            pid = ct_name.split('_')[0]
            if len(ct_name.split('_')) > 1:
                ct_date = ct_name.split('_')[1]
            else:
                ct_date = ''
            if os.path.join(CT_DATABASE_PATH,lc_set[0],lc_file.split('.')[0]) not in original_cts:
                print 'ERROR: MISSING CT: ', os.path.join(CT_DATABASE_PATH,lc_set[0],lc_file.split('.')[0]) 

            if ct_name not in lc_metadata:
                lc_metadata[ct_name] = {
                'lc_set' :lc_set[0],
                'lc_file_path' : lc_file_path,
                'lesion_centers' : lc_rows,
                'ct_date':ct_date,
                'pid':pid, 
                'ct_name':lc_file.split('.')[0],
                'ct_path': os.path.join(CT_DATABASE_PATH,lc_set[0],lc_file.split('.')[0],'XY'),
                }

            else:
                if lc_metadata[ct_name]['lc_set'] != lc_set[0]:
                    print lc_metadata[ct_name]['lc_file_path'] , lc_file_path

                if lc_metadata[ct_name]['lesion_centers'] != lc_rows:
                    print lc_metadata[ct_name]['lesion_centers'] , lc_rows

                if lc_metadata[ct_name]['ct_date'] != ct_date:
                    print lc_metadata[ct_name]['ct_date'] , ct_date

                if lc_metadata[ct_name]['pid'] != pid:
                    print lc_metadata[ct_name]['pid'] , pid



            # print lc_metadata[ct_name]['ct_path']
    return lc_metadata

def get_nlp_scores():
    pain_scores = {}
    with open(NLP_VPD_SCORE_LIST, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        for row in csvreader:
            note_id = row[0].split('.')[0]
            # print(note_id)
            if note_id in pain_scores:
                print('ERROR: note_id is not unique', note_id)
            pain_scores[note_id] = {
            'note_id' : row[0],
            'n_pains' : row[1],
            'n_nopains' : row[2],
            'api' : row[4],
            }
    return pain_scores

def get_manual_scores():
    manual_scores = {}
    labels = []
    with open(MANUAL_VALIDATED_VPD_SCORE_LIST, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        for row in csvreader:
            note_id = row[0].split('.')[0]
            # comments = [f for f in [row[2*i+2] for i in range(len(row)/2)] if f != '']
            # print comments
            valid_score = row[1]
            if valid_score == 'na':
                api = ''
            elif valid_score == 'none':
                api = -1
            else:
                api = 1
            # print valid_score, api
            # print L
            if note_id in manual_scores:
                print('ERROR: note_id is not unique', note_id)
            manual_scores[note_id] = {
            'note_id' : row[0],
            'pain_score' : valid_score,
            'n_pains' : '',
            'n_nopains' : '',
            'api' : api,
            }
    return manual_scores
def get_met_metadata(pain_scores):
    with_note_ct_keys = []
    met_metadata = {}
    with open(MET_NOTES_WITH_CT, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        # print header
        for row in csvreader:
            ct_key = row[6]
            # print row[6]
            # print ct_key
            note_key = '_'.join(row[1].split("_")[:2])
            with_note_ct_keys.append(ct_key)        

            # print(row[1], note_key)
            if note_key in met_metadata:
                print'ERROR: note_key is not unique', note_key
                break
            if note_key not in pain_scores:
                print pain_scores.keys()
                print'ERROR: note_key not in pain score', note_key#, row[1]
                # continue

            if ct_key not in lc_metadata:
                print 'ERROR: ct_key not in lesion center', ct_key, row[8]
                lc_file = ''
                lesion_centers = []  
            else:
                lc_file = lc_metadata[ct_key]['lc_file_path']
                lesion_centers = lc_metadata[ct_key]['lesion_centers']
            # print row[8]
            met_metadata[note_key] = {
                'pid':row[0],
                'ct_path': row[8],
                'note_name': row[1],
                'ct_batch': row[5],
                'note_id' : pain_scores[note_key]['note_id'],
                'n_pains' : pain_scores[note_key]['n_pains'],
                'n_nopains' : pain_scores[note_key]['n_nopains'],
                'pain_score' : pain_scores[note_key]['pain_score'],
                'api' : pain_scores[note_key]['api'],
                'lc_file' : lc_file,
                'lesion_centers' : lesion_centers,
            }
            if met_metadata[note_key]['api'] == "":
                met_metadata[note_key]['vdp'] = 'undocumented'
            elif float(met_metadata[note_key]['api']) >= 0:
                met_metadata[note_key]['vdp'] = 'pain'
            elif float(met_metadata[note_key]['api']) < 0:
                met_metadata[note_key]['vdp'] = 'no pain'
            else:
                print ('VPD ERROR')
    return met_metadata, with_note_ct_keys

def get_label_metadata(met_metadata):
    label_metadata = []
    label_metadata.append(['pid', 'ct_path', 'note_name', 'ct_batch', 
        'note_id', 'n_pains', 'n_nopains', 'api', 'vdp', 'pain_score', 
        'lc_file', 'file_id', 
        'base_ct_name', 'leasion_center', 'lc_comment', 'met_type'])
    for lc_key in lc_metadata:
        if 'CTRL' in lc_metadata[lc_key]['lc_set']:
            # print (lc_metadata)
            for lesion_center in lc_metadata[lc_key]['lesion_centers']:
                label_metadata.append([
                    lc_metadata[lc_key]['pid'],
                    lc_metadata[lc_key]['ct_path'],
                    '',
                    lc_metadata[lc_key]['lc_set'],
                    '',
                    0,
                    1,
                    -1,
                    'no pain',
                    'none',
                    lc_metadata[lc_key]['lc_file_path'],
                    ]+lesion_center)

    for note_key in met_metadata:
        for lesion_center in met_metadata[note_key]['lesion_centers']:
            label_metadata.append([
                met_metadata[note_key]['pid'],
                met_metadata[note_key]['ct_path'],
                met_metadata[note_key]['note_name'],
                met_metadata[note_key]['ct_batch'],
                met_metadata[note_key]['note_id'],
                met_metadata[note_key]['n_pains'],
                met_metadata[note_key]['n_nopains'],
                met_metadata[note_key]['api'],
                met_metadata[note_key]['vdp'],
                met_metadata[note_key]['pain_score'],
                met_metadata[note_key]['lc_file'],
                ]+lesion_center)
    return label_metadata


def preprocess_labels(label_metadata):
    comments = [
    'in this patient the last two mets were not showed',
    'not bone metastasis',
    # 'Marwan: not in the center',
    'Marwan: not centered properly',
    '(wrong)',
    'I would delete this one',
    'point not aligning to right',
    'not convincing on axial',
    'agree with left rib lesion',
    'point not aligning to right location',
    'point is outside of the vertebrae on the coronal',
    'point is not perfectly centered',
    # 'Marwan: shifted from center',
    'not able to cross-check the lesion',
    'Mame: not sure this is a met',
    'coordinates are wrong',
    'not centered properly ',
    'point is outside of the vertebrae',
    'the point is not well centered',
    'this is not well defined',
    'not well defined',
    'this met is outside of the spine',
    'would consider removing',
    'point not centered on mets',
    'could not identify the met',
    'so the mets are not well seen',
    "I don't think this is a met",
    "I'm not sure this is a bone met",
    "not convinced this is a bone met",
    "this is not a bone met",
    "Marwan: not in the center",
    "point not quite center on sagittal and coronal",
    "VERY ODD APPEARING!",
    " no spine disease remaining",
    "s/p surgery",
    "comment; remove",
    "5 lytic ( large mass, post operative)",
    "between 2 lytic lesions",
    "point not well centered",
    "COMMENT",
    "Comment: sagital and coronal pictures",
    "this is meant to be a comment",
    "Comment: the sagittal and coronal for this one",
    "Comment : sagital and coronal",
    "this point is duplicated so should be",
    "comment wrong"

    ]


    titles = label_metadata[0]
    cleaned_labels = []
    print titles
    for row in label_metadata:
        commented = False 
        if len(row) > 12:
            if '[0, 0, 0]' not in row[13]:
                for comment in comments:
                        if comment in row[14]:
                            commented = True
                            # print row[3],  row[8], row[12], row[13]
                if not commented:
                    if 'CTRL' not in row[3] and (len(row) <16 or (len(row) ==16 and row[15]=='')):
                        print row[0], row[3],  row[8], row[13], row[14]
                    # print len(row), row[3], row[8], row[12], row[13], row[14]
                    while len(row) < 16:
                        row.append("")
                    
                    # print len(row)
                    cleaned_labels.append(row)
                    # pass
                # else:
                #     # commented = False

    return cleaned_labels

def get_label_stat(label_metadata):
    n_total = len(label_metadata)
    n_ctrl = len([row for row in label_metadata if 'CTRL' in row[3]])
    n_met = len([row for row in label_metadata if 'MET' in row[3]])
    n_met_l = len([row for row in label_metadata if 'MET' in row[3] and 'lytic' == row[15]])
    n_met_b = len([row for row in label_metadata if 'MET' in row[3] and 'blastic' == row[15]])
    n_met_m = len([row for row in label_metadata if 'MET' in row[3] and 'mix' == row[15]])
    n_met_pain = len([row for row in label_metadata if 'MET' in row[3] and "pain" == row[8]])
    n_met_npain = len([row for row in label_metadata if 'MET' in row[3] and "no pain" == row[8]])
    n_met_unk = len([row for row in label_metadata if 'MET' in row[3] and "undocumented" == row[8]])
    n_met_l_pain = len([row for row in label_metadata if 'MET' in row[3] and 'lytic' in row[15] and "pain" == row[8]])
    n_met_l_npain = len([row for row in label_metadata if 'MET' in row[3] and 'lytic' in row[15] and "no pain" == row[8]])
    n_met_b_pain = len([row for row in label_metadata if 'MET' in row[3] and 'blastic' in row[15] and "pain" == row[8]])
    n_met_b_npain = len([row for row in label_metadata if 'MET' in row[3] and 'blastic' in row[15] and "no pain" == row[8]])
    
    print "Total: %s CTRL: %s + MET: %s = %s" %(n_total, n_ctrl, n_met, n_met+n_ctrl)
    print "MET: %s lytic: %s + blastic: %s + mix %s = %s" %(n_met, n_met_l, n_met_b, n_met_m, n_met_l+n_met_b+n_met_m)
    print "MET: %s pain: %s + no_pain: %s + unk %s = %s" %(n_met, n_met_pain, n_met_npain, n_met_unk, n_met_pain+n_met_npain+n_met_unk)
    print "LYTIC: %s pain: %s + nopain: %s = %s" %(n_met_l, n_met_l_pain, n_met_l_npain, n_met_l_pain+n_met_l_npain)
    print "BLASTIC: %s pain: %s + nopain: %s = %s" %(n_met_b, n_met_b_pain, n_met_b_npain, n_met_b_pain+n_met_b_npain)
            

original_cts = get_original_cts()
lc_metadata = get_lc_metadata()
# for key in lc_metadata:
#     print lc_metadata[key]
pain_scores = get_nlp_scores()
pain_scores = get_manual_scores()
print pain_scores
print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'

met_metadata, with_note_ct_keys = get_met_metadata(pain_scores)
# for key in lc_metadata:
#     if key not in with_note_ct_keys:
#         print key #lc_metadata[key]
# for key in met_metadata:
#     print met_metadata[key]['ct_batch']
label_metadata = get_label_metadata(met_metadata)
label_metadata = preprocess_labels(label_metadata)
get_label_stat(label_metadata[1:])

with open(LESION_CENTERS_WITH_LABEL_OUT, 'wb') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(label_metadata)

print 'FILE GENERATED: ', LESION_CENTERS_WITH_LABEL_OUT
