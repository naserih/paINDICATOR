import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import csv
from dotenv import load_dotenv
load_dotenv()
paindicator_results = os.environ.get("PAINDICATOR_RESULTS")

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
ANNOT_SIZE = 12
database =  'MNM'
rois = [
        'CY70',
        'CY100',
        'SP70',
        'SP100',
        'CY50', 'CY30','CY20', 'CY15', 'CY10',
        'SP50', 'SP30','SP20', 'SP15', 'SP10',
        'CY5030','CY3050','CY3020', 'CY2030',
        'allCY','allCYs', 'allSP', 'allCS']

data_resampling_methods = ['NONE', 'SMOTE', 'TL', 'ROS', 'RUS']
feature_selection_methods = ['NONE', 'LASSO','TREE','PFECV', 
                          'VT_0.0', 'VT_0.8',
                          'PCA_20', 'FastICA_20', 'PCA_2',  
                          'FastICA_2', 'FastICA_10', 
                          'PCA_10']
ml_models = [    
    "Gaussian Process",      #0
    "Linear SVM",            #1
    "Neural Net",            #2
    "Neural Net relu lbfgs", #3
    # "Neural Net reg",        #4
    # "Neural Net reg lbfgs ", #5
    "AdaBoost",              #6
    "Random Forest 100",     #7
    # "Random Forest",         #8
    # "Balanced Linear SVM",   #9
    "RBF SVM",               #10
    "Nearest Neighbors",     #11    
    "Decision Tree",         #12
    "Naive Bayes",           #13
    "QDA",                   #14
    "Bagging"                #15
]
ml_abrs = [    
    "GPR",      #0
    "L-SVM",            #1
    "NNet",            #2
    "NNet-bfgs", #3
    # "NNet-rg",        #4
    # "NNet-rg-lbfgs ", #5
    "AdaBoost",              #6
    "RF",     #7
    # "RF-2",         #8
    # "BL-SVM",   #9
    "SVM",               #10
    "kNN",     #11    
    "DT",         #12
    "NB",           #13
    "QDA",                   #14
    "Bagging"                #15
]

def get_hyperspace(database, rois, data_resampling_methods, feature_selection_methods, ml_models):
  hyperspace = []
  errorspace = []
  roi_names = []
  for roi_name in rois:
        roi_names.append(roi_name)
        hyperspace.append([])
        errorspace.append([])
        
        # rs_methods = []
        for rs_method in data_resampling_methods:
              hyperspace[roi_names.index(roi_name)].append([])
              errorspace[roi_names.index(roi_name)].append([])
              # rs_methods.append(rs_method)
              # fs_methods = []
              for fs_method in feature_selection_methods:
                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)].append([])
                  errorspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)].append([])
                  # model_names = []
                  auc_values = {}
                  auc_stdvs = {}
                  r2_values = {} 
                  pr_values = {} 
                  rc_values = {} 
                  f1_values = {}
                  # fs_methods.append(fs_method)
                  file_tag = '%s_%s_%s_%s'%(database, roi_name,fs_method,rs_method)
                  # print file_tag
                  file_full_path = os.path.join(paindicator_results, file_tag)
                  performance_files = [f for f in os.listdir(file_full_path) if '.npy' in f]
                  # print file_tag, performance_files
                  for performance_file in performance_files:
                      ml_model = performance_file.split('_')[4]
                      if ml_model in data_resampling_methods:
                        ml_model = performance_file.split('_')[5]
                      if ml_model in ml_models:
                        with open(os.path.join(file_full_path, performance_file), 'rb') as textfile:
                            csvreader = csv.reader(textfile)
                            header = next(csvreader)
                            # print header
                            ## this path keeps the value with best result
                            dataArray = next(csvreader)
                            # print dataArray[:5]
                            auc_sem = stats.sem([float(f) for f in dataArray[:5]])
                            if ml_models.index(ml_model) in auc_values:
                              if auc_values[ml_models.index(ml_model)] < float(dataArray[10]):
                                auc_values[ml_models.index(ml_model)] = float(dataArray[10])
                                auc_stdvs[ml_models.index(ml_model)] = '%i$\pm$%i'%(round(float(dataArray[10])*100),round(auc_sem*100))
                                r2_values[ml_models.index(ml_model)] = float(dataArray[11])
                                pr_values[ml_models.index(ml_model)] = float(dataArray[12])
                                rc_values[ml_models.index(ml_model)] = float(dataArray[13])
                                f1_values[ml_models.index(ml_model)] = float(dataArray[14])
                            else:
                              auc_values[ml_models.index(ml_model)] = float(dataArray[10])
                              auc_stdvs[ml_models.index(ml_model)] = '%i$\pm$%i'%(round(float(dataArray[10])*100),round(auc_sem*100))
                              r2_values[ml_models.index(ml_model)] = float(dataArray[11])
                              pr_values[ml_models.index(ml_model)] = float(dataArray[12])
                              rc_values[ml_models.index(ml_model)] = float(dataArray[13])
                              f1_values[ml_models.index(ml_model)] = float(dataArray[14])
                      else:
                        # print "ESC: ", ml_model
                        pass
                  # print model_names
                  if len(auc_values.keys()) != len(ml_models):
                    print 'Error in ML models: ', auc_values.keys()
                  errorspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][feature_selection_methods.index(fs_method)].append(np.array(auc_stdvs.values()))       
                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][feature_selection_methods.index(fs_method)].append(np.array(auc_values.values()))
                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][feature_selection_methods.index(fs_method)].append(np.array(r2_values.values()))
                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][feature_selection_methods.index(fs_method)].append(np.array(pr_values.values()))
                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][feature_selection_methods.index(fs_method)].append(np.array(rc_values.values()))
                  hyperspace[roi_names.index(roi_name)][data_resampling_methods.index(rs_method)][feature_selection_methods.index(fs_method)].append(np.array(f1_values.values()))
                      

                         
                          # print dataArray

  hyperspace = np.array(hyperspace)
  errorspace = np.array(errorspace)
  
  return hyperspace, errorspace

'''
VARIABLES:

rois =  [ 'CY70','CY100', 'SP70', 'SP100',
        'CY50', 'CY30','CY20', 'CY15', 'CY10',
        'SP50', 'SP30','SP20', 'SP15', 'SP10',
        'CY5030','CY3050','CY3020', 'CY2030',
        'allCY','allCYs', 'allSP', 'allCS']

data_resampling_methods = ['NONE', 'SMOTE', 'TL', 'ROS', 'RUS']

feature_selection_methods = ['NONE', 'LASSO','TREE','PFECV', 'VT_0.0', 'VT_0.8',
                          'PCA_20', 'FastICA_20', 'PCA_2',  
                          'FastICA_2', 'FastICA_10', 'PCA_10']

ml_models = [    
    "Gaussian Process",      #0
    "Linear SVM",            #1
    "Neural Net",            #2
    "Neural Net relu lbfgs", #3
    # "Neural Net reg",        #4
    # "Neural Net reg lbfgs ", #5
    "AdaBoost",              #6
    "Random Forest 100",     #7
    # "Random Forest",         #8
    # "Balanced Linear SVM",   #9
    "RBF SVM",               #10
    "Nearest Neighbors",     #11    
    "Decision Tree",         #12
    "Naive Bayes",           #13
    "QDA",                   #14
    "Bagging"                #15
]


HYPER SPACE
[roi][rs][fs][param][ml]

[params]: [0] ROC-AUC values
          [1] R2 values
          [2] Precision
          [3] Recall
          [4] F-1 Score 


'''
# print len(roi_names), len(data_resampling_methods),len(feature_selection_methods)


roi_i = 4 #   0:C70 1:C100 2:S70 3:S100 
          #   5:C50 1:C30 2:C20 3:C15 4:C10 
          #     S100 S70 5:S50 6:S30 7:S20 8:S15 9:S10
          #     10:C5x3 11:C3x5 12:C3x2 13:C2x3
          #     14:AC, 15:AS, 16:AA
roi_e = roi_i+22

rs_i = 0 #    0:NONE 1:SMOTE 2:TL 3:ROS 4:RUS
rs_e = rs_i+1 

fs_i = 0 #    0:NONE 1:LASSO 2:TREE 3:PFECV 4:VT_0.0 5:VT_0.8,
         #    6:PCA_20 7:FastICA_20 8:PCA_2 9:FastICA_2 10:FastICA_10 11:PCA_10 
fs_e = fs_i+1

ml_i = 0 #    0:GP 1:LSVM 2:NN 3:NN_l 4: NN_rul 5:NN_rgl 6: AB 7:RF100 8:RF
         #    9:BSVM 10:RBF 11:kNN 12:DT 13:NB 14:QDA 15:Bag  
ml_e = ml_i+19

rois = rois[roi_i:roi_e]
data_resampling_methods = data_resampling_methods[rs_i:rs_e]
feature_selection_methods = feature_selection_methods[fs_i:fs_e]
ml_models = ml_models[ml_i:ml_e]

hyperspace, errorspace = get_hyperspace(database, rois, data_resampling_methods, feature_selection_methods, ml_models)
print hyperspace.shape

param = 0 # 0:AUC 1:R2 2:PRECISION 3:RECALL 4:F1-SCORE

GRID_SEARCH = [0,3] # [0:'ROI' 1:'RS' 2:'FS' 3:'ML']

x_variable_name = ["ROI", "Resampling Method", "Feature Selection Method", "ML Model"][GRID_SEARCH[0]]
x_variable = [rois, data_resampling_methods, feature_selection_methods, ml_abrs][GRID_SEARCH[0]]
y_variable_name = ["ROI", "Resampling Method", "Feature Selection Method", "ML Model"][GRID_SEARCH[1]]
y_variable = [rois, data_resampling_methods, feature_selection_methods, ml_abrs][GRID_SEARCH[1]]

if    GRID_SEARCH[0] == 0 and GRID_SEARCH[1] == 1:
  measure = hyperspace[:, :, 0, param, 0]  # ROI/RS
  measure_sdv = errorspace[:, :, 0, 0, 0]  # ROI/RS
  color_map = "YlGn"

elif  GRID_SEARCH[0] == 0 and GRID_SEARCH[1] == 2:
  measure = hyperspace[:, 0, :, param, 0]  # ROI/FS
  measure_sdv = errorspace[:, 0, :, 0, 0]  # ROI/FS
  color_map = 'YlOrRd'
  
elif  GRID_SEARCH[0] == 0 and GRID_SEARCH[1] == 3:
  measure = hyperspace[:, 0, 0, param, :]  # ROI/ML
  measure_sdv = errorspace[:, 0, 0, 0, :]  # ROI/ML
  color_map = "YlOrBr"
  
elif  GRID_SEARCH[0] == 1 and GRID_SEARCH[1] == 2:
  measure = hyperspace[0, :, :, param, 0]  # RS/FS
  measure_sdv = errorspace[0, :, :, 0, 0]  # RS/FS
  color_map = 'Purples'

elif  GRID_SEARCH[0] == 1 and GRID_SEARCH[1] == 3:
  measure = hyperspace[0, :, 0, param, :]  # RS/ML
  measure_sdv = errorspace[0, :, 0, 0, :]  # RS/ML
  color_map = 'Blues'

elif  GRID_SEARCH[0] == 2 and GRID_SEARCH[1] == 3:
  measure = hyperspace[0, 0, :, param, :]  # FS/ML
  measure_sdv = errorspace[0, 0, :, 0, :]  # FS/ML
  color_map = 'Greens'



print measure.shape


df = pd.DataFrame(data=measure, columns=y_variable, index=x_variable)
df_sdv = pd.DataFrame(data=measure_sdv, columns=y_variable, index=x_variable)

'''
Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, 
CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, 
Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, 
Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, 
PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, 
RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, 
Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, 
Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, 
YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, 
bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, 
coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, 
flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, 
gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, 
gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, 
gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, 
jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, 
ocean_r, pink, pink_r, plasma, plasma_r, 
prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, 
summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, 
tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r
'''

# g = sns.clustermap(df, annot=True, cmap=color_map)
# g = sns.clustermap(df, annot=False, cmap=color_map, cbar=False, xticklabels=2, yticklabels=False)

# plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=30, ha="right",
#          rotation_mode="anchor")
# plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, ha="left",
#           rotation_mode="anchor"
#          )
# df = df.pivot(y_variable, y_variable, measure)
# df =df.sort_values(by=y_variable, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
# df =df.sort_values(by=x_variable, axis=1, ascending=True, inplace=False, kind='quicksort', na_position='last')
# df.mean().sort_values()
# df = df.reindex(df.mean(axis=1).sort_values().index, axis=1)
# df = df.reindex(df.mean().sort_values().index, axis=0)

# df = df.assign(m=df.mean(axis=1)).sort_values('m').drop('m', axis=1)
df['m1'] =  df.mean(axis=1)
df.loc['m0'] = df.mean()
# print df.mean()
df_sdv['m1'] =  df.mean(axis=1)
df_sdv.loc['m0'] = df.mean()

df = df.sort_values(by='m1', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
df = df.sort_values(by='m0', axis=1, ascending=True, inplace=False, kind='quicksort', na_position='last')
df = df.drop('m1', axis=1)
df = df.drop('m0', axis=0)

df_sdv = df_sdv.sort_values(by='m1', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
df_sdv = df_sdv.sort_values(by='m0', axis=1, ascending=True, inplace=False, kind='quicksort', na_position='last')
df_sdv = df_sdv.drop('m1', axis=1)
df_sdv = df_sdv.drop('m0', axis=0)
# print int(round(df.shape[1]*3/4))
# print int(round(df.shape[0]*3/4))
plt.figure(figsize=(int(round(df.shape[1]*1)), int(round(df.shape[0]*3/4))))
plt.gcf().subplots_adjust(bottom=.2)
g = sns.heatmap(df, annot=df_sdv, fmt='',annot_kws={"size": ANNOT_SIZE},  cmap=color_map, cbar=False)
plt.xticks(rotation=30,  rotation_mode="anchor", ha='right') 

df.to_csv(path_or_buf='ttestt.csv', sep=',')


plt.show()

