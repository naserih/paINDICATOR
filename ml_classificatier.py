#ml_classificatier
import csv
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import resample
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFECV, SelectFromModel #, SequentialFeatureSelector
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
import gc
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
t0 = datetime.now()


# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.serif"] = ['Times']
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
feature_root = os.environ.get("RADIOMICS_FEATURES_PATH")
label_file = os.environ.get("LESION_CENTERS_WITH_LABEL")
output_results = os.environ.get("PAINDICATOR_RESULTS")
class features():
    def __init__(self, filepaths):
        self.values = {}
        self.labels = []
        self.df = []
        self.filenames = []
        for filepath in filepaths:
            value = []
            self.filenames.append(filepath)
            with open(filepath, 'r') as csv_file:
                csvreader = csv.reader(csv_file)
                # print filepath
                header = csvreader.next()
                for row in csvreader:
                    value.append(float(row[1]))
                    if row[0] not in self.values:
                        self.labels.append(row[0])
                        self.values[row[0]] = [float(row[1])]
                    else:
                        self.values[row[0]].append(float(row[1]))
            self.df.append(value)
            # print self.labels



def plot_1D(X, y, names, i_index):
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#10f972', '#ff0000']) 
    plt.scatter(names, X[:, i_index], c=y, cmap=cm_bright,
                       edgecolors='k')
    plt.show()
    

def get_label_metadata(label_file, label_column):
    label_metadata = {}
    with open(label_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        # print header
        # if header[label_column] == 'ct_batch':
        # elif header[label_column] == 'vdp':
        for row in csvreader:
            file_id = "_".join([row[12]]+row[13][1:-1].split(', '))
            label = row[label_column]
            if 'MET' in label:
                y_label = 'MET'
            if 'CTRL' in label:
                y_label = 'CTRL'
            if y_label in label_metadata:
                label_metadata[y_label]['row'].append(row)
                label_metadata[y_label]['file_id'].append(file_id)
            else:
                label_metadata[y_label] = {'row':[row],
                                                    'file_id' : [file_id]}
    # print label_metadata
    return label_metadata


def get_feature_space(feature_path, label_metadata, labels):
    X = None
    y = None
    radiomics_labels_0 = None
    radiomics_fetures_0 = None
    radiomics_fetures_1 = None
    # print label_metadata[labels[0]]['file_id']
    # print ['_'.join(f.split('_')[:-1]) for f in os.listdir(feature_path)]
    # print ('HERE')
     # lc_file_name = 
    label_0_csvs = [os.path.join(feature_path, f) for f in os.listdir(feature_path) if '_'.join(f.split('_')[:-1]) in label_metadata[labels[0]]['file_id'] and 'FAILED' not in f]
    label_1_csvs = [os.path.join(feature_path, f) for f in os.listdir(feature_path) if '_'.join(f.split('_')[:-1]) in label_metadata[labels[1]]['file_id'] and 'FAILED' not in f] 
    # print label_0_csvs  
    # print label_1_csvs     
    radiomics_fetures_0 = features(label_0_csvs)
    radiomics_fetures_1 = features(label_1_csvs)
    radiomics_labels_0 = np.array(radiomics_fetures_0.labels)
    # radiomics_labels_1 = np.array(radiomics_fetures_1.labels)
    file_names = radiomics_fetures_0.filenames + radiomics_fetures_1.filenames
    # print 'feature space size:', np.array(radiomics_fetures_0.df).shape, np.array(radiomics_fetures_1.df).shape
    X = np.array(radiomics_fetures_0.df+radiomics_fetures_1.df)
    # print X.shape , '<<'
    # b = X == X[0,:]
    # print b.shape, '<><><'
    # c = b.all(axis=0)
    # # print c 
    # X = X[:, ~c]
    # print X
    y = np.array([0]*len(radiomics_fetures_0.df)+[1]*len(radiomics_fetures_1.df))
    # radiomics_labels = radiomics_labels_0
    # print 'HERE: ', X.shape, y.shape, radiomics_labels_0.shape
    return X, y, radiomics_labels_0, file_names


def run_classifiers(X, y, labels, rs_method, name_tag):
    classifiers = None
    X_train = None 
    X_test = None  
    y_train = None  
    y_test = None 
    md = 0 # start 
    mc = 16 # 16 total
    pt = datetime.now()
    names = [ 
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
    ][md:md+mc]
    classifiers = [
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        SVC(kernel="linear", C=1),
        MLPClassifier(alpha=1, max_iter=1000),
        MLPClassifier(solver='lbfgs', alpha=0.001,
        hidden_layer_sizes=(15,)),
        # MLPClassifier(alpha=0.001, activation='logistic',
        # hidden_layer_sizes=(15,), max_iter=1000),
        # MLPClassifier(solver='lbfgs', alpha=0.001, activation='logistic',
        # hidden_layer_sizes=(15,), max_iter=1000),
        AdaBoostClassifier(),
        RandomForestClassifier(n_estimators=100, max_features="auto"),
        # RandomForestClassifier(max_depth=8, n_estimators=4, max_features=2),
        # SVC(kernel="linear", class_weight="balanced", probability=True),
        SVC(gamma=2, C=1),
        KNeighborsClassifier(3),
        DecisionTreeClassifier(max_depth=8),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        BaggingClassifier(KNeighborsClassifier(),
                       max_samples=0.5, max_features=0.5)

        ][md:md+mc]
    # print labels
    
    #  Split dataset into training and test part
    # pt = datetime.now()
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.3, random_state=None, stratify = y)
    X_train, y_train = resampling(X_train, y_train, name_tag,rs_method)
    # print 
    print 'TRAN / TEST: \t %i : %i'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds())) 
    pt = datetime.now() 
    # print y_test

    # for i in range(len(labels)):
    #     plot_1D(X_test, y_test, range(len(y_test)), i)
    # print 'Total: ',len(y_test), '\t Class 0: ',len(y_test[y_test==0]),' \t Class 1: ', len(y_test[y_test==1])
    i_index = 0
    j_index = 1
    X_mesh = []
    # iterate over classifiers
    h = 300  # meshsize
    x_min, x_max = X[:, i_index].min() - .5, X[:, i_index].max() + .5
    y_min, y_max = X[:, j_index].min() - .5, X[:, j_index].max() + .5
    # print X.shape
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, h),
                         np.linspace(y_min, y_max, h))
    # print X_mesh

    X_mesh = np.array(X_mesh)
    # print X_mesh
    # print 'X_mesh', X_mesh.shape, X.shape
    # xx, yy = X_mesh[i_index], X_mesh[j_index]
    # print xx.ravel()
    # print xx
   
    # just plot the dataset first
    cm = plt.cm.RdYlGn_r
    cm_bright = ListedColormap(['#00ff00', '#ff0000']) 
    # cm_bright = ListedColormap(['#10f972', '#f60915']) ff0000
    
    # plt.show()
    clf = None
    processed_files = [f for f in 
            os.listdir(os.path.join(output_results,name_tag)) if '.npy' in f]
    for name, clf in zip(names, classifiers):
        
        processed = False
        for processed_file in processed_files:
            if "%s_%s"%(name_tag, name) in processed_file:
                if skip_processed:
                    print "%s_%s, in processed"%(name_tag, name)
                    processed = True
                if reprocess_again:
                    processed = False
                    os.remove(os.path.join(output_results,name_tag, processed_file))
                    print "%s_%s, reprocessing"%(name_tag, name)
        if processed:
            continue
        # fig = plt.figure(figsize=(5, 5))
        fig = None 
        ax = None
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.subplots_adjust(hspace=0.3)
        # ax = _axs.flatten()

        # fig, ax = plt.subplot(1, 1, 1)
        # Plot the training points
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        # plt.xlabel(labels[i_index])
        # plt.ylabel(labels[j_index])
        plt.xlabel('PCA_1')
        plt.ylabel('PCA_2')
        clf.fit(X_train, y_train)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        # score = clf.score(X_test, y_test)
        cv = StratifiedKFold(n_splits=5)
        tprs = []
        fprs = []
        roc_aucs = []
        # print X_train.shape, y_train.shape
        # print 'numpy.unique: ', np.unique(y_train)
        # print 'PLOT PREPER: \t %i : %i %s'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()), name)
        for i, (train, validation) in enumerate(cv.split(X_train, y_train)):
            # pt = datetime.now()
            clf.fit(X_train[train], y_train[train])
            # print 'FIT TRAIN %i: \t %i : %i'%(i, int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()))
            if hasattr(clf, "decision_function"):
                y_score = clf.fit(X_train[train], y_train[train]).decision_function(X_train[validation])
            else:
                y_score = clf.fit(X_train[train], y_train[train]).predict_proba(X_train[validation])[:, 1]

            
            # print X_train[validation].shape
            # print y_score.shape
            # print np.min(y_score), np.max(y_score)
            # pt = datetime.now()
            try:
                fpr, tpr, _ = roc_curve(y_train[validation], y_score)
            except Exception as e:
                y_score = np.array([0]*len(y_score))
                fpr, tpr, _ = roc_curve(y_train[validation], y_score)
                # print e
                print 'val roc_Error'
                # print np.unique(y_score)
                # y_score[y_score == np.nan] = 0.0
                # print e
                # for val in y_score:
                #     print type(val)
            roc_auc = auc(fpr, tpr)
            tprs.append(tpr)
            fprs.append(fpr)
            roc_aucs.append(roc_auc)
            # print 'ROC CURVE %i: \t %i : %i'%(i, int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()))
            
            # print i, X_train[validation].shape, y_score.shape, tpr.shape, fpr.shape
        # print 'scores', scores, score
        # pt = datetime.now()
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        # print 'TOTAL ROC:  \t %i %i %s'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()), name)
            
        # print("accuracy %0.3f  \t stdv: %0.3f" % (scores.mean(), scores.std()))
        # pt = datetime.now()
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        # print 'MODEL PRED: \t %i %i %s'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()), name)
         
        # pt = datetime.now()
        if hasattr(clf, "decision_function"):
            y_score = clf.fit(X_train, y_train).decision_function(X_test)
        else:
            y_score = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
                # y_score = clf.predict_proba(X_train, y_train).decision_function(X_test)
        # print 'Y_SCORE CAL: \t %i %i %s'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()), name)
         
        # print y_score
        try:
                fpr, tpr, _ = roc_curve(y_test, y_score)
        except Exception as e:
                y_score = np.array([0]*len(y_score))
                fpr, tpr, _ = roc_curve(y_test, y_score)
                print 'test roc_Error'
        # print y_score.shape
        # print v
        roc_auc = auc(fpr, tpr)
        # svc_disp = plot_roc_curve(clf, X_test, y_test)
        # roc_curve(y_test.ravel(), y_score.ravel())
        # plt.show()
        # print 'y_score', y_score
        c_m = confusion_matrix(y_test, y_pred)

        # print 'y_test.shape', y_test.shape
        # pt = datetime.now()
        p_r_f1 = precision_recall_fscore_support(y_test, y_pred, average='macro')
        # print 'y_test', y_test
        # print 'y_pred', y_pred
        # print 'p_r_f1', p_r_f1
        # print 'PREC_RECAL: \t %i %i %s'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()), name)
         
        tn = c_m[0, 0]
        fp = c_m[0, 1]
        fn = c_m[1, 0]
        tp = c_m[1, 1]

        # print 'model: %s score: %0.3f \t tp: %s  tn: %s  fp: %s  fn: %s '%(name, score, tp, tn, fp, fn)
        # std_tpr = np.std(tprs, axis=0) 
        # mean_tpr = np.mean(tprs, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
        #                 label=r'$\pm$ 1 std. dev.')
        # pt = datetime.now()
        fig_roc = plt.figure()
        
        inplot_label = 'AUC: %0.3f, R2: %0.3f' % (roc_auc, score)
        inplot_label_5fold = '5-fold mean AUC: %0.3f' % (np.mean(roc_aucs))
        cnt = 0
        for v_tpr, v_fpr in zip(tprs, fprs):
            # print v_fpr 
        # for v in r
            if cnt != 0:
                inplot_label_5fold = ''
            plt.plot(v_fpr, v_tpr, color='pink',
                 lw=1, label=inplot_label_5fold, alpha=1)
            cnt += 1
        # plt.show()
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.plot(fpr, tpr, color='brown', marker = 's',
            mfc='None', ms = 7,
                 lw=1, label=inplot_label)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('%s \n %s'%(' '.join(name_tag.split('_')[1:]), name))
        plt.legend(loc="lower right")
        out_file_name = "%s%s/%s_%s_ROC_%0.3f_%0.3f"%(output_results, name_tag,name_tag, name,roc_auc, score)
        plt.savefig(out_file_name+'.png')
        out_file_name = "%s%s/%s_%s_ROC_%0.3f_%0.3f"%(output_results, name_tag,name_tag, name,roc_auc, score)
        with open (out_file_name+'.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            for j in range(len(tprs)):
                csvwriter.writerow(tprs[j])
                csvwriter.writerow(fprs[j])
            csvwriter.writerow(tpr)
            csvwriter.writerow(fpr)

        # plt.show()
        plt.close(fig_roc)
        fig_roc.clear()
        fig_roc.clf()
        gc.collect()
        with open (out_file_name+'.npy', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['AUC1','AUC2','AUC3','AUC4','AUC5',
                    'R2_1','R2_2','R2_3','R2_4','R2_5',
                    'AUC', 'R2','P','R','F1'])
            csvwriter.writerow(np.concatenate((roc_aucs, scores, 
                [roc_auc, score],p_r_f1)))
        print 'PLOT ROCS : \t %i %i %s'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()), name)
         

        # Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].
        pt = datetime.now()
        if X.shape[1] == 2:
            # print 'HERE'
            fig, ax = plt.subplots(nrows=1, ncols=1)
            fig.subplots_adjust(hspace=0.3)
            pt = datetime.now()
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            # print 'MESH CALC: \t %i %i %s'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()), name)
            pt = datetime.now()
            # print 'Z.shape', Z.shape
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            # print Z.shape
            cset1 = ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            fig.colorbar(cset1, ax=ax)

            # Plot the training points
            ax.set_title("%s_ROI:%s"%(name, name_tag))
            ax.scatter(X_train[:, i_index], X_train[:, j_index], c=y_train, cmap=cm_bright,
                   alpha=0.3, edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, i_index], X_test[:, j_index], c=y_test, cmap=cm_bright, 
                alpha=0.6,   edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            # ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    # size=15, horizontalalignment='right')
            
                # print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
            plt.tight_layout()
            saved_files = {}
            for f in os.listdir(output_results):
                saved_files['_'.join(f.split('_')[:-1])] = f.split('_')[-1] 
            # print saved_files
            if ("%s_%s"%(name, name_tag) not in saved_files  or saved_files["%s_%s"%(name, name_tag)] < score) \
                and score > 0.65:
                plt.savefig("%s%s/%s_%s_%0.3f.jpg"%(output_results, name_tag, name_tag, name, score))
            # plt.show()
            plt.close('all')
            plt.close(fig)
            fig.clear()
            fig.clf()
            gc.collect()
            fig = None
            ax = None
            # print 'PLOT 2D: \t %i %i %s'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()), name)
    clf = None
    name = None    
    X_mesh = None
    xx = None
    yy = None   
    fig_roc = None
    c_m = None  
    cset1 = None
    Z = None

def resampling(X,y,name_tag,method):
    y_0 = np.where(y == 0)[0]
    y_1 = np.where(y == 1)[0]
    # print y_0.shape
    X_0 = X[y_0]
    X_1 = X[y_1]
    # print 'before resampling: ', X_0.shape, X_1.shape
    if method == 'UP':
        y_0 = np.array([0]*len(y_1))
        X_0 = resample(X_0, 
            replace=True,     # sample with replacement
            n_samples=len(y_1),    # to match majority class
            random_state=123) # reproducible results
        X = np.concatenate((X_0, X_1)) 
        y = np.concatenate((y_0, y_1)) 
    elif method == 'DOWN':
        y_1 = np.array([1]*len(y_0))
        X_1 = resample(X_1, 
            replace=True,     # sample with replacement
            n_samples=len(y_0),    # to match majority class
            random_state=123)
        X = np.concatenate((X_0, X_1))
        y = np.concatenate((y_0, y_1)) 
    elif method == 'ROS':
        sampler = RandomOverSampler()
        X, y = sampler.fit_sample(X, y)
    elif method == 'SMOTE':
        sampler = SMOTE(ratio='minority')
        X, y = sampler.fit_sample(X, y)
    elif method == 'RUS':
        sampler = RandomUnderSampler(return_indices=False)
        X, y = sampler.fit_sample(X, y)
    elif method == 'TL':
        sampler = TomekLinks(return_indices=False, ratio='majority')
        X, y = sampler.fit_sample(X, y)
    elif method == 'NONE':
        return X, y
    else:
        print '''WARNING: INVALID RESAMPLING METHOD.
        resampling methodes are; 
        NONE: No resampling
        DOWN: Random reproducable DOWN sampling
        RUS:Random under-sampling 
        TL: Tomek links
        UP: Random reproducable UP sampling
        ROS: random over-sampling
        SMOTE: Synthetic Minority Oversampling TEchnique
        '''

    sampler = None    
    return X, y




def feature_selection(X,y, name_tag, method):
    # Create the RFE object and compute a cross-validated score.
    svc = None
    knn = None
    svc = SVC(kernel="linear")
    knn = KNeighborsClassifier(3),
    
    # sfs = SequentialFeatureSelector(knn, n_features_to_select=20)
    # sfs.fit(X, y)
    # sfs.get_support()
    # print 'transform', sfs.transform(X).shape

    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    if method == 'PCA':
        n_components = 'mle'
        transformer = PCA(n_components=n_components,
            # svd_solver = 'arpack'
            )
    elif method == 'PCA_2':
        n_components = 2
        transformer = PCA(n_components=n_components)
    elif method == 'PCA_10':
        n_components = 10
        transformer = PCA(n_components=n_components)
    elif method == 'PCA_20':
        n_components = 20
        transformer = PCA(n_components=n_components)
    elif method == 'FastICA_2':
        n_components = 2
        transformer = FastICA(n_components=n_components, random_state=0)
    elif method == 'FastICA_10':
        n_components = 10
        transformer = FastICA(n_components=n_components, random_state=0)
    elif method == 'FastICA_20':
        n_components = 20
        transformer = FastICA(n_components=n_components, random_state=0)
    elif method == 'PFECV':
        min_features_to_select = 10
        transformer = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
                  scoring='accuracy',
                  min_features_to_select=min_features_to_select)
    elif method == 'LASSO':
        transformer = LinearSVC(C=0.01, penalty="l1", dual=False)
    elif method == 'TREE':
        transformer = ExtraTreesClassifier(n_estimators=50)
    elif method == 'VT_0.8':    
        transformer = VarianceThreshold(threshold=(.8 * (1 - .8)))
    elif method == 'VT_0.0':    
        transformer = VarianceThreshold()
    elif method == 'NONE':
        return X
    else:
        print 'UNKNOWN FREATURE SELECTION METHOD' 
    try:
        model = SelectFromModel(transformer.fit(X,y), prefit=True)
        X = model.transform(X)
        model = None
    except:
        print 'INFO: transformer used directrly'
        transformer.fit(X,y)
        X = transformer.transform(X)


    if method == 'PFECV':
        fig = None
        # print "Optimal number of features : %d" %transformer.n_features_
        fig = plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(min_features_to_select,
                       len(transformer.grid_scores_) + min_features_to_select),
                 transformer.grid_scores_)
        # plt.show()
        plt.savefig("%s%s/%s_%s.jpg"%(output_results, name_tag,name_tag, transformer.n_features_))
        fig.clear()
        fig.clf()
        plt.close('all')
        plt.close(fig)
        gc.collect()
    transformer = None

    return X

def plot_2d_space(X, y, name_tag):   
    markers = ['D', 's']
    label_names = ['Pain', 'No Pain']
    colors = ['#FF0000', '#00FF00']
    edgecolors = ['#800000', '#008000']
    fig = plt.figure()
    for l, ln, c, m, ec, in zip([1,0], label_names, colors, markers, edgecolors):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=ln, marker=m, edgecolor=ec
        )
    plt.title(name_tag)
    plt.legend(loc='upper right')
    # print output_results
    plt.savefig("%s%s/%s_resampling.jpg"%(output_results, name_tag,name_tag))
    plt.close()
    plt.close('all')
    plt.close(fig)
    fig.clear()
    fig.clf()
    gc.collect()

def plot_feature_vs_class(X,y,feature_names, file_names, name_tag):
    # print set(y)
    marker = ['D', 's']
    label_names = ['Pain', 'No Pain']
    colors = ['#fc3339', '#008001']
    edgecolors = ['r', 'g']
    for iy in range(X.shape[1])[:]:
        # print ix
        data = []
        for label in [1,0]: 
            row_iy = np.where(y == label)[0]
            data.append(X[row_iy, iy])
        fig = plt.figure()
        plt.ylabel(feature_names[iy])
        plt.violinplot(data)
        plt.xticks( [1,2], label_names)
        plt.savefig("%s%s/%s_Viol_%i_%s.jpg"%(output_results, name_tag, name_tag,iy,feature_names[iy]))
        plt.close()
        plt.close('all')
        plt.close(fig)
        fig.clear()
        fig.clf()
        gc.collect()


def main():     
    labels = ['MET', 'CTRL']  
    label_column = 3 # 3 cntrl/met # 13 met type #  VDP score 8 # 
    pt = datetime.now()
    label_metadata = get_label_metadata(label_file,label_column)
    print label_metadata
    # print label_metadata.keys()
    print '--------------'
    print 'GOT LABELS: \t %i : %i'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds())) 
    for label in labels:
        if label not in label_metadata.keys():
            print 'LABEL ERROR: ', label
            break

    '''
    resampling: 0, 
    feature selection: 0-8
    '''
 
    r = 4 # roi
    re = r+1 # 18 total
    o = 0 # 4 RS     0:NONE 1:SMOTE 2:TL 3:ROS 4:RUS 
    oe =o+1 # #RS methods
    n = 0 # 12 FS   0:NONE 1:LASSO 2:TREE 3:PFECV 4:VT_0.8 5:VT_0.0  
    ne = n+1 # FS methods

    all_rois = { 
            'allCY' :   ['CY100','CY70','CY50', 'CY30', 'CY20', 'CY15', 'CY10'],

            'allSP' :   ['SP100','SP70','SP50', 'SP30', 'SP20', 'SP15', 'SP10'],
            'allCYs' : 
                        ['CY100','CY70', 'CY50', 'CY30', 'CY20', 'CY15', 'CY10',
                            'CY5030','CY3050','CY3020', 'CY2030'],
            'allCS' : 
                        ['CY100','CY70', 'CY50', 'CY30', 'CY20', 'CY15', 'CY10',
                            'CY5030','CY3050','CY3020', 'CY2030',
                            'SP100','SP70','SP50', 'SP30', 'SP20', 'SP15', 'SP10'],
            }
    
    rois = [   'SP100','SP70','CY100','CY70',
                'CY50', 'SP50', 'CY5030','CY3050',
                'CY30', 'CY20', 'CY15', 'CY10', #0-6
                'SP30', 'SP20', 'SP15', 'SP10', #7-13
                'CY3020', 'CY2030', #14-17
                'allCY','allCYs', 'allSP', 'allCS', #18-21 
             ][r:re]
    database =  'MNM'
    # labels = ['no pain', 'pain']
    data_resampling_methods = ['NONE', 'SMOTE', 'TL', 'ROS', 'RUS'][o:oe]
    feature_selection_methods = ['NONE', 'LASSO','TREE','PFECV', 'VT_0.8', 'VT_0.0', 
                                    # b'PCA',
                                    'PCA_20', 'FastICA_20', 'FastICA_10', 
                                    'PCA_10', 'PCA_2', 'FastICA_2'][n:ne]
    
    t_cnt = len(feature_selection_methods)*len(data_resampling_methods)*len(rois)
    dts = [0] 
    cnt = 0

    # print rois
    for roi in rois:
            # name_tag = '%s_%s%s'%(database,roi)
            feature_path = '%s/%s'%(feature_root,roi)
            for rs_method in data_resampling_methods:
                gc.collect()
                # print roi
                if os.path.exists(feature_path) or 'all' in roi:
                    # print 'shape: ', X.shape, y.shape, feature_names.shape
                    # print y
                    # Standardize features by removing the mean and scaling to unit variance
                    # rs_method = 'ROS'
                    # print 'RESAMPELED: ', X.shape, y.shape
                    # plot_feature_vs_class(X,y,feature_names, file_names, name_tag)
                    for fs_method in feature_selection_methods:
                        cnt +=1 
                        name_tag = '%s_%s_%s_%s'%(database,roi,fs_method,rs_method)
                        print '-------------'
                        print name_tag
                        pt = datetime.now()
                        t1 = datetime.now()
                        X = None 
                        y = None
                        if  'all' in roi:
                            Xs = []
                            ys = []
                            rfile_names = []
                            rfeature_names = []
                            f_dic = {}
                            f_names = []

                            for r in all_rois[roi]: 
                                r_feature_path = '%s/%s'%(feature_root,r)
                                rX, ry, rfeature_name, rfile_name = get_feature_space(r_feature_path, label_metadata, labels)
                                f_names.append(rfeature_name)
                                # print rX.shape
                                for fi in range(len(rfile_name)):
                                    point_name = '_'.join((rfile_name[fi].split('/')[-1]).split('_')[:-1])
                                    if point_name in f_dic:
                                        f_dic[point_name][0].append(rX[fi,:])
                                        f_dic[point_name][1].append(ry[fi])
                                    else:
                                        f_dic[point_name] = [[rX[fi,:]],[ry[fi]]]
                            X = []
                            y = []
                            for point_name in f_dic:
                                if len(f_dic[point_name][0]) == len(all_rois[roi]):
                                    # print len(f_dic[point_name][0][0])
                                    all_f = np.concatenate(f_dic[point_name][0], axis=0)
                                    # print len(all_f)
                                    # print f_dic[point_name][1]
                                    X.append(all_f)
                                    y.append(f_dic[point_name][1][0])

                                
                            # print rfile_name
                            X = np.array(X)
                            y = np.array(y)
                            feature_names = np.concatenate(f_names, axis=0)


                        else:
                            X, y, feature_names, _ = get_feature_space(feature_path, label_metadata, labels)
                        
                        # print feature_path
                        print X.shape, y.shape
                        
                        # np.concatenate((a, b), axis=1)
                        # print '>>>',  X.shape
                        # print feature_names
                        # print 'FEATURE SPACE: \t %i : %i'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds())) 
                        # pt = datetime.now()
                        # X = StandardScaler().fit_transform(X)
                        # print 'X NRMILIZED: \t %i : %i'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds())) 
                        # pt = datetime.now()
                        # X, y = resampling(X,y,name_tag,rs_method)
                        # print 'X RESAMPLED:\t %i : %i'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds()))         
                        # fs_method = 'PCA_2'
                        if not os.path.exists(os.path.join(output_results, name_tag)):
                            os.mkdir(os.path.join(output_results, name_tag))
                        # pt = datetime.now()
                        X = feature_selection(X,y, name_tag, fs_method)
                        # print  'X.shape', name_tag, X.shape
                        # print 'FEATURE REDUC: \t %i : %i'%(int((datetime.now() - t0).total_seconds()), int((datetime.now() - pt).total_seconds())) 
                        plot_2d_space( X, y,name_tag)
                        # print 'FEATURE REDUCTION: ', X.shape, y.shape 
                        # print chi2(X,y)
                        # X = sel.fit_transform(X)
                        # X = SelectKBest(chi2, k=2).fit_transform(X, y)

                        run_classifiers(X, y, feature_names, rs_method, name_tag)
                        dt = (datetime.now() - t1).total_seconds()
                        
                        if dt > 20:
                            dts.append(dt)
                        print  'REMAINING TIME: %6.2f hrs'%(np.median(dts)*(t_cnt-cnt)/3600.0)




skip_processed = False      ## TRUE: SKIP FALSE: DO NOT REMOVE OLD RESULTS


### DO NOT MAKE THIS TRUE  UNLESS IF YOU WANT TO DELETE
reprocess_again = False    ## TRUE: REMOVES OLD RESULTS
###
###
if __name__ == "__main__":
    main()
