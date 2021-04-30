#ml_classificatier
import csv
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

class features():
    def __init__(self, filepaths):
        self.values = {}
        self.labels = []
        self.df = []
        for filepath in filepaths:
            value = []
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



def plot_1D(X, y, names, i_index):
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#10f972', '#ff0000']) 
    plt.scatter(names, X[:, i_index], c=y, cmap=cm_bright,
                       edgecolors='k')
    plt.show()
    

def get_dataset(roi, size):
    p_met = 'MET'
    n_met = 'CON'
    print roi, size
    p_met_path  = "/var/www/devDocuments/hossein/Galenus/data/radiomics/TS_%s_%s%s/"%(p_met, roi, size)
    n_met_path  = "/var/www/devDocuments/hossein/Galenus/data/radiomics/TS_%s_%s%s/"%(n_met, roi, size)
    files = os.listdir(p_met_path)+os.listdir(n_met_path)
    p_met_csvs = [os.path.join(p_met_path, f) for f in os.listdir(p_met_path)]
    p_fetures = features(p_met_csvs)
    n_met_csvs = [os.path.join(n_met_path, f) for f in os.listdir(n_met_path)]
    n_fetures = features(n_met_csvs)
    labels = np.array(n_fetures.labels)
    X = np.array(p_fetures.df+n_fetures.df)
    b = X == X[0,:]
    c = b.all(axis=0)
    # print c 
    X = X[:, ~c]
    y = np.array([1]*len(p_fetures.df)+[0]*len(n_fetures.df))
    labels = labels[~c]
    # print X.shape, y.shape, labels.shape
    return X, y, labels, files


def run_classifiers(X, y, labels, files, roi, size):
    output_results = "/var/www/devDocuments/hossein/Galenus/data/results/"
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=8),
        RandomForestClassifier(max_depth=8, n_estimators=4, max_features=2),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
    # print labels

    # Standardize features by removing the mean and scaling to unit variance
    X = StandardScaler().fit_transform(X)
    #  Split dataset into training and test part
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.3, random_state=None)

    # for i in range(len(labels)):
    #     plot_1D(X_test, y_test, range(len(y_test)), i)

    i_index = 0
    j_index = 1
    X_mesh = []
    # iterate over classifiers
    h = 1000  # meshsize
    x_min, x_max = X[:, i_index].min() - .5, X[:, i_index].max() + .5
    y_min, y_max = X[:, j_index].min() - .5, X[:, j_index].max() + .5
    # print X.shape
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, h),
                         np.linspace(y_min, y_max, h))
    # print X_mesh

    X_mesh = np.array(X_mesh)
    # print X_mesh
    # print X_mesh.shape, X.shape
    # xx, yy = X_mesh[i_index], X_mesh[j_index]
    # print xx.ravel()
    # print xx
   
    # just plot the dataset first
    cm = plt.cm.RdYlGn_r
    cm_bright = ListedColormap(['#00ff00', '#ff0000']) 
    # cm_bright = ListedColormap(['#10f972', '#f60915']) ff0000
    
    # plt.show()
    for name, clf in zip(names, classifiers):
        # fig = plt.figure(figsize=(5, 5))
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
        score = clf.score(X_test, y_test)
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        # Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        # print Z.shape
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        # print Z.shape
        cset1 = ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        fig.colorbar(cset1, ax=ax)

        # Plot the training points
        ax.set_title("%s_ROI:%s%s"%(name, roi,size))
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

        if score > 0.65:
            print name, ',', score,',', tp,',', tn,',', fp,',', fn
            # print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
        plt.tight_layout()
        saved_files = {}
        for f in os.listdir(output_results):
            saved_files['_'.join(f.split('_')[:-1])] = f.split('_')[-1] 
        # print saved_files
        if ("%s_%s_%s"%(name, roi, size) not in saved_files  or saved_files["%s_%s_%s"%(name, roi, size)] < score) \
            and score > 0.65:
            plt.savefig("/var/www/devDocuments/hossein/Galenus/data/results/%s_%s_%s_%s.jpg"%(name, roi, size, score))
        # plt.show()
        plt.close('all')
def main():
    rois = [ 'CY'] #'SP', 'CU',
    sizes = ['50']

    for roi in rois:
        for size in sizes:
            if roi == 'CY' and size == '30':
                size = '3050'
            X, y, labels, files = get_dataset(roi, size)
            pca = PCA(n_components=2)
            pca.fit(X)
            X_pca = pca.transform(X)
            X_new = pca.inverse_transform(X_pca)
            X = X[:,0:2]
            # print X.shape, y.shape, X_pca.shape
            # run_classifiers(X, y, labels, files)
            run_classifiers(X_pca, y, labels, files, roi, size)


if __name__ == "__main__":
    main()