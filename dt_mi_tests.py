#!/usr/bin/env python

import os

import numpy as np
import scipy.io as sio

from sklearn import tree
from sklearn import datasets

from sklearn import __version__ as sklearn_version
if sklearn_version < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

import mutual_info
import dt_mi

# enable off-load MI processing to Matlab
import matlab.engine

def get_matlab_engine():
    eng = matlab.engine.start_matlab()
    # add all the paths that we need to access the estimators that we need
    eng.addpath('/home/kiran/stochasticresearch/depmeas/algorithms')
    eng.addpath('/home/kiran/stochasticresearch/depmeas/algorithms/mex')
    eng.addpath('/home/kiran/stochasticresearch/ite/estimators/base_estimators')
    eng.addpath('/home/kiran/stochasticresearch/ite/shared/embedded/MI_AP/')
    eng.addpath('/home/kiran/stochasticresearch/ite/shared/')

    # various estimators of MI we have available in matlab
    MIN_SAMPS = 3
    matlab_estimators = {}
    matlab_estimators['taukl'] = (eng.taukl_cc_mex,
                                  'Matlab',
                                  [matlab.double([0]),matlab.double([1]),matlab.double([0])], # autoDetect=0,isHybrid=1,continuous=x
                                  MIN_SAMPS) # min # samples
    matlab_estimators['cim']   = (eng.cim,
                                  'Matlab',
                                  [matlab.double([0.015625]),matlab.double([0.2]), # cim algo params, msi & alpha
                                   matlab.double([0]),matlab.double([1]),matlab.double([0])], # autoDetect=0,isHybrid=1,continuous=x
                                  MIN_SAMPS) # min # samples
    matlab_estimators['knn_1'] = (eng.KraskovMI_cc_mex,
                                  'Matlab',
                                  [matlab.double([1])], # #neighbors in KNN estimation = 1
                                  2) # min # samples = 1+K
    matlab_estimators['knn_6'] = (eng.KraskovMI_cc_mex,
                                  'Matlab',
                                  [matlab.double([6])], # #neighbors in KNN estimation = 1
                                  7) # min # samples = 1+K
    matlab_estimators['knn_20'] = (eng.KraskovMI_cc_mex,
                                  'Matlab',
                                  [matlab.double([20])], # #neighbors in KNN estimation = 1
                                  21) # min # samples = 1+K
    matlab_estimators['ap']     = (eng.apMI_interface,
                                  'Matlab',
                                  [], # No additional args required
                                  MIN_SAMPS)
    matlab_estimators['vme']    = (eng.vmeMI_interface,
                                  'Matlab',
                                  [], # No additional args required
                                  MIN_SAMPS)
    
    return (eng,matlab_estimators)

def load_dataset(dataset_name='iris',skew=None):
    main_folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge'
    NUM_FEATURES = 10

    if(dataset_name=='iris'):
        dataset = datasets.load_iris()
        X = dataset.data  # sepal length and petal length
        y = dataset.target
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0)
    elif(dataset_name=='digits'):
        dataset = datasets.load_digits()
        X = dataset.data  # sepal length and petal length
        y = dataset.target
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0)
    elif(dataset_name=='wine'):
        dataset = datasets.load_wine()
        X = dataset.data  # sepal length and petal length
        y = dataset.target
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0)
    elif(dataset_name=='arcene'):
        if(skew==None):
            dataset_fname = os.path.join(main_folder,'arcene','data.mat')
        else:
            dataset_fname = os.path.join(main_folder,'arcene','data_skew_%0.02f.mat' % (skew))
        z = sio.loadmat(dataset_fname)

        # get the CIM selected features and only use top 50 for processing speed :/
        featureVec = sio.loadmat(os.path.join(main_folder,'arcene','arcene_fs_cim.mat'))
        features = np.squeeze(np.asarray(featureVec['featureVec']))-1

        X_train = z['X_train']
        X_train = X_train[:,features[0:NUM_FEATURES]]
        y_train = z['y_train']
        X_valid = z['X_valid']
        X_valid = X_valid[:,features[0:NUM_FEATURES]]
        y_valid = z['y_valid']
        y_train = np.squeeze(np.asarray(y_train))
        y_valid = np.squeeze(np.asarray(y_valid))

    elif(dataset_name=='dexter'):
        if(skew==None):
            dataset_fname = os.path.join(main_folder,'dexter','data.mat')
        else:
            dataset_fname = os.path.join(main_folder,'dexter','data_skew_%0.02f.mat' % (skew))
        z = sio.loadmat(dataset_fname)
        
        # get the CIM selected features and only use top 50 for processing speed :/
        featureVec = sio.loadmat(os.path.join(main_folder,'dexter','dexter_fs_cim.mat'))
        features = np.squeeze(np.asarray(featureVec['featureVec']))-1

        X_train = z['X_train']
        X_train = X_train[:,features[0:NUM_FEATURES]]
        y_train = z['y_train']
        X_valid = z['X_valid']
        X_valid = X_valid[:,features[0:NUM_FEATURES]]
        y_valid = z['y_valid']
        y_train = np.squeeze(np.asarray(y_train))
        y_valid = np.squeeze(np.asarray(y_valid))
    elif(dataset_name=='gisette'):
        if(skew==None):
            dataset_fname = os.path.join(main_folder,'gisette','data.mat')
        else:
            dataset_fname = os.path.join(main_folder,'gisette','data_skew_%0.02f.mat' % (skew))
        z = sio.loadmat(dataset_fname)
        
        # get the CIM selected features and only use top 50 for processing speed :/
        featureVec = sio.loadmat(os.path.join(main_folder,'gisette','gisette_fs_cim.mat'))
        features = np.squeeze(np.asarray(featureVec['featureVec']))-1

        X_train = z['X_train']
        X_train = X_train[:,features[0:NUM_FEATURES]]
        y_train = z['y_train']
        X_valid = z['X_valid']
        X_valid = X_valid[:,features[0:NUM_FEATURES]]
        y_valid = z['y_valid']
        y_train = np.squeeze(np.asarray(y_train))
        y_valid = np.squeeze(np.asarray(y_valid))
    elif(dataset_name=='madelon'):
        if(skew==None):
            dataset_fname = os.path.join(main_folder,'madelon','data.mat')
        else:
            dataset_fname = os.path.join(main_folder,'madelon','data_skew_%0.02f.mat' % (skew))
        z = sio.loadmat(dataset_fname)

        # get the CIM selected features and only use top 50 for processing speed :/
        featureVec = sio.loadmat(os.path.join(main_folder,'madelon','madelon_fs_cim.mat'))
        features = np.squeeze(np.asarray(featureVec['featureVec']))-1

        X_train = z['X_train']
        X_train = X_train[:,features[0:NUM_FEATURES]]
        y_train = z['y_train']
        X_valid = z['X_valid']
        X_valid = X_valid[:,features[0:NUM_FEATURES]]
        y_valid = z['y_valid']
        y_train = np.squeeze(np.asarray(y_train))
        y_valid = np.squeeze(np.asarray(y_valid))

    return (X_train,X_valid,y_train,y_valid)

def main():

    print('Starting Matlab Engine ...')
    engine, matlab_estimators = get_matlab_engine()
    print('Matlab initialized successfully!')

    dataset_list = [#('arcene',None) , ('arcene',0.10), ('arcene',0.25), ('arcene',0.50), ('arcene',0.75),
                    #('dexter',None),  ('dexter',0.10), ('dexter',0.25), ('dexter',0.50), ('dexter',0.75),
                    #('gisette',None), ('gisette',0.10),('gisette',0.25),('gisette',0.50),('gisette',0.75),
                    ('madelon',None), ('madelon',0.10),('madelon',0.25),('madelon',0.50),('madelon',0.75)]
    #X_train, X_test, y_train, y_test = load_dataset('iris')
    for ds in dataset_list:
        X_train, X_test, y_train, y_test = load_dataset(ds[0],skew=ds[1])
        
        max_depth    = None
        random_state = 3

        estimators_to_test = ['taukl','cim','knn_1','knn_6',
                              'knn_20','ap']
        scores = {}
        for estimator in estimators_to_test:
            criterion_args = matlab_estimators[estimator]
            criterion_func_handle = criterion_args[0]
            criterion_transform = criterion_args[1]
            criterion_additional_args = criterion_args[2]
            criterion_minnumsamps = criterion_args[3]
            clf_m = dt_mi.DecisionTree(criterion_func_handle,
                criterion_transform=criterion_transform,criterion_additional_args=criterion_additional_args,
                criterion_minnumsamps=criterion_minnumsamps,
                max_depth=max_depth, random_state=random_state)
            clf_m.fit(X_train, y_train)
            scores[estimator] = clf_m.score(X_test, y_test)

        clf_s = tree.DecisionTreeClassifier(criterion="gini", max_depth=max_depth, random_state=random_state)
        clf_s.fit(X_train, y_train)
        sklearn_score = clf_s.score(X_test ,y_test)
        scores['sklearn'] = sklearn_score
        
        #--- print score
        print("-"*50)
        print(ds)
        print(scores)
    

    # #---print feature importances
    # print("-"*50)
    # f_importance_m = clf_m.feature_importances_
    # f_importance_s = clf_s.feature_importances_

    # print ("my decision tree feature importances:")
    # for f_name, f_importance in zip(np.array(iris.feature_names)[[0,2]], f_importance_m):
    #     print "    ",f_name,":", f_importance

    # print ("sklearn decision tree feature importances:")
    # for f_name, f_importance in zip(np.array(iris.feature_names)[[0,2]], f_importance_s):
    #     print "    ",f_name,":", f_importance
        
    # #--- output decision region
    # plot_result(clf_m, X_train,y_train, X_test, y_test, "my_decision_tree")
    # plot_result(clf_s, X_train,y_train, X_test, y_test, "sklearn_decision_tree")
    
    # #---output decision tree chart
    # tree_ = TreeStructure()
    # dot_data_m = tree_.export_graphviz(clf_m.tree, feature_names=np.array(iris.feature_names)[[0,2]], class_names=iris.target_names)
    # graph_m = pydotplus.graph_from_dot_data(dot_data_m)

    # dot_data_s = tree.export_graphviz(clf_s, out_file=None, feature_names=np.array(iris.feature_names)[[0,2]], class_names=iris.target_names, 
    #                                   filled=True, rounded=True, special_characters=True)  
    # graph_s = pydotplus.graph_from_dot_data(dot_data_s)

    # graph_m.write_png("chart_my_decision_tree.png")
    # graph_s.write_png("chart_sklearn_decision_tree.png")

def plot_result(clf, X_train,y_train, X_test, y_test, png_name):
    X = np.r_[X_train, X_test]
    y = np.r_[y_train, y_test]
    
    markers = ('s','d', 'x','o', '^', 'v')
    colors = ('green', 'yellow','red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    labels = ('setosa', 'versicolor', 'virginica')

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    dx = 0.02
    X1 = np.arange(x1_min, x1_max, dx)
    X2 = np.arange(x2_min, x2_max, dx)
    X1, X2 = np.meshgrid(X1, X2)
    Z = clf.predict(np.array([X1.ravel(), X2.ravel()]).T)
    Z = Z.reshape(X1.shape)

    plt.figure(figsize=(12, 10))
    plt.clf()
    plt.contourf(X1, X2, Z, alpha=0.5, cmap=cmap)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    for idx, cl in enumerate(np.unique(y_train)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=1.0, c=cmap(idx),
                    marker=markers[idx], label=labels[idx])
        
    plt.scatter(x=X_test[:, 0], y=X_test[:, 1], c="", marker="o", s=100,  label="test set")

    plt.title("Decision region(" + png_name + ")")
    plt.xlabel("Sepal length [cm]")
    plt.ylabel("Petal length [cm]")
    plt.legend(loc="upper left")
    plt.grid()
    #--plt.show()
    plt.savefig("decision_region_" + png_name + ".png", dpi=300)

if __name__=='__main__':
    main()