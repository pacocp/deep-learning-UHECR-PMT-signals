# -*- coding: utf-8 -*-
"""
@author: paco
"""
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import argparse

from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score
from read_data import read_computed_variables, read_traces_CNN_muon_elec
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
from utils import plot_conf_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Global Parameters
path_files = "/home/paquillo/TFM/TFM/Datos/Data-paco/train-test-500-1000/"
path_images = "/home/paquillo/TFM/TFM_Code/plots/"
numItera = 5
# params_search = [2**-11, 2**-8, 2**-5, 2**-2, 2, 2**4, 2**7, 2**10]
params_search = [2**-5, 2**-2, 2, 2**4, 2**7]
parameters = {'C':params_search, 'gamma':params_search}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train svm for predicting cosmic air showers")
    parser.add_argument(
        '--name',
        nargs='*',
        dest='name',
        default='plots',
        help='Name for plots',
    )
    parser.add_argument(
        '--onetrace',
        nargs='*',
        dest='one_trace',
        default=1,
        help='If we want to use one or multiple traces',
    )
    parser.add_argument(
        '--onevsall',
        nargs='*',
        dest='one_vs_all',
        default=1,
        help='If we want to do one vs all',
    )

    parser.add_argument(
        '--traces',
        nargs='*',
        dest='traces',
        default=0,
        help='If we want to use traces',
    )

    parser.add_argument(
        '--variables',
        nargs='*',
        dest='variables',
        default='computed',
        help='Which variables to use. Options are: computed, external or all.',
    )

    parser.add_argument(
        '--splits',
        nargs='*',
        dest='n_splits',
        default=5,
        help='How many train, val and test sets you want to create.',
    )

    args = parser.parse_args()
    name = str(args.name[0])
    one_trace = int(args.one_trace[0])
    one_vs_all = int(args.one_vs_all[0])
    traces = int(args.traces[0])
    n_splits = int(args.n_splits[0])
    variables = str(args.variables[0])


if not os.path.exists(path_images+name):
        os.makedirs(path_images+name)

if(not traces):
    '''
    inputs, labels = read_computed_variables(path=path_files, min_distance=500,
                                       max_distance=1000, one_trace=one_trace,
                                        one_vs_all=one_vs_all, variables=variables,
                                        balanced=one_vs_all)
    '''
    if(one_trace):
        if(one_vs_all):
            if(variables == "computed"):
                inputs = np.load('inputs/inputs_computed_2classes_good.npy').item()
            elif(variables == "external"):
                inputs = np.load('inputs/inputs_external_2classes_good.npy').item()
            elif(variables == "all"):
                inputs = np.load('inputs/inputs_computed_and_external_2classes.npy').item()
            else:
                raise("Variables need to be: computed, external or all")
            labels = np.load('inputs/labels_2classes.npy').item()
        else:
            if(variables == "computed"):
                inputs = np.load('inputs/inputs_computed_5classes_good.npy').item()
            elif(variables == "external"):
                inputs = np.load('inputs/inputs_external_5classes_good.npy').item()
            elif(variables == "all"):
                inputs = np.load('inputs/inputs_computed_and_external_5classes.npy').item()
            else:
                raise("Variables need to be: computed, external or all")
            labels = np.load('inputs/labels_5classes.npy').item()
    else:
        if(one_vs_all):
            if(variables == "computed"):
                inputs_exo_tank1 = np.load('inputs/inputs_tank1_computed_2classes_good.npy').item()
                inputs_exo_tank2 = np.load('inputs/inputs_tank2_computed_2classes_good.npy').item()
                inputs_exo_tank3 = np.load('inputs/inputs_tank3_computed_2classes_good.npy').item()
            elif(variables == "external"):
                inputs_exo_tank1 = np.load('inputs/inputs_tank1_external_2classes_good.npy').item()
                inputs_exo_tank2 = np.load('inputs/inputs_tank2_external_2classes_good.npy').item()
                inputs_exo_tank3 = np.load('inputs/inputs_tank3_external_2classes_good.npy').item()
            elif(variables == "all"):
                inputs_exo_tank1 = np.load('inputs/inputs_tank1_computed_and_external_2classes.npy').item()
                inputs_exo_tank2 = np.load('inputs/inputs_tank2_computed_and_external_2classes.npy').item()
                inputs_exo_tank3 = np.load('inputs/inputs_tank3_computed_and_external_2classes.npy').item()
            else:
                raise("Variables need to be: computed, external or all")
            labels = np.load('inputs/labels_3tanks-2classes.npy').item()
        else:
            if(variables == "computed"):
                inputs_exo_tank1 = np.load('inputs/inputs_tank1_computed_5classes_good.npy').item()
                inputs_exo_tank2 = np.load('inputs/inputs_tank2_computed_5classes_good.npy').item()
                inputs_exo_tank3 = np.load('inputs/inputs_tank3_computed_5classes_good.npy').item()
            elif(variables == "external"):
                inputs_exo_tank1 = np.load('inputs/inputs_tank1_external_5classes_good.npy').item()
                inputs_exo_tank2 = np.load('inputs/inputs_tank2_external_5classes_good.npy').item()
                inputs_exo_tank3 = np.load('inputs/inputs_tank3_external_5classes_good.npy').item()
            elif(variables == "all"):
                inputs_exo_tank1 = np.load('inputs/inputs_tank1_computed_and_external_5classes.npy').item()
                inputs_exo_tank2 = np.load('inputs/inputs_tank2_computed_and_external_5classes.npy').item()
                inputs_exo_tank3 = np.load('inputs/inputs_tank3_computed_and_external_5classes.npy').item()
            else:
                raise("Variables need to be: computed, external or all")
            labels = np.load('inputs/labels_3tanks-5classes.npy').item()
else:
    inputs, labels = read_traces_CNN_muon_elec(path=path_files, min_distance=500,
                                                max_distance=1000, one_trace=one_trace,
                                               one_vs_all=one_vs_all, balanced=one_vs_all,
                                               n_splits=n_splits, svm=True)

test_accs = []
val_accs = []
val_sen = []
val_spe = []
test_sen = []
test_spe = []
for split in range(n_splits):
    print("You are on iteration {}".format(split))
    if(one_trace):
        X_test = np.asarray(inputs['test'][split])
        y_test = np.asarray(labels['test'][split])
        # Normalize with mean = 0 and std = 1
        X_train_iter = np.asarray(inputs['train'][split])
        X_val = np.asarray(inputs['val'][split])
        y_train_iter = np.asarray(labels['train'][split])
        y_val = np.asarray(labels['val'][split])
    else:
        X_test_tank1 = np.asarray(inputs_exo_tank1['test'][split])
        X_test_tank2 = np.asarray(inputs_exo_tank2['test'][split])
        X_test_tank3 = np.asarray(inputs_exo_tank3['test'][split])
        X_train_tank1 = np.asarray(inputs_exo_tank1['train'][split])
        X_train_tank2 = np.asarray(inputs_exo_tank2['train'][split])
        X_train_tank3 = np.asarray(inputs_exo_tank3['train'][split])
        X_val_tank1 = np.asarray(inputs_exo_tank1['val'][split])
        X_val_tank2 = np.asarray(inputs_exo_tank2['val'][split])
        X_val_tank3 = np.asarray(inputs_exo_tank3['val'][split])
        X_test = np.concatenate((X_test_tank1,X_test_tank2,X_test_tank3), axis=1)
        X_train_iter = np.concatenate((X_train_tank1,X_train_tank2,X_train_tank3), axis=1)
        X_val = np.concatenate((X_val_tank1,X_val_tank2,X_val_tank3), axis=1)
        y_test = np.asarray(labels['test'][split])
        y_train_iter = np.asarray(labels['train'][split])
        y_val = np.asarray(labels['val'][split])

    # Normalize with mean = 0 and std = 1
    scaler = preprocessing.StandardScaler()
    X_train_iter = scaler.fit_transform(X_train_iter)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if(one_vs_all):
        if(not one_trace and not traces):
            svm_  = svm.SVC(kernel='rbf')
            clf = GridSearchCV(svm_, parameters, cv=5)
        else:
            clf = svm.SVC(gamma='scale')
    else:
        if(not one_trace and not traces):
            svm_ = svm.SVC(kernel='rbf', decision_function_shape='ovo')
            clf = GridSearchCV(svm_, parameters, cv=5)
        else:
            clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    

    clf.fit(X_train_iter, y_train_iter)
    if(not one_trace and not traces):
        print("SVM best configuration is: \n")
        print(clf.best_params_)
    else:
        print("SVM configuration is: \n")
        print(clf)
    
    
    # Validation metrics
    print("Results in val set \n")
    val_accs.append(clf.score(X_val, y_val))
    print("Accuracy val set: {}".format(val_accs[-1]))
    pred = clf.predict(X_val)
    f1 = f1_score(y_val, pred, average=None)
    print("F1 Score\n")
    print(f1)
    print("\n")
    if(one_vs_all):
        cm = confusion_matrix(y_val, pred)
        print("Confusion matrix")
        print(cm)
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        cm_plot = np.asarray([[tp,fn],[fp,tn]])
        sen = tp/(tp + fn)
        spe = tn/(tn+fp)
        val_sen.append(sen)
        val_spe.append(spe)
        print("Sen val set: {}".format(val_sen[-1]))
        print("Spe val set: {}\n".format(val_spe[-1]))

    # Test metrics
    print("Results in test set\n")
    test_accs.append(clf.score(X_test, y_test))
    print("Accuracy test set: {}".format(test_accs[-1]))
    pred = clf.predict(X_test)
    f1 = f1_score(y_test, pred, average=None)
    print("F1 Score\n")
    print(f1)
    print("\n")
    if(one_vs_all):
        plot_name = path_images+name+"/"+"cm"+str(split)+"-"+name+".png"
        cm = confusion_matrix(y_test, pred)
        print("Confusion matrix")
        print(cm)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        sen = tp/(tp + fn)
        spe = tn/(tn+fp)
        test_sen.append(sen)
        test_spe.append(spe)
        print("Sen test set: {}".format(test_sen[-1]))
        print("Spe test set: {}\n".format(test_spe[-1]))
        plot_conf_matrix(cm_plot, classes=['Photon', 'Hadron'],
                              normalize=False,
                              title=name,
                              cmap=plt.cm.Blues,
                              path=plot_name)

print("Mean accuracy in val set: {} +- {}".format(np.mean(val_accs),
                                                np.std(val_accs)))
print("Mean accuracy in test set: {} +- {}".format(np.mean(test_accs),
                                                np.std(test_accs)))
if(one_vs_all):
    print("Mean sensitivity in val set: {} +- {}".format(np.mean(val_sen),
                                                        np.std(val_sen)))
    print("Mean sensitivity in test set: {} +- {}".format(np.mean(test_sen),
                                                    np.std(test_sen)))
    print("Mean specicificity in val set: {} +- {}".format(np.mean(val_spe),
                                                        np.std(val_spe)))
    print("Mean specicificity in test set: {} +- {}".format(np.mean(test_spe),
                                                        np.std(test_spe)))
