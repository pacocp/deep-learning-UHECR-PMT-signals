import argparse
import time
import os
import pickle
from read_data import (read_traces_CNN, read_traces_NN,
                       read_traces_CNN_ordered, read_traces_NN_ordered,
                       read_traces_CNN_muon_elec,
                       read_traces_CNN_muon_elec_exo,
                       read_traces_NN_exo,
                       get_from_csv)
from dataset import Dataset, DatasetExo, Dataset3TanksSeparated, Dataset3TanksSeparatedExo
from NN import (NeuralNet, ConvNeuralNet2Dense, ConvNeuralNet2DenseExo,
                ConvNeuralNet2Dense3Tanks, ConvBlock)
from train import train, test
from utils import plot_conf_matrix

from torch.utils import data
import torch
from torch import nn

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.backends import cudnn

# set seed for weight initialization
torch.manual_seed(1234)
np.random.seed(123)
torch.cuda.manual_seed(123)
cudnn.benchmark = True

# Global Parameters
path_files = "/home/paquillo/TFM/TFM/Datos/Data-paco/train-test-500-1000/"
path_images = "/home/paquillo/TFM/TFM_Code/plots/"
path_models = "/home/paquillo/TFM/TFM_Code/models/"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train convolutional neural network for predicting cosmic air showers")
    parser.add_argument(
        '--conv',
        nargs='*',
        dest='conv',
        default=1,
        help='If we want the conv net',
    )
    parser.add_argument(
        '--name',
        nargs='*',
        dest='name',
        default='plots',
        help='Name for plots',
    )
    parser.add_argument(
        '--epochs',
        nargs='*',
        dest='epochs',
        default=40,
        help='Number of epochs',
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
        '--ordered',
        nargs='*',
        dest='ordered',
        default=0,
        help='If you want to order the traces ascending',
    )
    parser.add_argument(
        '--elec',
        nargs='*',
        dest='elec',
        default=0,
        help='If you want to use muonic and electromagnetic traces together',
    )

    parser.add_argument(
        '--exo',
        nargs='*',
        dest='exo',
        default=0,
        help='If you want to use exogenous values',
    )

    parser.add_argument(
        '--threeconvs',
        nargs='*',
        dest='three_convs',
        default=0,
        help='If using three traces you want three different networks',
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

    parser.add_argument(
        '--weights',
        nargs='*',
        dest='weights',
        default='None',
        help='Weights file name.',
    )

    parser.add_argument(
        '--saveweights',
        nargs='*',
        dest='save_weights',
        default=0,
        help='If weights wanted to be saved',
    )

    parser.add_argument(
        '--fc',
        nargs='*',
        dest='fc_units',
        default=10,
        help='Number of neurons for the first dense layer',
    )

    parser.add_argument(
        '--fullbatch',
        nargs='*',
        dest='full_batch',
        default=0,
        help='if the whole batch wants to be used',
    )

    parser.add_argument(
        '--savefig',
        nargs='*',
        dest='save_fig',
        default=0,
        help='if plots want to be saved',
    )

    parser.add_argument(
        '--svm',
        nargs='*',
        dest='svm_',
        default=0,
        help='if SVM wants to be used as classifier',
    )

    args = parser.parse_args()
    conv = int(args.conv[0])
    name = str(args.name[0])
    epochs = int(args.epochs[0])
    one_trace = int(args.one_trace[0])
    one_vs_all = int(args.one_vs_all[0])
    ordered = int(args.ordered[0])
    elec = int(args.elec[0])
    exo = int(args.exo[0])
    three_convs = int(args.three_convs[0])
    variables = str(args.variables[0])
    n_splits = int(args.n_splits[0])
    weights = str(args.weights[0])
    save_weights = int(args.save_weights[0])
    fc_units = int(args.fc_units[0])
    full_batch = int(args.full_batch[0])
    save_fig = int(args.save_fig[0])
    svm_ = int(args.svm_[0])

# Parameters
if(one_trace):
    batch_size = 128
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 12}
else:
    batch_size = 128
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 12}

units_CNN = [10, 10, fc_units, 5]
units_NN = [100, 100]

if(conv):
    # Read traces and labels for CNN
    if(ordered):
        inputs, labels = read_traces_CNN_ordered(path=path_files, min_distance=500,
                                     max_distance=1000, one_trace=one_trace,
                                     one_vs_all=one_vs_all, balanced=one_vs_all)
    elif(exo):
        if(not three_convs):
            '''
            inputs, inputs_exo, labels = read_traces_CNN_muon_elec_exo(path=path_files, min_distance=500,
                                         max_distance=1000, one_trace=one_trace,
                                         one_vs_all=one_vs_all, balanced=one_vs_all,
                                         variables=variables, n_splits=n_splits, verbose=False)
            '''
            print("Loading exo variables and traces...")
            if(one_vs_all):
                inputs = np.load('inputs/inputs_2classes.npy').item()
                labels = np.load('inputs/labels_2classes.npy').item()
                if(variables == "computed"):
                    inputs_exo = np.load('inputs/inputs_computed_2classes_good.npy').item()
                elif(variables == "external"):
                    inputs_exo = np.load('inputs/inputs_external_2classes_good.npy').item()
                elif(variables == "all"):
                    inputs_exo = np.load('inputs/inputs_computed_and_external_2classes.npy').item()
                else:
                    raise("Variables need to be: computed, external or all")
            else:
                inputs = np.load('inputs/inputs_5classes.npy').item()
                labels = np.load('inputs/labels_5classes.npy').item()
                if(variables == "computed"):
                    inputs_exo = np.load('inputs/inputs_computed_5classes_good.npy').item()
                elif(variables == "external"):
                    inputs_exo = np.load('inputs/inputs_external_5classes_good.npy').item()
                elif(variables == "all"):
                    inputs_exo = np.load('inputs/inputs_computed_and_external_5classes.npy').item()
                else:
                    raise("Variables need to be: computed, external or all")
        else:
            print("Loading exo variables and traces...")
            if(one_vs_all):
                inputs_tank1 = np.load('inputs/inputs_tank1_2classes.npy').item()
                inputs_tank2 = np.load('inputs/inputs_tank2_2classes.npy').item()
                inputs_tank3 = np.load('inputs/inputs_tank3_2classes.npy').item()
                labels = np.load('inputs/labels_3tanks-2classes.npy').item()
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
            else:
                inputs_tank1 = np.load('inputs/inputs_tank1_5classes.npy').item()
                inputs_tank2 = np.load('inputs/inputs_tank2_5classes.npy').item()
                inputs_tank3 = np.load('inputs/inputs_tank3_5classes.npy').item()
                labels = np.load('inputs/labels_3tanks-5classes.npy').item()
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
    elif(elec):
        if(not three_convs and one_trace):
            '''
            inputs, labels = get_from_csv(path="datasets/", one_trace=one_trace,
                                          n_splits=n_splits, one_vs_all=one_vs_all,
                                          three_convs=three_convs)
            np.save('inputs/inputs_2classes.npy', inputs)
            np.save('inputs/labels_2classes.npy', labels)

            
            inputs, labels = read_traces_CNN_muon_elec(path=path_files, min_distance=500,
                                         max_distance=1000, one_trace=one_trace,
                                         one_vs_all=one_vs_all, balanced=one_vs_all, 
                                         three_convs=three_convs, n_splits=n_splits, verbose=False)
            '''
            print("Loading data...")
            if(one_vs_all):
                inputs = np.load('inputs/inputs_2classes.npy').item()
                labels = np.load('inputs/labels_2classes.npy').item()
            else:
                inputs = np.load('inputs/inputs_5classes.npy').item()
                labels = np.load('inputs/labels_5classes.npy').item()

            print("Number of samples validation: {}".format(len(inputs['train'][0])))
            print("Labels in train set")
            unique, counts = np.unique(labels['train'][0], return_counts=True)
            print(dict(zip(unique, counts)))
            print("Number of samples validation: {}".format(len(inputs['val'][0])))
            print("Labels in val set")
            unique, counts = np.unique(labels['val'][0], return_counts=True)
            print(dict(zip(unique, counts)))
            print("Number of samples test: {}".format(len(inputs['test'][0])))
            print("Labels in test set")
            unique, counts = np.unique(labels['test'][0], return_counts=True)
            print(dict(zip(unique, counts)))
        
        elif(three_convs and one_trace):
            inputs_tank1, inputs_tank2, inputs_tank3, labels = read_traces_CNN_muon_elec(path=path_files, min_distance=500,
                                         max_distance=1000, one_trace=one_trace,
                                         one_vs_all=one_vs_all, balanced=one_vs_all, 
                                         three_convs=three_convs, n_splits=n_splits, verbose=False)
        else:
            '''
            inputs_tank1, inputs_tank2, inputs_tank3, labels = get_exo_from_csv(path="datasets/", one_trace=one_trace,
                              n_splits=n_splits, one_vs_all=one_vs_all,
                              three_convs=three_convs)
            np.save('inputs/inputs_tank1_2classes.npy', inputs_tank1)
            np.save('inputs/inputs_tank2_2classes.npy', inputs_tank2)
            np.save('inputs/inputs_tank3_2classes.npy', inputs_tank3)
            np.save('inputs/labels_3tanks-2classes.npy', labels)


            inputs_tank1, inputs_tank2, inputs_tank3, labels = read_traces_CNN_muon_elec(path=path_files, min_distance=500,
                                         max_distance=1000, one_trace=one_trace,
                                         one_vs_all=one_vs_all, balanced=one_vs_all, 
                                         three_convs=three_convs, n_splits=n_splits, verbose=False)
            
            np.save('inputs/inputs_tank1_5classes.npy', inputs_tank1)
            np.save('inputs/inputs_tank2_5classes.npy', inputs_tank2)
            np.save('inputs/inputs_tank3_5classes.npy', inputs_tank3)
            np.save('inputs/labels_3tanks-5classes.npy', labels)
            '''
            
            if(one_vs_all):
                inputs_tank1 = np.load('inputs/inputs_tank1_2classes.npy').item()
                inputs_tank2 = np.load('inputs/inputs_tank2_2classes.npy').item()
                inputs_tank3 = np.load('inputs/inputs_tank3_2classes.npy').item()
                labels = np.load('inputs/labels_3tanks-2classes.npy').item()
            else:
                inputs_tank1 = np.load('inputs/inputs_tank1_5classes.npy').item()
                inputs_tank2 = np.load('inputs/inputs_tank2_5classes.npy').item()
                inputs_tank3 = np.load('inputs/inputs_tank3_5classes.npy').item()
                labels = np.load('inputs/labels_3tanks-5classes.npy').item()

            print("Labels in train set")
            unique, counts = np.unique(labels['train'][0], return_counts=True)
            print(dict(zip(unique, counts)))
            print("Number of samples validation: {}".format(len(inputs_tank1['val'][0])))
            print("Labels in val set")
            unique, counts = np.unique(labels['val'][0], return_counts=True)
            print(dict(zip(unique, counts)))
            print("Number of samples test: {}".format(len(inputs_tank1['test'][0])))
            print("Labels in test set")
            unique, counts = np.unique(labels['test'][0], return_counts=True)
            print(dict(zip(unique, counts)))
            
    else:
        inputs, labels = read_traces_CNN(path=path_files, min_distance=500,
                                     max_distance=1000, one_trace=one_trace,
                                     one_vs_all=one_vs_all, balanced=one_vs_all)
else:
    # Read traces and labels for NN
    if(ordered):
        inputs, labels = read_traces_NN_ordered(path=path_files, min_distance=500,
                                    max_distance=1000, one_trace=one_trace,
                                    one_vs_all=one_vs_all)
    else:
        inputs, labels = read_traces_NN_exo(path=path_files, min_distance=500,
                                    max_distance=1000, one_trace=one_trace,
                                    one_vs_all=one_vs_all)
if(elec and one_trace):
    input_channels = 3
elif(elec and not one_trace and not three_convs):
    input_channels = 6
elif(elec and not one_trace and three_convs):
    input_channels = 3
elif(one_trace):
    input_channels = 1
else:
    input_channels = 3

if(one_vs_all):
    output_size = 2
else:
    output_size = 5

test_accs = []
val_accs = []
val_sen = []
val_spe = []
test_sen = []
test_spe = []

# Creating folders for plots and models
if(save_fig):
    if not os.path.exists(path_images+name):
        os.makedirs(path_images+name)
if(save_weights):
    if not os.path.exists(path_models+name):
        os.makedirs(path_models+name)

for split in range(n_splits):
    # If we are not using 3 conv architecture
    if(not three_convs):
        X_test = np.asarray(inputs['test'][split])
        y_test = np.asarray(labels['test'][split])
    else:
        X_test_tank1 = np.asarray(inputs_tank1['test'][split])
        X_test_tank2 = np.asarray(inputs_tank2['test'][split])
        X_test_tank3 = np.asarray(inputs_tank3['test'][split])  
        
        y_test = np.asarray(labels['test'][split])

    if(exo):
        if(one_trace):
            X_test_exo = np.asarray(inputs_exo['test'][split])
        else:
            X_test_exo_tank1 = np.asarray(inputs_exo_tank1['test'][split])
            X_test_exo_tank2 = np.asarray(inputs_exo_tank2['test'][split])
            X_test_exo_tank3 = np.asarray(inputs_exo_tank3['test'][split])
       
    # Generating dataset generator for test
    if(full_batch and not three_convs):
        # Parameters
        params = {'batch_size': X_test.shape[0],
                  'shuffle': True,
                  'num_workers': 12}
    elif(full_batch and three_convs):
        # Parameters
        params = {'batch_size': X_test_tank1.shape[0],
                  'shuffle': True,
                  'num_workers': 12}
    
    if(exo):
        if(not three_convs):
            model = ConvNeuralNet2DenseExo(input_channels=1,
                                           units=units_CNN,
                                           output_size=output_size,
                                           input_size=[batch_size, 1, input_channels, 200],
                                           exo_size=[batch_size, X_test_exo.shape[1]])
            #print("Loading weights...")
            #model.load_my_state_dict(torch.load("models/"+weights+"/"+weights+"-"+str(split)))
            #model.freeze_layers()
        else:
            model_aux = ConvNeuralNet2DenseExo(input_channels=1,
                                        units=units_CNN,
                                        output_size=output_size,
                                        input_size=[batch_size, 1, input_channels, 200],
                                        exo_size=[batch_size, X_test_exo_tank1.shape[1]])
            print("Loading weights...")
            model_aux.load_state_dict(torch.load("models/"+weights+"/"+weights+"-"+str(split)))
            block1 = ConvBlock(input_channels=1,
                               units=units_CNN,
                               output_size=output_size,
                               input_size=[batch_size, 1, input_channels, 200],
                               exo_size=[batch_size, X_test_exo_tank1.shape[1]])
            block2 = ConvBlock(input_channels=1,
                               units=units_CNN,
                               output_size=output_size,
                               input_size=[batch_size, 1, input_channels, 200],
                               exo_size=[batch_size, X_test_exo_tank1.shape[1]])
            block3 = ConvBlock(input_channels=1,
                               units=units_CNN,
                               output_size=output_size,
                               input_size=[batch_size, 1, input_channels, 200],
                               exo_size=[batch_size, X_test_exo_tank1.shape[1]])

            block1.load_my_state_dict(model_aux.state_dict())
            block2.load_my_state_dict(model_aux.state_dict())
            block3.load_my_state_dict(model_aux.state_dict())
            
            block1.freeze_layers(svm_)
            block2.freeze_layers(svm_)
            block3.freeze_layers(svm_)
            
            conv_blocks = [block1, block2, block3]
            
            model = ConvNeuralNet2Dense3Tanks(input_channels=1,
                                        units=units_CNN,
                                        output_size=output_size,
                                        conv_blocks=conv_blocks, 
                                        input_size=[batch_size, 1, input_channels, 200],
                                        exo_size=[batch_size, X_test_exo_tank1.shape[1]])
    elif(conv):
        if(not three_convs):
            model = ConvNeuralNet2Dense(input_channels=1,
                                        units=units_CNN,
                                        output_size=output_size,
                                        input_size=[batch_size, 1, input_channels, 200])
            if(svm_):
                print("Loading weights...")
                model.load_my_state_dict(torch.load("models/"+weights+"/"+weights+"-"+str(split)))
                block = ConvBlock(input_channels=input_channels,
                                   units=units_CNN,
                                   output_size=output_size,
                                   input_size=[batch_size, input_channels, 200])

                block.load_my_state_dict(model.state_dict())
                block.freeze_layers(svm_)

                model = block
        elif(three_convs and not one_trace):
            model_aux = ConvNeuralNet2Dense(input_channels=1,
                                        units=units_CNN,
                                        output_size=output_size,
                                        input_size=[batch_size, 1, input_channels, 200])
            print("Loading weights...")
            model_aux.load_my_state_dict(torch.load("models/"+weights+"/"+weights+"-"+str(split)))
            block1 = ConvBlock(input_channels=1,
                               units=units_CNN,
                               output_size=output_size,
                               input_size=[batch_size, 1, input_channels, 200],
                               exo_size=[])
            block2 = ConvBlock(input_channels=1,
                               units=units_CNN,
                               output_size=output_size,
                               input_size=[batch_size, 1, input_channels, 200],
                               exo_size=[])
            block3 = ConvBlock(input_channels=1,
                               units=units_CNN,
                               output_size=output_size,
                               input_size=[batch_size, 1, input_channels, 200],
                               exo_size=[])
            
            block1.load_my_state_dict(model_aux.state_dict())
            block2.load_my_state_dict(model_aux.state_dict())
            block3.load_my_state_dict(model_aux.state_dict())
            
            block1.freeze_layers(svm_)
            block2.freeze_layers(svm_)
            block3.freeze_layers(svm_)
            
            conv_blocks = [block1, block2, block3]

            model = ConvNeuralNet2Dense3Tanks(input_channels=1,
                                              units=units_CNN,
                                              output_size=output_size,
                                              conv_blocks=conv_blocks, 
                                              input_size=[batch_size, 1, input_channels, 200])
        elif(three_convs and one_trace):
            model_aux = ConvNeuralNet2Dense(input_channels=input_channels,
                                        units=units_CNN,
                                        output_size=output_size,
                                        input_size=[batch_size, input_channels, 200])

            block1 = ConvBlock(input_channels=input_channels,
                               units=units_CNN,
                               output_size=output_size,
                               input_size=[batch_size, input_channels, 200])
            block2 = ConvBlock(input_channels=input_channels,
                               units=units_CNN,
                               output_size=output_size,
                               input_size=[batch_size, input_channels, 200])
            block3 = ConvBlock(input_channels=input_channels,
                               units=units_CNN,
                               output_size=output_size,
                               input_size=[batch_size, input_channels, 200])

            conv_blocks = [block1, block2, block3]

            model = ConvNeuralNet2Dense3Tanks(input_channels=input_channels,
                                              units=units_CNN,
                                              output_size=output_size,
                                              conv_blocks=conv_blocks, 
                                              input_size=[batch_size, input_channels, 200],
                                              exo_size=[], 
                                              exo=False)
    else:
        model = NeuralNet(input_size=inputs.shape[2],
                          units=units_NN,
                          output_size=output_size)
    model = model.cuda()

    if(three_convs):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=0.01, #eps=1e-08,
                                 momentum=0.9000,
                                 weight_decay=0.0000)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, #eps=1e-08,
                                 momentum=0.9000,
                                 weight_decay=0.0000)

    if(not three_convs):
        X_train_iter = np.asarray(inputs['train'][split])
        X_val = np.asarray(inputs['val'][split])
        y_train_iter = np.asarray(labels['train'][split])
        y_val = np.asarray(labels['val'][split])
        # Normalize with mean = 0 and std = 1
        for i_c in range(input_channels):
            scaler = preprocessing.StandardScaler()
            X_train_iter[:, i_c, :] = scaler.fit_transform(X_train_iter[:, i_c, :])
            X_val[:, i_c, :] = scaler.transform(X_val[:, i_c, :])
            X_test[:, i_c, :] = scaler.transform(X_test[:, i_c, :])
    elif(three_convs and one_trace):
        X_train_iter_tank1 = np.asarray(inputs_tank1['train'][split])
        X_val_tank1 = np.asarray(inputs_tank1['val'][split])
        X_train_iter_tank2 = np.asarray(inputs_tank2['train'][split])
        X_val_tank2 = np.asarray(inputs_tank2['val'][split])
        X_train_iter_tank3 = np.asarray(inputs_tank3['train'][split])
        X_val_tank3 = np.asarray(inputs_tank3['val'][split])
        y_train_iter = np.asarray(labels['train'][split])
        y_val = np.asarray(labels['val'][split])

        # Normalize with mean = 0 and std = 1
        scaler_1 = preprocessing.StandardScaler()
        X_train_iter_tank1 = scaler_1.fit_transform(X_train_iter_tank1)
        scaler_2 = preprocessing.StandardScaler()
        X_train_iter_tank2 = scaler_2.fit_transform(X_train_iter_tank2)
        scaler_3 = preprocessing.StandardScaler()
        X_train_iter_tank3 = scaler_3.fit_transform(X_train_iter_tank3)
        X_val_tank1 = scaler_1.transform(X_val_tank1)
        X_val_tank2 = scaler_2.transform(X_val_tank2)
        X_val_tank3 = scaler_3.transform(X_val_tank3)
        X_test_tank1 = scaler_1.transform(X_test_tank1)
        X_test_tank2 = scaler_2.transform(X_test_tank2)
        X_test_tank3 = scaler_3.transform(X_test_tank3)
    else:
        X_train_iter_tank1 = np.asarray(inputs_tank1['train'][split])
        X_val_tank1 = np.asarray(inputs_tank1['val'][split])
        X_train_iter_tank2 = np.asarray(inputs_tank2['train'][split])
        X_val_tank2 = np.asarray(inputs_tank2['val'][split])
        X_train_iter_tank3 = np.asarray(inputs_tank3['train'][split])
        X_val_tank3 = np.asarray(inputs_tank3['val'][split])
        
        y_train_iter = np.asarray(labels['train'][split])
        y_val = np.asarray(labels['val'][split])

        # Normalize with mean = 0 and std = 1
        for i_c in range(input_channels):
            scaler_1 = preprocessing.StandardScaler()
            X_train_iter_tank1[:, i_c, :] = scaler_1.fit_transform(X_train_iter_tank1[:, i_c, :])
            scaler_2 = preprocessing.StandardScaler()
            X_train_iter_tank2[:, i_c, :] = scaler_2.fit_transform(X_train_iter_tank2[:, i_c, :])
            scaler_3 = preprocessing.StandardScaler()
            X_train_iter_tank3[:, i_c, :] = scaler_3.fit_transform(X_train_iter_tank3[:, i_c, :])
            X_val_tank1[:, i_c, :] = scaler_1.transform(X_val_tank1[:, i_c, :])
            X_val_tank2[:, i_c, :] = scaler_2.transform(X_val_tank2[:, i_c, :])
            X_val_tank3[:, i_c, :] = scaler_3.transform(X_val_tank3[:, i_c, :])
            X_test_tank1[:, i_c, :] = scaler_1.transform(X_test_tank1[:, i_c, :])
            X_test_tank2[:, i_c, :] = scaler_2.transform(X_test_tank2[:, i_c, :])
            X_test_tank3[:, i_c, :] = scaler_3.transform(X_test_tank3[:, i_c, :])


    if(exo):
        if(one_trace):
            X_train_iter_exo = np.asarray(inputs_exo['train'][split])
            X_val_exo = np.asarray(inputs_exo['val'][split])
            scaler = preprocessing.StandardScaler()
            X_train_iter_exo = scaler.fit_transform(X_train_iter_exo)
            X_val_exo = scaler.transform(X_val_exo)
            X_test_exo = scaler.transform(X_test_exo)
        else:
            X_train_iter_exo_tank1 = np.asarray(inputs_exo_tank1['train'][split])
            X_val_exo_tank1 = np.asarray(inputs_exo_tank1['val'][split])
            X_train_iter_exo_tank2 = np.asarray(inputs_exo_tank2['train'][split])
            X_val_exo_tank2 = np.asarray(inputs_exo_tank2['val'][split])
            X_train_iter_exo_tank3 = np.asarray(inputs_exo_tank3['train'][split])
            X_val_exo_tank3 = np.asarray(inputs_exo_tank3['val'][split])
            
            # Normalize with mean = 0 and std = 1
            scaler_1 = preprocessing.StandardScaler()
            X_train_iter_exo_tank1 = scaler_1.fit_transform(X_train_iter_exo_tank1)
            scaler_2 = preprocessing.StandardScaler()
            X_train_iter_exo_tank2 = scaler_2.fit_transform(X_train_iter_exo_tank2)
            scaler_2 = preprocessing.StandardScaler()
            X_train_iter_exo_tank3 = scaler_2.fit_transform(X_train_iter_exo_tank3)
            X_val_exo_tank1 = scaler_1.transform(X_val_exo_tank1)
            X_val_exo_tank2 = scaler_2.transform(X_val_exo_tank2)
            X_val_exo_tank3 = scaler_2.transform(X_val_exo_tank3)
            X_test_exo_tank1 = scaler_1.transform(X_test_exo_tank1)
            X_test_exo_tank2 = scaler_2.transform(X_test_exo_tank2)
            X_test_exo_tank3 = scaler_2.transform(X_test_exo_tank3)
            exos_train = [X_train_iter_exo_tank1, X_train_iter_exo_tank2, X_train_iter_exo_tank3]
            exos_val = [X_val_exo_tank1, X_val_exo_tank2, X_val_exo_tank3]
            exos_test = [X_test_exo_tank1, X_test_exo_tank2, X_test_exo_tank3]
    loss = nn.CrossEntropyLoss()
    loss = loss.cuda()
    if(exo):
        if(one_trace and not three_convs):
            dataset_train = DatasetExo(X_train_iter, X_train_iter_exo,
                                       y_train_iter)
            dataset_val = DatasetExo(X_val, X_val_exo, y_val)
        else:

            dataset_train = Dataset3TanksSeparatedExo(X_train_iter_tank1, X_train_iter_tank2, 
                                                      X_train_iter_tank3, exos_train, y_train_iter)
            dataset_val = Dataset3TanksSeparatedExo(X_val_tank1, X_val_tank2, 
                                                      X_val_tank3, exos_val, y_val)
    else:
        if(not three_convs):
            dataset_train = Dataset(X_train_iter, y_train_iter)
            dataset_val = Dataset(X_val, y_val)
        else:
            dataset_train = Dataset3TanksSeparated(X_train_iter_tank1, X_train_iter_tank2, X_train_iter_tank3, y_train_iter)
            dataset_val = Dataset3TanksSeparated(X_val_tank1, X_val_tank2, X_val_tank3, y_val)

    if(exo):
        if(not three_convs):
            dataset_test = DatasetExo(X_test, X_test_exo, y_test)
            test_generator = data.DataLoader(dataset_test, **params)
        else:
            dataset_test = Dataset3TanksSeparatedExo(X_test_tank1, X_test_tank2,   
                                                     X_test_tank3, exos_test, y_test)
            test_generator = data.DataLoader(dataset_test, **params)
    else:
        if(not three_convs):
            dataset_test = Dataset(X_test, y_test)
            test_generator = data.DataLoader(dataset_test, **params)
        else:
            dataset_test = Dataset3TanksSeparated(X_test_tank1, X_test_tank2, X_test_tank3, y_test)
            test_generator = data.DataLoader(dataset_test, **params)

    if(full_batch and not three_convs):
        # Parameters
        params_train = {'batch_size': X_train_iter.shape[0],
                  'shuffle': True,
                  'num_workers': 12}
        params_val = {'batch_size': X_val.shape[0],
                  'shuffle': True,
                  'num_workers': 12}
    elif(full_batch and three_convs):
        # Parameters
        params_train = {'batch_size': X_train_iter_tank1.shape[0],
                  'shuffle': True,
                  'num_workers': 12}
        params_val = {'batch_size': X_val_tank1.shape[0],
                  'shuffle': True,
                  'num_workers': 12}
    else:
        # Parameters
        if(one_trace):
            params_train = {'batch_size': batch_size,
                      'shuffle': True,
                      'num_workers': 12}
            params_val = {'batch_size': batch_size,
                      'shuffle': True,
                      'num_workers': 12}
        else:
            params_train = {'batch_size': batch_size,
                      'shuffle': True,
                      'num_workers': 12}
            params_val = {'batch_size': batch_size,
                      'shuffle': True,
                      'num_workers': 12}
    training_generator = data.DataLoader(dataset_train, **params_train)
    val_generator = data.DataLoader(dataset_val, **params_val)
    
    print("Training")
    start = time.time()
    if(three_convs):
        total_size = X_train_iter_tank1.shape[0]
        val_size = X_val_tank1.shape[0]
    else:
        total_size = X_train_iter.shape[0]
        val_size = X_val.shape[0]

    best_model, clf, results = train(model, training_generator, epochs,
                                total_size=total_size, loss=loss,
                                optimizer=optimizer,
                                val_generator=val_generator,
                                val_size=val_size,
                                conv=conv, one_trace=one_trace,
                                verbose=False,
                                elec=elec, output_size=output_size, exo=exo,
                                three_convs=three_convs,
                                input_channels=input_channels,
                                usesvm=svm_)
    end = time.time()
    print("Time elapsed {}".format(end - start))
    print("End Training")

    # getting results values
    cms = results['confusion_matrix']
    validation_accuracy = results["validation_accuracy"]
    validation_losses = results["validation_losses"]
    training_accuracy = results["training_accuracy"]
    training_losses = results["training_losses"]

    max_acc_val = np.where(validation_accuracy == np.max(validation_accuracy))
    cm = cms[max_acc_val[0][0]]
    # printing results
    print("Confusion Matrix: \n")
    print(cm.conf)
    print("\n")

    if(one_vs_all):
        tp = cm.conf[0][0]
        fn = cm.conf[0][1]
        fp = cm.conf[1][0]
        tn = cm.conf[1][1]
        sen = tp/(tp + fn)
        spe = tn/(tn+fp)
        val_sen.append(sen)
        val_spe.append(spe)

        print("Sensitivity: {}".format(sen))
        print("Specificity: {}".format(spe))
        print("\n")
    F1Score = np.zeros(output_size)
    for cls in range(output_size):
        try:
            F1Score[cls] = 2.*cm.conf[cls, cls]/(np.sum(cm.conf[cls, :])+np.sum(cm.conf[:, cls]))
        except:
            pass

    print("F1Score: ")
    for cls, score in enumerate(F1Score):
        print("{}: {:.2f}".format(cls, score))

    print("\n")
    print("Best accuracy in validation {} at epoch {}".format(
        np.max(validation_accuracy), np.argmax(validation_accuracy)+1))
    print("Accuracy in validation {} at epoch {}".format(
        validation_accuracy[-1], len(validation_accuracy)))
    print("\n")
    print("Best accuracy in training {} at epoch {}".format(
        np.max(training_accuracy), np.argmax(training_accuracy)+1))
    print("\n")
    print("Accuracy in validation {} at epoch {}".format(
        training_accuracy[-1], len(training_accuracy)))
    print("\n")
    print("Lower loss in validation {} at epoch {}".format(
        np.min(validation_losses), np.argmin(validation_losses)+1))

    print("\n")
    print("Lower loss in training {} at epoch {}".format(
        np.min(training_losses), np.argmin(training_losses)+1))

    if(save_fig):
        # plotting validation loss
        plt.figure()
        plt.title("Validation loss through epochs")
        plt.xlabel("Nº of epochs")
        plt.ylabel("Validation Loss")
        plt.plot(list(range(0, len(validation_losses))), validation_losses)
        plt.savefig(path_images+name+"/"+"validation_loss_"+str(split)+"-"+name+".png")

        # plotting training loss
        plt.figure()
        plt.title("Training loss through epochs")
        plt.xlabel("Nº of epochs")
        plt.ylabel("Training Loss")
        plt.plot(list(range(0, len(training_losses))), training_losses)
        plt.savefig(path_images+name+"/"+"training_loss_"+str(split)+"-"+name+".png")

        # plotting validation accuracy
        plt.figure()
        plt.title("Validation Accuracy through epochs")
        plt.xlabel("Nº of epochs")
        plt.ylabel("Accuracy")
        plt.plot(list(range(0, len(validation_accuracy))), validation_accuracy)
        plt.savefig(path_images+name+"/"+"validation_accuracy_"+str(split)+"-"+name+".png")
        
    # Obtaining results for test
    start = time.time()
    if(three_convs):
        test_size = X_test_tank1.shape[0]
    else:
        test_size = X_test.shape[0]

    best_model = best_model.eval()
    test_loss, test_accuracy, cm = test(best_model, test_generator,
                                        test_size, loss,
                                        confusion_matrix=None,
                                        conv=conv, one_trace=one_trace,
                                        verbose=False,
                                        elec=elec, output_size=output_size,
                                        exo=exo, three_convs=three_convs,
                                        input_channels=input_channels,
                                        clf=clf)
    
    # printing results
    print("Results for Test set: \n")
    print("Confusion Matrix: \n")
    print(cm.conf)
    print("\n")
    
    if(one_vs_all):
        tp = cm.conf[0][0]
        fn = cm.conf[0][1]
        fp = cm.conf[1][0]
        tn = cm.conf[1][1]
        cm_plot = np.asarray([[tp,fn],[fp,tn]])
        sen = tp/(tp + fn)
        spe = tn/(tn+fp)
        test_sen.append(sen)
        test_spe.append(spe)
        if(save_fig):
            plot_name = path_images+name+"/"+"cm"+str(split)+"-"+name+".png"
            plot_conf_matrix(cm_plot, classes=['Photon', 'Hadron'],
                              normalize=False,
                              title=name,
                              cmap=plt.cm.Blues,
                              path=plot_name)
            plot_name = path_images+name+"/"+"cm"+str(split)+"-"+name+"-normalized.png"
            plot_conf_matrix(cm_plot, classes=['Photon', 'Hadron'],
                              normalize=True,
                              title=name,
                              cmap=plt.cm.Blues,
                              path=plot_name)
        print("Sensitivity: {}".format(sen))
        print("Specificity: {}".format(spe))
        print("\n")
    
    F1Score = np.zeros(output_size)
    for cls in range(output_size):
        try:
            F1Score[cls] = 2.*cm.conf[cls, cls]/(np.sum(cm.conf[cls, :])+np.sum(cm.conf[:, cls]))
        except:
            pass

    print("F1Score: ")
    for cls, score in enumerate(F1Score):
        print("{}: {:.2f}".format(cls, score))

    print("Loss in test set: {}".format(test_loss))
    print("Accuracy in test set: {}".format(test_accuracy))


    test_accs.append(test_accuracy)
    val_accs.append(validation_accuracy[-1])

    # saving model
    if(save_weights):
        torch.save(best_model.state_dict(), path_models+name+"/"+name+"-"+str(split))

print("Mean results \n")
print("Mean accuracy in val set: {} {}".format(np.mean(val_accs),
                                               np.std(val_accs)))
print("Mean accuracy in test set: {} {}".format(np.mean(test_accs),
                                                np.std(test_accs)))
if(one_vs_all):
    print("Mean sensitivity in val set: {} {}".format(np.mean(val_sen),
                                                      np.std(val_sen)))
    print("Mean sensitivity in test set: {} {}".format(np.mean(test_sen),
                                                       np.std(test_sen)))
    print("Mean specicificity in val set: {} {}".format(np.mean(val_spe),
                                                        np.std(val_spe)))
    print("Mean specicificity in test set: {} {}".format(np.mean(test_spe),
                                                         np.std(test_spe)))

