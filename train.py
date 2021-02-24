import torch
from torch.autograd.variable import Variable
import numpy as np
from tqdm import tqdm
from torchnet import meter
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set seed for weight initialization
torch.manual_seed(1234)

params_search = [2**-11, 2**-8, 2**-5, 2**-2, 2, 2**4, 2**7, 2**10]
parameters = {'C':params_search, 'gamma':params_search}
def train(model, training_generator, n_epochs, total_size, loss,
          optimizer, val_generator, val_size, conv, one_trace, verbose,
          elec, output_size, exo, three_convs, input_channels, usesvm):
    # Loss and optimizer
    best_model = model
    i = 0
    validation_losses = []
    training_losses = []
    validation_accuracy = []
    training_accuracy = []
    cms = []
    clf = []
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch, n_epochs))
        batch_loss = []
        correct = 0
        total = 0
        stop = 0
        for values in tqdm(training_generator):
            if(conv and one_trace and not three_convs):
                if(elec):
                    inp = Variable(values[0]).view(values[0].size(0), 1, values[0].size(1), values[0].size(2))
                    # inp = Variable(values[0]).view(values[0].size(0), 1, values[0].size(1))
                    inp = inp.cuda()
                else:
                    inp = Variable(values[0]).view(values[0].size(0), 1, values[0].size(1))
            elif(conv and not three_convs):
                inp = Variable(values[0]).view(values[0].size(0), values[0].size(1), values[0].size(2))
                inp = inp.cuda()
            elif(conv and three_convs):
                inp_tank1 = Variable(values[0]).view(values[0].size(0), 1, values[0].size(1), values[0].size(2))
                inp_tank2 = Variable(values[1]).view(values[1].size(0), 1, values[0].size(1), values[1].size(2))
                inp_tank3 = Variable(values[2]).view(values[2].size(0), 1, values[0].size(1), values[2].size(2))
                inp_tank1 = inp_tank1.cuda()
                inp_tank2 = inp_tank2.cuda()
                inp_tank3 = inp_tank3.cuda()
            else:
                inp = Variable(values[0])

            if(exo):
                if(not three_convs):
                    inp_exo = Variable(values[1]).view(values[1].size(0), values[1].size(1))
                    output = Variable(values[2].squeeze()).type(torch.LongTensor)
                    inp_exo = inp_exo.cuda()
                    output = output.cuda()
                else:
                    inp_exo = []
                    for value in values[3]:
                        aux = Variable(value).view(value.size(0), value.size(1))
                        aux = aux.cuda()
                        inp_exo.append(aux)
                    output = Variable(values[4].squeeze()).type(torch.LongTensor)
                    output = output.cuda()
            else:
                if(not three_convs):
                    output = Variable(values[1].squeeze()).type(torch.LongTensor)
                    output = output.cuda()
                else:
                    output = Variable(values[3].squeeze()).type(torch.LongTensor)
                    output = output.cuda()
            
            # Forward pass
            if(exo):
                if(not three_convs):
                    predicted = model(inp, inp_exo)
                else:
                    predicted = model(inp_tank1, inp_tank2, inp_tank3, inp_exo, usesvm)
            elif(not three_convs):
                if(usesvm):
                    predicted = model(inp)
                else:
                    predicted = model(inp, usesvm)
            else:
                predicted = model(inp_tank1, inp_tank2, inp_tank3, svm_=usesvm)

            if(usesvm):
                predicted_cpu = predicted.cpu().detach().numpy()
                output_cpu = output.cpu().detach().numpy()

                if(np.unique(output_cpu).shape[0] > 1):
                    svm_ = svm.SVC(kernel='rbf', gamma='scale')
                    if(not one_trace):
                        clf = GridSearchCV(svm_, parameters, cv=5)
                    else:
                        clf = svm_
                    clf.fit(predicted_cpu, output_cpu)
                    predicted = clf.predict(predicted_cpu)
                    # softmax to get the probability of the classes
                    # predicted = F.softmax(predicted, dim=1)
                    # _, labels = torch.max(predicted.data, 1)

                    try:
                        total += output.size(0)
                    except:
                        total += 1
                    correct += (predicted == output_cpu).sum().item()
                    # correct += (labels == output).sum().item()
                    try:
                        computed_loss = np.abs(predicted - output_cpu)
                        computed_loss = np.asarray(np.sum(computed_loss)/computed_loss.shape[0])
                        computed_loss = torch.tensor(computed_loss, requires_grad=True)
                        # computed_loss = loss(predicted, output)
                    except:
                        computed_loss = loss(predicted, output.unsqueeze(0))
                    batch_loss.append(computed_loss.item())
            else:
                # softmax to get the probability of the classes
                # predicted = F.softmax(predicted, dim=1)
                _, labels = torch.max(predicted.data, 1)
                try:
                    total += output.size(0)
                except:
                    total += 1
                correct += (labels == output).sum().item()
                try:
                    computed_loss = loss(predicted, output)
                except:
                    computed_loss = loss(predicted, output.unsqueeze(0))

                batch_loss.append(computed_loss.item())
                # Backward and optimize
                optimizer.zero_grad()
                computed_loss.backward()
                optimizer.step()
                stop += 1
            
        training_losses.append(np.mean(batch_loss))
        
        model = model.eval()
        i += 1
        confusion_matrix = meter.ConfusionMeter(output_size)  # I have 5 classes here
        val_loss, accuracy, confusion_matrix = validation(model, val_generator,
                                                          val_size, loss,
                                                          confusion_matrix=confusion_matrix,
                                                          conv=conv,
                                                          one_trace=one_trace,
                                                          verbose=verbose,
                                                          elec=elec,
                                                          exo=exo,
                                                          three_convs=three_convs,
                                                          input_channels=input_channels,
                                                          clf=clf)
        model = model.train()
        validation_losses.append(val_loss)
        print('Training: Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
              .format(epoch+1, n_epochs,
                      computed_loss.item(), (100*correct)/total))
        print('Val: Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
              .format(epoch+1, n_epochs,
                      val_loss, accuracy))

        validation_accuracy.append(accuracy)
        cms.append(confusion_matrix)
        training_accuracy.append((100*correct)/total)
        
    results = {
        "validation_losses": validation_losses,
        "training_losses": training_losses,
        "validation_accuracy": validation_accuracy,
        "training_accuracy": training_accuracy,
        "confusion_matrix": cms
        }
    return model.eval(), clf , results


def validation(model, test_generator, total_size, loss, input_channels, clf=[],
               confusion_matrix=None,
               conv=True, one_trace=False, verbose=False, elec=False, 
               exo=False, three_convs=False):
    total_loss = []
    correct = 0
    total = 0
    cm = confusion_matrix
    with torch.no_grad():
        for values in tqdm(test_generator):
            if(conv and one_trace and not three_convs):
                if(elec):
                    
                    inp = Variable(values[0]).view(values[0].size(0), 1, values[0].size(1), values[0].size(2))
                    
                    # inp = Variable(values[0]).view(values[0].size(0), 1, values[0].size(1))
                    inp = inp.cuda()
                else:
                    inp = Variable(values[0]).view(values[0].size(0), 1, values[0].size(1))
            elif(conv and not three_convs):
                inp = Variable(values[0]).view(values[0].size(0), values[0].size(1), values[0].size(2))
                inp = inp.cuda()
            elif(conv and three_convs):
                inp_tank1 = Variable(values[0]).view(values[0].size(0), 1, values[0].size(1), values[0].size(2))
                inp_tank2 = Variable(values[1]).view(values[1].size(0), 1, values[0].size(1), values[1].size(2))
                inp_tank3 = Variable(values[2]).view(values[2].size(0), 1, values[0].size(1), values[2].size(2))
                inp_tank1 = inp_tank1.cuda()
                inp_tank2 = inp_tank2.cuda()
                inp_tank3 = inp_tank3.cuda()
            else:
                inp = Variable(values[0])
            if(exo):
                if(one_trace):
                    inp_exo = Variable(values[1]).view(values[1].size(0), values[1].size(1))
                    output = Variable(values[2].squeeze()).type(torch.LongTensor)
                    inp_exo = inp_exo.cuda()
                    output = output.cuda()
                else:
                    if(not three_convs):
                        inp_exo = Variable(values[1]).view(values[1].size(0), values[1].size(1))
                        output = Variable(values[2].squeeze()).type(torch.LongTensor)
                        inp_exo = inp_exo.cuda()
                        output = output.cuda()
                    else:
                        inp_exo = []
                        for value in values[3]:
                            aux = Variable(value).view(value.size(0), value.size(1))
                            aux = aux.cuda()
                            inp_exo.append(aux)
                        output = Variable(values[4].squeeze()).type(torch.LongTensor)
                        output = output.cuda()
            else:
                if(not three_convs):
                    output = Variable(values[1].squeeze()).type(torch.LongTensor)
                    output = output.cuda()
                else:
                    output = Variable(values[3].squeeze()).type(torch.LongTensor)
                    output = output.cuda()
            if(clf != []):
                usesvm = True
            else:
                usesvm = False
            if(exo):
                if(not three_convs):
                    predicted = model(inp, inp_exo)
                else:
                    predicted = model(inp_tank1, inp_tank2, inp_tank3, inp_exo, usesvm)
            elif(not three_convs):
                if(usesvm):
                    predicted = model(inp)
                else:
                    predicted = model(inp, usesvm)
            else:
                predicted = model(inp_tank1, inp_tank2, inp_tank3, svm_=usesvm)

            if(clf != []):
                predicted_cpu = predicted.cpu().detach().numpy()
                output_cpu = output.cpu().detach().numpy()
                predicted = clf.predict(predicted_cpu)

                if(cm is not None):
                    try:
                        cm.add(torch.tensor(predicted).data.squeeze(),
                               output.type(torch.LongTensor))
                    except:
                        cm.add(predicted.data,
                               output.unsqueeze(0).type(torch.LongTensor))

                try:
                    # total_loss.append(loss(predicted, output))
                    computed_loss = np.abs(predicted - output_cpu)
                    computed_loss = np.asarray(np.sum(computed_loss)/computed_loss.shape[0])
                    total_loss.append(computed_loss)
                except:
                    total_loss.append(loss(predicted, output.unsqueeze(0)))
                try:
                    total += output.size(0)
                except:
                    total += 1
                # correct += (labels == output).sum().item()
                correct += (predicted == output_cpu).sum().item()
            else:
                _, labels = torch.max(predicted.data, 1)
                if(cm is not None):
                    try:
                        cm.add(predicted.data.squeeze(),
                               output.type(torch.LongTensor))
                    except:
                        cm.add(predicted.data,
                               output.unsqueeze(0).type(torch.LongTensor))
                try:
                    total_loss.append(loss(predicted, output))
                except:
                    total_loss.append(loss(predicted, output.unsqueeze(0)))
                try:
                    total += output.size(0)
                except:
                    total += 1
                correct += (labels == output).sum().item()
    if(verbose):
        print('Mean and standard deviation in dataset with size {} are: {} +- {}'.format(
            total_size, np.mean(total_loss), np.std(total_loss)))
    try:
        return np.mean(total_loss), (100*correct)/total, cm
    except:
        return torch.mean(torch.stack(total_loss), dim=0), (100*correct)/total, cm


def test(model, test_generator, total_size, loss, input_channels, confusion_matrix=None,
         conv=True, one_trace=False, verbose=False, elec=False, output_size=2,
         exo=False, three_convs=False, clf=[]):
    '''
    Function to compute results in test set given a model.
    @model: model to use
    '''
    total_loss = []
    correct = 0
    total = 0
    cm = meter.ConfusionMeter(output_size)
    with torch.no_grad():
        for values in tqdm(test_generator):
            if(conv and one_trace and not three_convs):
                if(elec):
                    
                    inp = Variable(values[0]).view(values[0].size(0), 1, values[0].size(1), values[0].size(2))
                    
                    # inp = Variable(values[0]).view(values[0].size(0), 1, values[0].size(1))
                    inp = inp.cuda()
                else:
                    inp = Variable(values[0]).view(values[0].size(0), 1, values[0].size(1))
            elif(conv and not three_convs):
                inp = Variable(values[0]).view(values[0].size(0), values[0].size(1), values[0].size(2))
                inp = inp.cuda()
            elif(conv and three_convs):
                inp_tank1 = Variable(values[0]).view(values[0].size(0), 1, values[0].size(1), values[0].size(2))
                inp_tank2 = Variable(values[1]).view(values[1].size(0), 1, values[0].size(1), values[1].size(2))
                inp_tank3 = Variable(values[2]).view(values[2].size(0), 1, values[0].size(1), values[2].size(2))
                inp_tank1 = inp_tank1.cuda()
                inp_tank2 = inp_tank2.cuda()
                inp_tank3 = inp_tank3.cuda()
            else:
                inp = Variable(values[0])
            if(exo):
                if(one_trace):
                    inp_exo = Variable(values[1]).view(values[1].size(0), values[1].size(1))
                    output = Variable(values[2].squeeze()).type(torch.LongTensor)
                    inp_exo = inp_exo.cuda()
                    output = output.cuda()
                else:
                    if(not three_convs):
                        inp_exo = Variable(values[1]).view(values[1].size(0), values[1].size(1))
                        output = Variable(values[2].squeeze()).type(torch.LongTensor)
                        inp_exo = inp_exo.cuda()
                        output = output.cuda()
                    else:
                        inp_exo = []
                        for value in values[3]:
                            aux = Variable(value).view(value.size(0), value.size(1))
                            aux = aux.cuda()
                            inp_exo.append(aux)
                        output = Variable(values[4].squeeze()).type(torch.LongTensor)
                        output = output.cuda()
            else:
                if(not three_convs):
                    output = Variable(values[1].squeeze()).type(torch.LongTensor)
                    output = output.cuda()
                else:
                    output = Variable(values[3].squeeze()).type(torch.LongTensor)
                    output = output.cuda()

            if(clf != []):
                usesvm = True
            else:
                usesvm = False

            if(exo):
                if(not three_convs):
                    predicted = model(inp, inp_exo)
                else:
                    predicted = model(inp_tank1, inp_tank2, inp_tank3, inp_exo, usesvm)
            elif(not three_convs):
                if(usesvm):
                    predicted = model(inp)
                else:
                    predicted = model(inp, usesvm)
            else:
                predicted = model(inp_tank1, inp_tank2, inp_tank3, svm_=usesvm)

            if(clf != []):
                predicted_cpu = predicted.cpu().detach().numpy()
                output_cpu = output.cpu().detach().numpy()
                predicted = clf.predict(predicted_cpu)
                if(cm is not None):
                    try:
                        cm.add(torch.tensor(predicted).data.squeeze(),
                               output.type(torch.LongTensor))
                    except:
                        cm.add(predicted.data,
                               output.unsqueeze(0).type(torch.LongTensor))
                # softmax to get the probability of the classes
                # predicted = F.softmax(predicted, dim=1)
                try:
                    # total_loss.append(loss(predicted, output))
                    computed_loss = np.abs(predicted - output_cpu)
                    computed_loss = np.asarray(np.sum(computed_loss)/computed_loss.shape[0])
                    total_loss.append(computed_loss)
                except:
                    total_loss.append(loss(predicted, output.unsqueeze(0)))
                try:
                    total += output.size(0)
                except:
                    total += 1
                # correct += (labels == output).sum().item()
                correct += (predicted == output_cpu).sum().item()
            else:
                _, labels = torch.max(predicted.data, 1)
                if(cm is not None):
                    try:
                        cm.add(predicted.data.squeeze(),
                               output.type(torch.LongTensor))
                    except:
                        cm.add(predicted.data,
                               output.unsqueeze(0).type(torch.LongTensor))
                try:
                    total_loss.append(loss(predicted, output))
                except:
                    total_loss.append(loss(predicted, output.unsqueeze(0)))
                try:
                    total += output.size(0)
                except:
                    total += 1
                correct += (labels == output).sum().item()
    if(verbose):
        print('Mean and standard deviation in dataset with size {} are: {} +- {}'.format(
            total_size, np.mean(total_loss), np.std(total_loss)))
    try:
        return np.mean(total_loss), (100*correct)/total, cm
    except:
        return torch.mean(torch.stack(total_loss), dim=0), (100*correct)/total, cm

