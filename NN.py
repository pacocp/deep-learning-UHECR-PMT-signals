import torch
import torch.nn as nn

# set seed for weight initialization
torch.manual_seed(1234)

# Init weights with cavier initilialization
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# seed for weight initialization
torch.manual_seed(1234)


# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, units, output_size):
        super(NeuralNet, self).__init__()
        self.input = nn.Linear(input_size, units[0])
        self.relu = nn.ReLU()
        self.fcs = []
        if len(units) > 1:
            for i in range(0, len(units) - 1):
                self.fcs.append(nn.Linear(units[i], units[i + 1]))
        self.output = nn.Linear(units[-1], output_size)

    def forward(self, x):
        out = self.input(x)
        out = self.relu(out)
        if(len(self.fcs) >= 1):
            for f in self.fcs:
                out = f(out)
                out = self.relu(out)
        out = self.output(out)
        return out


# 1D convolutional neural network
class ConvNeuralNet(nn.Module):
    def __init__(self, input_size, units, output_size):
        super(ConvNeuralNet, self).__init__()
        self.input = nn.Sequential(
            nn.Conv1d(1, units[0], kernel_size=10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1)
        )
        self.hiddens = []
        if(len(units) > 1):
            for i in range(0, len(units)-1):
                hidden = nn.Sequential(
                            nn.Conv1d(units[i], units[i+1], kernel_size=10),
                            nn.ReLU(),
                            nn.MaxPool1d(kernel_size=1, stride=1)
                            )
                self.hiddens.append(hidden)

        self.units = units
        
    def forward(self, x):
        out = self.input(x)
        if(len(self.hiddens) >= 1):
            i = 1
            for h in self.hiddens:
                out = h(out)
                i += 1
        out_ch = out.size(2)
        out = out.view(out.size(0), -1)
        output = nn.Sequential(
            nn.Linear(self.units[-1]*out_ch, 5)
        )
        out = output(out)
        return out


# 1D convolutional neural network
class ConvNeuralNet2Dense(nn.Module):
    def __init__(self, input_channels, units, output_size, input_size):
        super(ConvNeuralNet2Dense, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(input_channels, units[0], kernel_size=(3,8), stride=2),
            nn.BatchNorm2d(units[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Xavier weight initialization
        self.input.apply(init_weights)

        self.h1 = nn.Sequential(
            nn.Conv2d(units[0], units[1], kernel_size=(1,8), 
                      stride=2),
            nn.BatchNorm2d(units[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=2))
        # Xavier weight initialization
        self.h1.apply(init_weights)

        self.h2 = nn.Sequential(
            nn.Conv2d(units[1], units[1], kernel_size=(1,8), 
                      stride=2),
            nn.BatchNorm2d(units[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=2))
        # Xavier weight initialization
        self.h2.apply(init_weights)

        # computing the size for the fc layer
        self.fc_input_size = self.calculate_size(input_size)

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_input_size, units[2]),
            nn.ReLU()
        )
        
        # Xavier weight initialization
        self.fc1.apply(init_weights)

        self.fc2 = nn.Sequential(
            nn.Linear(units[2], units[2]),
            nn.ReLU()
        )
        
        # Xavier weight initialization
        self.fc2.apply(init_weights)

        self.fc3 = nn.Sequential(
            nn.Linear(units[2], units[2]),
            nn.ReLU()
        )
        
        # Xavier weight initialization
        self.fc3.apply(init_weights)

        self.output = nn.Sequential(
            nn.Linear(units[2], output_size)
        )
        # Xavier weight initialization
        self.output.apply(init_weights)

        self.units = units

    def forward(self, x, svm_):
        out = self.input(x)
        out = self.h1(out)
        out = self.h2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        if(not svm_):
            out = self.output(out)
        return out

    def calculate_size(self, input_size):
        x = torch.randn(input_size)
        output = self.input(x)
        output = self.h1(output)
        output = self.h2(output)
        output = output.view(output.size(0), -1)
        return output.size(1)

    def load_my_state_dict(self, state_dict):
        '''Load weights from conv layers'''
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


# 1D convolutional neural network with Exogenous variables
class ConvNeuralNet2DenseExo(nn.Module):
    def __init__(self, input_channels, units, 
                 output_size, input_size, exo_size):
        super(ConvNeuralNet2DenseExo, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(input_channels, units[0], kernel_size=(3,8), stride=2),
            nn.BatchNorm2d(units[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Xavier weight initialization
        self.input.apply(init_weights)

        self.h1 = nn.Sequential(
            nn.Conv2d(units[0], units[1], kernel_size=(1,8), 
                      stride=2),
            nn.BatchNorm2d(units[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=2))
        # Xavier weight initialization
        self.h1.apply(init_weights)

        self.h2 = nn.Sequential(
            nn.Conv2d(units[1], units[1], kernel_size=(1,8), 
                      stride=2),
            nn.BatchNorm2d(units[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=2))
        # Xavier weight initialization
        self.h2.apply(init_weights)

        # computing the size for the fc layer
        self.fc_input_size = self.calculate_size(input_size, exo_size)

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_input_size, units[2]),
            nn.ReLU()
        )
        
        # Xavier weight initialization
        self.fc1.apply(init_weights)

        self.fc2 = nn.Sequential(
            nn.Linear(units[2], units[2]),
            nn.ReLU()
        )
        
        # Xavier weight initialization
        self.fc2.apply(init_weights)

        self.fc3 = nn.Sequential(
            nn.Linear(units[2], units[2]),
            nn.ReLU()
        )
        
        # Xavier weight initialization
        self.fc3.apply(init_weights)

        self.output_exo = nn.Sequential(
            nn.Linear(units[2], output_size)
        )
        # Xavier weight initialization
        self.output_exo.apply(init_weights)

        self.units = units

    def forward(self, x, exo):
        out_cnn = self.input(x)
        out_cnn = self.h1(out_cnn)
        out_cnn = self.h2(out_cnn)
        out_cnn = out_cnn.view(out_cnn.size(0), -1)
        out_exo = exo.view(exo.size(0), -1)
        out = torch.cat((out_cnn, out_exo), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.output_exo(out)
        return out

    def load_my_state_dict(self, state_dict):
        '''Load weights from conv layers'''
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def calculate_size(self, input_size, exo_size):
        x = torch.randn(input_size)
        x_exo = torch.randn(exo_size)
        output = self.input(x)
        output = self.h1(output)
        output = self.h2(output)
        output = output.view(output.size(0), -1)
        out_exo = x_exo.view(x_exo.size(0), -1)
        out = torch.cat((output, out_exo), dim=1)
        return out.size(1)

    def freeze_layers(self):
        for param in self.input.parameters():
            param.requires_grad = False
        for param in self.h1.parameters():
            param.requires_grad = False
        for param in self.h2.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = False
        for param in self.fc3.parameters():
            param.requires_grad = False

# Conv block in order to restore weights pretrained for 3 tanks
class ConvBlock(nn.Module):
    def __init__(self, input_channels, units, 
                 output_size, input_size, exo_size=[]):
        super(ConvBlock, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(input_channels, units[0], kernel_size=(3,8), stride=2),
            nn.BatchNorm2d(units[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Xavier weight initialization
        self.input.apply(init_weights)

        self.h1 = nn.Sequential(
            nn.Conv2d(units[0], units[1], kernel_size=(1,8), 
                      stride=2),
            nn.BatchNorm2d(units[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=2))
        # Xavier weight initialization
        self.h1.apply(init_weights)

        self.h2 = nn.Sequential(
            nn.Conv2d(units[1], units[1], kernel_size=(1,8), 
                      stride=2),
            nn.BatchNorm2d(units[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=2))
        # Xavier weight initialization
        self.h2.apply(init_weights)

        # computing the size for the fc layer
        self.fc_input_size = self.calculate_size(input_size, exo_size)

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_input_size, units[2]),
            nn.ReLU()
        )
        
        # Xavier weight initialization
        self.fc1.apply(init_weights)

        self.fc2 = nn.Sequential(
            nn.Linear(units[2], units[2]),
            nn.ReLU()
        )
        
        # Xavier weight initialization
        self.fc2.apply(init_weights)

        self.fc3 = nn.Sequential(
            nn.Linear(units[2], units[2]),
            nn.ReLU()
        )
        
        # Xavier weight initialization
        self.fc3.apply(init_weights)

        self.fc4 = nn.Sequential(
            nn.Linear(units[2], units[2]),
            nn.ReLU()
        )
        
        # Xavier weight initialization
        self.fc4.apply(init_weights)

    def forward(self, x, x_exo=[], exo=False):
        out_cnn = self.input(x)
        out_cnn = self.h1(out_cnn)
        out_cnn = self.h2(out_cnn)
        out_cnn = out_cnn.view(out_cnn.size(0), -1)
        if(exo):
            out_exo = x_exo.view(x_exo.size(0), -1)
            out = torch.cat((out_cnn, out_exo), dim=1)
        else:
            out = out_cnn
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def load_my_state_dict(self, state_dict):
        '''Load weights from conv layers'''
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def calculate_size(self, input_size, exo_size):
        x = torch.randn(input_size)
        output = self.input(x)
        output = self.h1(output)
        output = self.h2(output)
        output = output.view(output.size(0), -1)
        if(exo_size != []):
            x_exo = torch.randn(exo_size)
            out_exo = x_exo.view(x_exo.size(0), -1)
            out = torch.cat((output, out_exo), dim=1)
        else:
            out = output
        return out.size(1)

    def freeze_layers(self, svm_):
        for param in self.input.parameters():
            param.requires_grad = False
        '''
        for param in self.h1.parameters():
            param.requires_grad = False
        for param in self.h2.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = False
        for param in self.fc3.parameters():
            param.requires_grad = False
        '''
        if(svm_):
            for param in self.h1.parameters():
                param.requires_grad = False
            for param in self.h2.parameters():
                param.requires_grad = False
            for param in self.fc1.parameters():
                param.requires_grad = False

# 1D convolutional neural network using the information
# of three tanks
class ConvNeuralNet2Dense3Tanks(nn.Module):
    def __init__(self, input_channels, units, output_size,
                 input_size, conv_blocks, exo_size=[]):
        super(ConvNeuralNet2Dense3Tanks, self).__init__()
        
        self.tank1 = conv_blocks[0]
                       
        self.tank2 = conv_blocks[1]
        
        self.tank3 = conv_blocks[2]
        
        self.output_input = self.calculate_size(input_size, exo_size)

        self.output = nn.Sequential(
            nn.Linear(self.output_input, output_size)
        )

        self.units = units

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, tank1, tank2, tank3, exos=[], svm_=False):
        # Tank 1 pass
        if(exos != []):
            out_tank1 = self.tank1(tank1, exos[0], exo=True)
            out_tank2 = self.tank2(tank2, exos[1], exo=True)
            out_tank3 = self.tank3(tank3, exos[2], exo=True)
        else:
            out_tank1 = self.tank1(tank1, exo=False)
            out_tank2 = self.tank2(tank2, exo=False)
            out_tank3 = self.tank3(tank3, exo=False)
        # out_tank1 = out_tank1.view(out_tank1.size(0), -1)
        '''
        if(exos != []):
            out_exo_1 = exos[0].view(exos[0].size(0), -1)
            out_tank1 = torch.cat((out_tank1, out_exo_1), dim=1)
        '''
        # out_tank1 = self.fc1_tank1(out_tank1)
        # Tank 2 pass
        
        # out_tank2 = out_tank2.view(out_tank2.size(0), -1)
        '''
        if(exos != []):
            out_exo_2 = exos[1].view(exos[1].size(0), -1)
            out_tank2 = torch.cat((out_tank2, out_exo_2), dim=1)
        '''
        # out_tank2 = self.fc1_tank2(out_tank2)
        # Tank 3 pass
        
        # out_tank3 = out_tank3.view(out_tank3.size(0), -1)
        '''
        if(exos != []):
            out_exo_3 = exos[2].view(exos[2].size(0), -1)
            out_tank3 = torch.cat((out_tank3, out_exo_3), dim=1)
        '''
        # out_tank3 = self.fc1_tank3(out_tank3)

        out = torch.cat((out_tank1, out_tank2, out_tank3), dim=1)
        # out = out_tank1 + out_tank2 + out_tank3
        # out = self.fc2(out)
        # out = self.fc3(out)
        if(not svm_):
            out = self.output(out)
        return out

    def calculate_size(self, input_size, exo_size=[]):
        x = torch.randn(input_size)
        if(exo_size != []):
            x_exo = torch.randn(exo_size)
            output_conv = self.tank1(x, x_exo, exo=True)
        else:
            output_conv = self.tank1(x, exo=False)
        out = torch.cat((output_conv, output_conv, output_conv), dim=1)
        return out.size(1)

    def load_my_state_dict(self, state_dict):
        '''Load weights from conv layers'''
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

