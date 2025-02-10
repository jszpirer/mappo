import torch.nn as nn
from torch import cat
from .util import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNLayer(nn.Module):
    def __init__(self, obs_shape, output_size, use_orthogonal, use_ReLU, kernel_size=2, stride=1):
        super(CNNLayer, self).__init__()


        #TODO: Maybe change the active_func
        active_func = nn.ReLU()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain('relu')

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_width = obs_shape[0]
        input_height = obs_shape[1]

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=1,
                            out_channels=1,
                            kernel_size=kernel_size,
                            stride=stride)
                  ),
            active_func,
            Flatten(),
            init_(nn.Linear((input_width - kernel_size + stride) * (input_height - kernel_size + stride),
                            6)
                  ),
            active_func
        )

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1],x.shape[2])
        x = self.cnn(x)
        return x

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size)) for i in range(self._layer_N)])

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x

class MergedModel(nn.Module):
    def __init__(self, mlp_args, obs_shape):
       super(MergedModel, self).__init__()
       #self.cnn = CNNLayer((16, 16), 10, mlp_args.use_orthogonal, mlp_args.use_ReLU)
       self.cnn = CNNLayer((32, 32), 10, mlp_args.use_orthogonal, mlp_args.use_ReLU)
       # TODO: Chnage this 6 when more agents
       flattened_size = 6

       #input_size = flattened_size*2 + 4
       # Without position
       input_size = flattened_size*2 + 2
       
       #TODOD: find a way to have access to 34 and 3 through args
       #if obs_shape[0]/34 > 1:
       #if obs_shape[0]/66 > 1:
       if obs_shape[0]/65 > 1:
           input_size *= 3
       # TODO: find a way to make this math automatic cause not correct for the critic
       self.mlp = MLPLayer(input_size, mlp_args.hidden_size, mlp_args.layer_N, mlp_args.use_orthogonal, mlp_args.use_ReLU)

    def forward(self, x):
        # Séparer le tenseur en trois parties autant de fois que nécessaire
        x_inter_list = []
        #TODO: find a way to have access to 34
        #for i in range(x.size()[1]//34):
        #for i in range(x.size()[1]//66):
        for i in range(x.size()[1]//65):
            #tensor1 = x[:, i*34:2+i*34, :]
            #tensor1 = x[:, i*66:2+i*66, :]

            # Without the position
            tensor1 = x[:, i*65:1+i*65, :]

            result = tensor1[:, :, :2]
            #additional_data = result.reshape(result.shape[0], 4)

            # Without the position
            additional_data = result.reshape(result.shape[0], 2)
            
            #tensor2 = x[:, 2+i*34:18+i*34, :]
            #tensor3 = x[:, 18+i*34:34+i*34, :]
            
            #tensor2 = x[:, 2+i*66:34+i*66, :]
            #tensor3 = x[:, 34+i*66:66+i*66, :]

            # Without the position
            tensor2 = x[:, 1+i*65:33+i*65, :]
            tensor3 = x[:, 33+i*65:65+i*65, :]

            x1 = self.cnn(tensor2)
            x2 = self.cnn(tensor3)
            x_inter = cat((additional_data, x1, x2), dim=1)
            x_inter_list.append(x_inter)
        # Concatenate the output of the CNN with position and velocity
        x = cat(x_inter_list, dim=1)
        # Give x to the MLP
        x = self.mlp(x)
        return x
