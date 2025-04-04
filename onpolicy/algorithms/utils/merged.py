import torch.nn as nn
from torch import cat, chunk, div, add, abs, max
from .util import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SimpleCNN(nn.Module):
    def __init__(self, obs_shape, output_size, use_orthogonal, use_ReLU, kernel_size=2, stride=1, input_channels=1, output_channels=1):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=kernel_size, stride=stride, groups=input_channels)
        self.tanh = nn.Tanh()
        #print(use_ReLU)
        #self.final_activ = nn.ReLU() if use_ReLU else nn.Tanh()
        size = (obs_shape[1] - (kernel_size - 1) - 1) // stride + 1
        self.fc_red = nn.Linear(in_features=size * size, out_features=output_size//3)
        self.fc_green = nn.Linear(in_features=size * size, out_features=output_size//3)
        self.fc_blue = nn.Linear(in_features=size * size, out_features=output_size//3)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        # flatten
        x = x.view(x.size(0), -1)
        # chunk the 1D vector to separate the colors
        red, green, blue = chunk(x, 3, dim=1)
        # dense layers
        red_out = self.fc_red(red)
        green_out = self.fc_green(green)
        blue_out = self.fc_blue(blue)
        # Concatenation of the colors
        x = self.tanh(cat((red_out, green_out, blue_out), dim=1))
        return x

class CNNLayer(nn.Module):
    def __init__(self, obs_shape, output_size, use_orthogonal, use_ReLU, kernel_size=2, stride=1, input_channels=1, output_channels=1):
        super(CNNLayer, self).__init__()


        #TODO: Maybe change the active_func
        active_func = nn.ReLU()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain('relu')
        if input_channels > 1:
            self.multiple_channels = True
        else:
            self.multiple_channels = False

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_width = obs_shape[0]
        input_height = obs_shape[1]

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channels,
                            out_channels=output_channels,
                            kernel_size=kernel_size,
                            stride=stride)
                  ),
            active_func,
            Flatten(),
            init_(nn.Linear(((input_width - kernel_size) // stride + 1) * ((input_height - kernel_size) // stride + 1) * output_channels,
                            output_size)
                  ),
            active_func
        )

    def forward(self, x):
        if not self.multiple_channels:
            x = x.view(x.shape[0], 1, x.shape[1],x.shape[2])
        x = self.cnn(x)
        return x

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        print("In MLP " + str(input_dim))
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
       print("In the MergedModel initialization")
       super(MergedModel, self).__init__()
       self.experiment_name = mlp_args.experiment_name
       self._use_feature_normalization = mlp_args.use_feature_normalization

       if "simple_spread" in self.experiment_name:
           flattened_size = max(mlp_args.num_agents*2, mlp_args.num_landmarks*2)
           input_size = flattened_size*2 + mlp_args.nb_additional_data*2
           self.dim_actor = mlp_args.grid_resolution*2 + mlp_args.nb_additional_data
           self.cnn1 = CNNLayer((mlp_args.grid_resolution, mlp_args.grid_resolution), flattened_size, mlp_args.use_orthogonal, mlp_args.use_ReLU)
           self.cnn2 = CNNLayer((mlp_args.grid_resolution, mlp_args.grid_resolution), flattened_size, mlp_args.use_orthogonal, mlp_args.use_ReLU)
       elif "speaker" in self.experiment_name:
           self.agent_ID = mlp_args.ID
           output_comm = mlp_args.output_comm
           nb_output_channels = mlp_args.num_output_channels
           input_size = 6 + output_comm + 2
           print(input_size)
           self.dim_actor = 1 + (3+output_comm)*mlp_args.grid_resolution
           if self.agent_ID == 1 or obs_shape[0]/(self.dim_actor) > 1:
               print("Movable agent or critic")
               self.cnn2 = SimpleCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), 6, mlp_args.use_orthogonal, mlp_args.use_ReLU, input_channels=3, output_channels=nb_output_channels, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
           else:
               input_size = 3
       
       self.grid_resolution = mlp_args.grid_resolution
       self.nb_additional_data = mlp_args.nb_additional_data
       
       if obs_shape[0]/(self.dim_actor) > 1:
           if "speaker" in self.experiment_name:
               input_size = 6 + output_comm + 2 + 3
           else:
               input_size *= mlp_args.num_agents


       if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_size)

       self.mlp = MLPLayer(input_size, mlp_args.hidden_size, mlp_args.layer_N, mlp_args.use_orthogonal, mlp_args.use_ReLU)

    def forward(self, x):
        # Séparer le tenseur en trois parties autant de fois que nécessaire
        x_inter_list = []

        if "speaker" in self.experiment_name:
            if x.size()[1]//(self.dim_actor) > 1:
                tensor2 = x[:, 1:3*self.grid_resolution+1, :]
                reshaped_tensor = tensor2.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                result = reshaped_tensor[:, :, 0, 0]
                goal_color = result.reshape(-1, 3)

                tensor1 = x[:, self.dim_actor:1+self.dim_actor, :]
                result = tensor1[:, :, :2]
                velocity = result.reshape(-1, 2)

                if "suppbit" not in self.experiment_name:
                    tensor2 = x[:, 1+self.dim_actor:3*self.grid_resolution+1+self.dim_actor, :]
                    reshaped_tensor = tensor2.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                    result = reshaped_tensor[:, :, 0, 0]
                    x2 = result.reshape(-1, 3)
                    tensor3 = x[:, 3*self.grid_resolution+1+self.dim_actor:2*self.dim_actor, :]
                else:
                    tensor2 = x[:, 1+self.dim_actor:4*self.grid_resolution+1+self.dim_actor, :]
                    reshaped_tensor = tensor2.reshape(-1, 4, self.grid_resolution, self.grid_resolution)
                    result = reshaped_tensor[:, :, 0, 0]
                    x2 = result.reshape(-1, 4)
                    tensor3 = x[:, 4*self.grid_resolution+1+self.dim_actor:2*self.dim_actor, :]

                reshaped_tensor = tensor3.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                x3 = self.cnn2(reshaped_tensor)

                x_inter = cat((goal_color, velocity, x2, x3), dim=1)
                x_inter_list.append(x_inter)
            else:
                if self.agent_ID == 1:
                    tensor1 = x[:, 0:1, :]
                    result = tensor1[:, :, :2]
                    velocity = result.reshape(result.shape[0], 2)

                    if "suppbit" not in self.experiment_name:
                        tensor2 = x[:, 1:3*self.grid_resolution+1, :]
                        reshaped_tensor = tensor2.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                        result = reshaped_tensor[:, :, 0, 0]
                        x2 = result.reshape(-1, 3)
                        tensor3 = x[:, 3*self.grid_resolution+1:self.dim_actor, :]
                    else:
                        tensor2 = x[:, 1:4*self.grid_resolution+1, :]
                        reshaped_tensor = tensor2.reshape(-1, 4, self.grid_resolution, self.grid_resolution)
                        result = reshaped_tensor[:, :, 0, 0]
                        x2 = result.reshape(-1, 4)
                        tensor3 = x[:, 4*self.grid_resolution+1:self.dim_actor, :]

                    reshaped_tensor = tensor3.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                    x3 = self.cnn2(reshaped_tensor)

                    x_inter = cat((velocity, x2, x3), dim=1)
                    x_inter_list.append(x_inter)
                else:
                    tensor2 = x[:, 1:3*self.grid_resolution+1, :]
                    reshaped_tensor = tensor2.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                    result = reshaped_tensor[:, :, 0, 0]
                    goal_color = result.reshape(-1, 3)

                    x_inter = goal_color
                    x_inter_list.append(x_inter)
                    
        else:
            for i in range(x.size()[1]//(self.dim_actor)):
                if "simple_spread" in self.experiment_name:
                    tensor1 = x[:, i*self.dim_actor:self.nb_additional_data+i*self.dim_actor, :]
        
                    result = tensor1[:, :, :2]
                    additional_data = result.reshape(result.shape[0], 2*self.nb_additional_data)
                    
                    tensor2 = x[:, self.nb_additional_data+i*self.dim_actor:self.grid_resolution+self.nb_additional_data+i*self.dim_actor, :]
                    tensor3 = x[:, self.grid_resolution+self.nb_additional_data+i*self.dim_actor:self.dim_actor+i*self.dim_actor, :]
        
                    x1 = self.cnn1(tensor2)
                    x2 = self.cnn2(tensor3)
                    x_inter = cat((additional_data, x1, x2), dim=1)
                    x_inter_list.append(x_inter)
                else:
                    tensor1 = x[:, i*self.dim_actor:1+i*self.dim_actor, :]
                    result = tensor1[:, :, :2]
                    velocity = result.reshape(result.shape[0], 2)
    
                    tensor2 = x[:, 1+i*self.dim_actor:2+i*self.dim_actor, :]
                    result = tensor2[:, :, :3]
                    color = result.reshape(result.shape[0], 3)
    
                    tensor3 = x[:, 2+i*self.dim_actor:3+i*self.dim_actor, :]
                    result = tensor3[:, :, :10]
                    comm = result.reshape(result.shape[0], 10)
    
                    tensor4 = x[:, 3+i*self.dim_actor:(i+1)*self.dim_actor, :]
    
                    x4 = self.cnn(tensor4)
                    x_inter = cat((velocity, color, comm, x4), dim=1)
                    x_inter_list.append(x_inter)
        # Concatenate the output of the CNN with position and velocity
        x = cat(x_inter_list, dim=1)
        # Give x to the MLP
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x = self.mlp(x)
        return x
