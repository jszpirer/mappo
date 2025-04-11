import torch.nn as nn
import spconv.pytorch as spconv
from torch import cat, chunk, div, add, abs, max, int32
from .util import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class SimpleSparseCNN(nn.Module):
    def __init__(self, obs_shape, output_size, use_orthogonal, use_ReLU, kernel_size=2, stride=1, input_channels=1, output_channels=1):
        super(SimpleSparseCNN, self).__init__()

        # Create separate convolutional layers for each channel
        self.conv_red = spconv.SparseConv2d(
            in_channels=1,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False
        )
        self.conv_green = spconv.SparseConv2d(
            in_channels=1,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False
        )
        self.conv_blue = spconv.SparseConv2d(
            in_channels=1,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False
        )
        self.tanh = nn.Tanh()
        #print(use_ReLU)
        #self.final_activ = nn.ReLU() if use_ReLU else nn.Tanh()
        size = (obs_shape[1] - (kernel_size - 1) - 1) // stride + 1
        self.output_channels = output_channels
        self.fc_red = nn.Linear(in_features=size * size, out_features=output_size//3)
        self.fc_green = nn.Linear(in_features=size * size, out_features=output_size//3)
        self.fc_blue = nn.Linear(in_features=size * size, out_features=output_size//3)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # Convert x to a sparse tensor
        sparse_x = x.to_sparse()

        # Extract features and indices from sparse tensor
        features = sparse_x.values()
        indices = sparse_x.indices().permute(1, 0).contiguous().int()
        spatial_shape = sparse_x.shape[1:]

        # Split the features into three channels
        num_features = features.size(0) // 3
        red_features = features[:num_features]
        green_features = features[num_features:2*num_features]
        blue_features = features[2*num_features:]

        # Create SparseConvTensors for each channel
        red_tensor = spconv.SparseConvTensor(red_features, indices[:num_features], spatial_shape, batch_size=x.size(0))
        green_tensor = spconv.SparseConvTensor(green_features, indices[num_features:2*num_features], spatial_shape, batch_size=x.size(0))
        blue_tensor = spconv.SparseConvTensor(blue_features, indices[2*num_features:], spatial_shape, batch_size=x.size(0))

        # Apply convolutional layers to each sparse tensor
        red_output = self.tanh(self.conv_red(red_tensor).dense())
        green_output = self.tanh(self.conv_green(green_tensor).dense())
        blue_output = self.tanh(self.conv_blue(blue_tensor).dense())

        print(self.conv_red.weight)

        # Flatten the outputs
        red_flat = red_output.view(red_output.size(0), -1)
        green_flat = green_output.view(green_output.size(0), -1)
        blue_flat = blue_output.view(blue_output.size(0), -1)

        # Pass the flattened outputs through the linear layers
        red_out = self.fc_red(red_flat)
        green_out = self.fc_green(green_flat)
        blue_out = self.fc_blue(blue_flat)

        # Concatenate the outputs of the linear layers
        x = cat((red_out, green_out, blue_out), dim=1)

        return self.tanh(x)

class SimpleCNN(nn.Module):
    def __init__(self, obs_shape, output_size, use_orthogonal, use_ReLU, kernel_size=2, stride=1, input_channels=1, output_channels=1):
        super(SimpleCNN, self).__init__()

        #self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=kernel_size, stride=stride, groups=input_channels, bias=False)
        
        #Separate convolutional layers for each channel
        self.conv_red = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, bias=False)
        self.conv_green = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, bias=False)
        self.conv_blue = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, bias=False)

        self.tanh = nn.Tanh()
        #print(use_ReLU)
        #self.final_activ = nn.ReLU() if use_ReLU else nn.Tanh()
        size = (obs_shape[1] - (kernel_size - 1) - 1) // stride + 1
        self.output_channels = output_channels
        if self.output_channels == 1:
            self.fc = nn.Linear(in_features=size * size, out_features=output_size)
        else:
            self.fc_red = nn.Linear(in_features=size * size, out_features=output_size//3)
            self.fc_green = nn.Linear(in_features=size * size, out_features=output_size//3)
            self.fc_blue = nn.Linear(in_features=size * size, out_features=output_size//3)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.tanh(x)
        #print(self.conv1.weight)
        # flatten
        #x = x.view(x.size(0), -1)
        
        # Split the input into three channels
        red, green, blue = chunk(x, 3, dim=1)

        # Apply convolution to each channel
        red = self.conv_red(red)
        green = self.conv_green(green)
        blue = self.conv_blue(blue)

        # Apply activation function
        red = self.tanh(red)
        green = self.tanh(green)
        blue = self.tanh(blue)

        # Flatten each channel
        red = red.view(red.size(0), -1)
        green = green.view(green.size(0), -1)
        blue = blue.view(blue.size(0), -1)


        if self.output_channels == 1:
            out = self.fc(x)
            x = self.tanh(out)
        else:
            # chunk the 1D vector to separate the colors
            #red, green, blue = chunk(x, 3, dim=1)
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
           output_entities = mlp_args.output_entities
           output_other = mlp_args.output_other
           nb_output_channels = mlp_args.num_output_channels
           input_size = output_entities + output_other + output_comm + 2
           print(input_size)
           self.dim_actor = 1 + (3+output_comm)*mlp_args.grid_resolution
           if "multiple" in self.experiment_name:
               self.dim_actor += mlp_args.grid_resolution
           print("Sizes")
           print(obs_shape[0])
           print(self.dim_actor)
           print(output_entities)
           if self.agent_ID != 0 or obs_shape[0]/(self.dim_actor) > 1:
               print("Movable agent or critic")
               if "sparse" in self.experiment_name:
                   print("Sparse convolution")
                   self.cnn2 = SimpleSparseCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), output_entities, mlp_args.use_orthogonal, mlp_args.use_ReLU, input_channels=1, output_channels=1, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
               else:
                   self.cnn2 = SimpleCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), output_entities, mlp_args.use_orthogonal, mlp_args.use_ReLU, input_channels=3, output_channels=nb_output_channels, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
               if "multiple" in self.experiment_name:
                   print("Multiple case")
                   self.cnn3 = SimpleCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), output_other, mlp_args.use_orthogonal, mlp_args.use_ReLU, input_channels=1, output_channels=1, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
           else:
               print("Speaker")
               input_size = 3
       
       self.grid_resolution = mlp_args.grid_resolution
       self.nb_additional_data = mlp_args.nb_additional_data
       
       if obs_shape[0]/(self.dim_actor) > 1:
           if "speaker" in self.experiment_name:
               input_size = output_entities + output_comm + 2 + 3 + output_other
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
                    tensor3 = x[:, 3*self.grid_resolution+1+self.dim_actor:6*self.grid_resolution+1+self.dim_actor, :]
                else:
                    tensor2 = x[:, 1+self.dim_actor:4*self.grid_resolution+1+self.dim_actor, :]
                    reshaped_tensor = tensor2.reshape(-1, 4, self.grid_resolution, self.grid_resolution)
                    result = reshaped_tensor[:, :, 0, 0]
                    x2 = result.reshape(-1, 4)
                    tensor3 = x[:, 4*self.grid_resolution+1+self.dim_actor:2*self.dim_actor, :]

                reshaped_tensor = tensor3.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                x3 = self.cnn2(reshaped_tensor)

                if "multiple" in self.experiment_name:
                    tensor4 = x[:, 6*self.grid_resolution+1+self.dim_actor:2*self.dim_actor, :]
                    reshaped_tensor = tensor4.reshape(-1, 1, self.grid_resolution, self.grid_resolution)
                    x4 = self.cnn3(reshaped_tensor)
                    x_inter = cat((goal_color, velocity, x2, x3, x4), dim=1)
                    x_inter_list.append(x_inter)
                else:
                    x_inter = cat((goal_color, velocity, x2, x3), dim=1)
                    x_inter_list.append(x_inter)
            else:
                if self.agent_ID != 0:
                    tensor1 = x[:, 0:1, :]
                    result = tensor1[:, :, :2]
                    velocity = result.reshape(result.shape[0], 2)

                    if "suppbit" not in self.experiment_name:
                        tensor2 = x[:, 1:3*self.grid_resolution+1, :]
                        reshaped_tensor = tensor2.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                        result = reshaped_tensor[:, :, 0, 0]
                        x2 = result.reshape(-1, 3)
                        tensor3 = x[:, 3*self.grid_resolution+1:6*self.grid_resolution+1, :]
                    else:
                        tensor2 = x[:, 1:4*self.grid_resolution+1, :]
                        reshaped_tensor = tensor2.reshape(-1, 4, self.grid_resolution, self.grid_resolution)
                        result = reshaped_tensor[:, :, 0, 0]
                        x2 = result.reshape(-1, 4)
                        tensor3 = x[:, 4*self.grid_resolution+1:self.dim_actor, :]

                    reshaped_tensor = tensor3.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                    x3 = self.cnn2(reshaped_tensor)

                    if "multiple" in self.experiment_name:
                        tensor4 = x[:, 6*self.grid_resolution+1:self.dim_actor, :]
                        reshaped_tensor = tensor4.reshape(-1, 1, self.grid_resolution, self.grid_resolution)
                        x4 = self.cnn3(reshaped_tensor)
                        x_inter = cat((velocity, x2, x3, x4), dim=1)
                        x_inter_list.append(x_inter)
                    else:
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
