import torch.nn as nn
import spconv.pytorch as spconv
from torch.cuda.amp import autocast
from torch import cat, chunk, set_printoptions, inf, sparse_coo_tensor
from .util import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class SimpleSparseMemoryOptimized(nn.Module):
    def __init__(self, obs_shape, output_size, use_orthogonal, use_ReLU, kernel_size=2, stride=1, input_channels=1, output_channels=1):
        super(SimpleSparseMemoryOptimized, self).__init__()

        
        self.net = spconv.SparseSequential(
            spconv.SparseConv2d(in_channels=1, out_channels=output_channels, kernel_size=kernel_size, stride=stride, bias=False),
            nn.Tanh()
        )
        input_width = obs_shape[0]
        self.size = ((input_width - kernel_size) // stride + 1)
        self.fc = nn.Linear(in_features=self.size * self.size, out_features=output_size)       
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Use autocast for mixed precision
        #with autocast():
        # Coalesce once
        sparse = x.coalesce()
        indices = sparse.indices().permute(1, 0).contiguous().int()
        values = sparse.values().view(-1, 1).float() #.half() # Use float16 if supported
            
        sparse_tensor = spconv.SparseConvTensor(
                features=values,
                indices=indices,
                spatial_shape=x.size()[1:],
                batch_size=x.size(0)
            )

        output = self.net(sparse_tensor)
            
            # Flatten spatial coordinates
        coords = output.indices
        coords[:, 1] = coords[:, 1] * self.size + coords[:, 2]
        flat_indices = coords[:, :2].permute(1, 0).contiguous().int()
        flat_values = output.features.view(-1)

        flat = sparse_coo_tensor(flat_indices, flat_values, size=(x.size(0), self.size * self.size))
        x = self.fc(flat)
        return self.tanh(x)

class SimplSparseSpreadCNN(nn.Module):
    def __init__(self, obs_shape, output_size, use_orthogonal, use_ReLU, kernel_size=2, stride=1, input_channels=1, output_channels=1):
        super(SimplSparseSpreadCNN, self).__init__()

        # Create separate convolutional layers for each channel
        self.net = spconv.SparseSequential(
            spconv.SparseConv2d(in_channels=1, out_channels=output_channels, kernel_size=kernel_size, stride=stride, bias=False),
            nn.Tanh()
        )
        self.tanh = nn.Tanh()
        input_width = obs_shape[0]
        input_height = obs_shape[1]
        self.size = ((input_width - kernel_size) // stride + 1)
        print("La size est")
        print(self.size)
        self.fc = nn.Linear(in_features=self.size * self.size, out_features=output_size)

    def forward(self, x):
        #set_printoptions(threshold=inf)

        sparse = x.coalesce()
        indices = sparse.indices().permute(1, 0).contiguous().int()
        values = sparse.values().view(-1, 1)
        sparse = spconv.SparseConvTensor(values, indices, x.size()[1:], batch_size = x.size()[0])

        # Apply convolutional layers to each sparse tensor
        output = self.net(sparse)

        coords = output.indices
        new_coords = coords[:, :2].clone()
        new_coords[:,1] = coords[:, 1] * self.size + coords[:, 2]
        output.indices = new_coords

        # Flatten the outputs
        flat_indices = output.indices.permute(1, 0).contiguous().int()
        flat_values = output.features.view(output.features.shape[0])
        flat = sparse_coo_tensor(flat_indices, flat_values, size=(x.size()[0], self.size*self.size))

        # Pass the flattened outputs through the linear layers
        x = self.fc(flat)

        return self.tanh(x)

class SimpleCNN(nn.Module):
    def __init__(self, obs_shape, output_size, use_orthogonal, use_ReLU, kernel_size=2, stride=1, input_channels=1, output_channels=1):
        super(SimpleCNN, self).__init__()

        #Separate convolutional layers for each channel
        self.conv_red = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, bias=False)
        self.conv_green = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, bias=False)
        self.conv_blue = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, bias=False)

        self.tanh = nn.Tanh()
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
            red_out = self.fc_red(red)
            green_out = self.fc_green(green)
            blue_out = self.fc_blue(blue)
            # Concatenation of the colors
            x = self.tanh(cat((red_out, green_out, blue_out), dim=1))
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

       if "simple_spread" in self.experiment_name or "aggregation" in self.experiment_name or "chocolate" in self.experiment_name:
           flattened_size = max(mlp_args.num_agents*2, mlp_args.num_landmarks*2)
           input_size = flattened_size*2 + mlp_args.nb_additional_data*2
           if "local" in self.experiment_name:
               input_size = flattened_size*2 + mlp_args.nb_additional_data
               self.dim_actor = 3
           else:
               self.dim_actor = 4
           if "memory" in self.experiment_name:
               flattened_size = 4
               input_size = flattened_size*2 + mlp_args.nb_additional_data*2
               self.cnn1 = SimpleSparseMemoryOptimized((mlp_args.grid_resolution, mlp_args.grid_resolution), flattened_size, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
               self.cnn2 = SimpleSparseMemoryOptimized((mlp_args.grid_resolution, mlp_args.grid_resolution), flattened_size, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
           else:
               self.cnn1 = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), flattened_size, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
               self.cnn2 = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), flattened_size, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
       elif "coverage" in self.experiment_name:
           if "memory" in self.experiment_name:
               flattened_size = 20
           else:
               flattened_size = mlp_args.num_agents*2
           input_size = flattened_size + mlp_args.nb_additional_data*2
           if "local" in self.experiment_name:
               input_size = flattened_size + mlp_args.nb_additional_data
               self.dim_actor = 2
           else:
               self.dim_actor = 3
           self.cnn1 = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), flattened_size, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
       elif "reference" in self.experiment_name:
           output_entities = mlp_args.output_entities
           self.dim_actor = 6
           output_comm = mlp_args.output_comm
           input_size = output_entities + output_comm + 2 + 3
           self.cnn2_red = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), output_entities//3, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
           self.cnn2_blue = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), output_entities//3, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
           self.cnn2_green = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), output_entities//3, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
       elif "speaker" in self.experiment_name:
           self.agent_ID = mlp_args.ID
           output_comm = mlp_args.output_comm
           output_entities = mlp_args.output_entities
           output_other = mlp_args.output_other
           nb_output_channels = mlp_args.num_output_channels
           input_size = output_entities + output_other + output_comm + 2
           self.dim_actor = 5
           if "multiple" in self.experiment_name:
               self.dim_actor += mlp_args.grid_resolution
           if self.agent_ID != 0 or obs_shape[0]/(self.dim_actor) > 1:
               print("Movable agent or critic")
               if "sparse" in self.experiment_name:
                   print("Sparse convolution")
                   self.cnn2_red = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), output_entities//3, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
                   self.cnn2_blue = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), output_entities//3, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
                   self.cnn2_green = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), output_entities//3, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
               else:
                   self.cnn2 = SimpleCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), output_entities, mlp_args.use_orthogonal, mlp_args.use_ReLU, input_channels=3, output_channels=nb_output_channels, stride=mlp_args.stride, kernel_size=mlp_args.kernel)

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
            # First, need to differentiate the critic and the actor
            if len(x) > (self.dim_actor):
                goal_color = x[0]

                velocity = x[1]

                x2 = x[5]
                
                x3 = self.cnn2_red(x[2])

                x4 = self.cnn2_blue(x[3])

                x5 = self.cnn2_green(x[4])
                    
                x_inter = cat((goal_color, velocity, x2, x3, x4, x5), dim=1)
                x_inter_list.append(x_inter)
            else:
                if self.agent_ID != 0:
                    velocity = x[0]

                    x2 = x[4]

                    x3 = self.cnn2_red(x[1])

                    x4 = self.cnn2_blue(x[2])

                    x5 = self.cnn2_green(x[3])
                        
                    x_inter = cat((velocity, x2, x3, x4, x5), dim=1)
                    x_inter_list.append(x_inter)
                else:
                    x_inter = x[0]
                    x_inter_list.append(x_inter)
                    
        else:
            for i in range(len(x)//(self.dim_actor)):
                if "simple_spread" in self.experiment_name or "aggregation" in self.experiment_name or "chocolate" in self.experiment_name:
                    if "local" in self.experiment_name:
                        velocity = x[i*self.dim_actor + 0]

                        x1 = self.cnn1(x[i*self.dim_actor + 1])
                        x2 = self.cnn2(x[i*self.dim_actor + 2])
                        x_inter = cat((velocity, x1, x2), dim=1)
                    else:
                        velocity = x[i*self.dim_actor + 0]

                        position = x[i*self.dim_actor + 1]

                        x1 = self.cnn1(x[i*self.dim_actor + 2])
                        x2 = self.cnn2(x[i*self.dim_actor + 3])
                        x_inter = cat((velocity, position, x1, x2), dim=1)
                    x_inter_list.append(x_inter)
                elif "coverage" in self.experiment_name:
                    if "local" in self.experiment_name:
                        velocity = x[i*self.dim_actor + 0]

                        x1 = self.cnn1(x[i*self.dim_actor + 1])
                        x_inter = cat((velocity, x1), dim=1)
                    else:
                        velocity = x[i*self.dim_actor + 0]

                        position = x[i*self.dim_actor + 1]

                        x1 = self.cnn1(x[i*self.dim_actor + 2])
                        x_inter = cat((velocity, position, x1), dim=1)
                    x_inter_list.append(x_inter)
                elif "reference" in self.experiment_name:
                    goal_color = x[i*self.dim_actor + 0]

                    velocity = x[i*self.dim_actor + 1]

                    x1 = self.cnn2_red(x[i*self.dim_actor + 2])

                    x2 = self.cnn2_blue(x[i*self.dim_actor + 3])

                    x3 = self.cnn2_green(x[i*self.dim_actor + 4])

                    comm = x[i*self.dim_actor + 5]

                    x_inter = cat((goal_color, velocity, x1, x2, x3, comm), dim=1)
                    x_inter_list.append(x_inter)
        # Concatenate the output of the CNN with position and velocity
        x = cat(x_inter_list, dim=1)
        # Give x to the MLP
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x = self.mlp(x)
        return x
