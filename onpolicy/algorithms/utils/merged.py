import torch.nn as nn
import spconv.pytorch as spconv
from torch import cat, chunk, inf, sparse_coo_tensor, float32, empty, device, zeros
from .util import init
from math import ceil

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class SimplSparseSpreadCNN(nn.Module):
    def __init__(self, obs_shape, output_size, use_orthogonal, use_ReLU, kernel_size=2, stride=1, input_channels=1, output_channels=1, padding_size=0):
        super(SimplSparseSpreadCNN, self).__init__()

        # Create separate convolutional layers for each channel
        self.net = spconv.SparseSequential(
            spconv.SparseConv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride, bias=False, padding=padding_size),
            nn.Tanh()
        )
        self.tanh = nn.Tanh()
        input_width = obs_shape[0]
        self.size = ((input_width - kernel_size + 2*padding_size) // stride + 1)
        self.output_size = output_size
        self.fc = nn.Linear(in_features=self.size * self.size, out_features=output_size)

    def forward(self, list_x):
        channels_values = []
        
        for x in list_x:
            sparse = x.coalesce()
            values = sparse.values().view(-1, 1).to(float32)
            channels_values.append(values)
        if len(channels_values) > 1:
            values = cat(channels_values, dim=1)
        indices = sparse.indices().permute(1, 0).contiguous().int() 
        sparse = spconv.SparseConvTensor(values, indices, x.size()[1:], batch_size = x.size()[0])

        # Apply convolutional layers to each sparse tensor
        if sparse.features.size()[0] < 1:
            dummy_features = zeros((1, 1), dtype=values.dtype, device=values.device)
            ndim = len(x.size()) 
            dummy_index = zeros((1, ndim), dtype=indices.dtype, device=indices.device)
            sparse = spconv.SparseConvTensor(dummy_features, dummy_index, x.size()[1:], batch_size=x.size()[0])
            #device_for_tensor = device("cuda:0")
            #i = empty((2, 0), device=device_for_tensor)
            #v = empty((0,), device=device_for_tensor)
            #flat = sparse_coo_tensor(i, v, size=(x.size()[0], self.size*self.size), device=device_for_tensor)
            #x = zeros((x.size()[0], self.output_size), device=device_for_tensor)
            #print("No landmark")
        #else:
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
        #print("Something detected")

        return self.tanh(x)

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
    def __init__(self, mlp_args, obs_shape, critic=False):
       super(MergedModel, self).__init__()
       self.experiment_name = mlp_args.experiment_name
       self._use_feature_normalization = mlp_args.use_feature_normalization
       self.omniscient_critic = mlp_args.omniscient_critic
       self.dim_actor = mlp_args.dim_actor
       self.critic = critic
       padding_actor=mlp_args.padding
       if self.critic and self.omniscient_critic:
           self.dim_actor = 3

       if mlp_args.num_landmarks == 0:
           num_landmarks_features = 6
       else:
           num_landmarks_features = mlp_args.num_landmarks*2
       flattened_size = mlp_args.num_agents*2 + num_landmarks_features
       input_size = flattened_size + mlp_args.nb_additional_data*2
       if "local" in self.experiment_name:
            input_size = flattened_size + mlp_args.nb_additional_data

       if self.omniscient_critic and self.critic:
            self.cnn1 = SimplSparseSpreadCNN((mlp_args.grid_resolution_critic, mlp_args.grid_resolution_critic), 12, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel, input_channels=3)
            self.cnn2 = SimplSparseSpreadCNN((mlp_args.grid_resolution_critic, mlp_args.grid_resolution_critic), 5, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel, input_channels=1)
            input_size = 17
       else:
            if "rvr" in self.experiment_name:
                self.cnn1 = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), 12, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel, input_channels=3, padding_size=padding_actor)
                self.cnn2 = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), 5, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel, input_channels=1)
                input_size = 19
            else:
                self.cnn1 = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), flattened_size, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
              
       self.nb_additional_data = mlp_args.nb_additional_data
       
       if not self.omniscient_critic and obs_shape[0]/(self.dim_actor) > 1:
            input_size *= mlp_args.num_agents

       if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_size)

       self.mlp = MLPLayer(input_size, mlp_args.hidden_size, mlp_args.layer_N, mlp_args.use_orthogonal, mlp_args.use_ReLU)

    def forward(self, x):
        # Séparer le tenseur en trois parties autant de fois que nécessaire
        x_inter_list = []
        for i in range(len(x)//(self.dim_actor)):
            if "local" in self.experiment_name:
                if self.critic and self.omniscient_critic:
                    x1 = self.cnn1(x[:3])
                    x2 = self.cnn2([x[3]])
                    x_inter = cat((x1, x2), dim=1)
                    #x_inter = x1
                else:
                    velocity = x[i*self.dim_actor + 0]

                    if "rvr" in self.experiment_name:
                        x1 = self.cnn1(x[1:4])
                        x2 = self.cnn2([x[4]])
                        x_inter = cat((velocity, x1, x2), dim=1)
                        #x_inter = cat((velocity, x1), dim=1)
                    else:
                        x1 = self.cnn1([x[i*self.dim_actor + 1]])
                        x_inter = cat((velocity, x1), dim=1)
            else:
                velocity = x[i*self.dim_actor + 0]

                position = x[i*self.dim_actor + 1]

                x1 = self.cnn1(x[i*self.dim_actor + 2])

                x_inter = cat((velocity, position, x1), dim=1)
            x_inter_list.append(x_inter)
        # Concatenate the output of the CNN with position and velocity
        x = cat(x_inter_list, dim=1)
        # Give x to the MLP
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x = self.mlp(x)
        return x
