import torch.nn as nn
import spconv.pytorch as spconv
from torch import cat, sparse_coo_tensor, float32, zeros, bool, randn
from .util import init
from math import ceil

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class EgoAttentionMechanism(nn.Module):
    def __init__(self, output_dim, d_model=64, nhead=4, nb_features=2):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim

        # Learned information that we are missing
        self.ego_query = nn.Parameter(randn(1, 1, d_model) * 0.02)
        self.null_token = nn.Parameter(zeros(1, 1, d_model))
        
        # Encoder for the positions
        self.neighbor_encoder = nn.Sequential(nn.Linear(nb_features, self.d_model),
                                                nn.Tanh(),
                                                nn.Linear(self.d_model, self.d_model),
                                                nn.Tanh())

        # Attention module
        self.lnkv = nn.LayerNorm(self.d_model)
        self.lnq = nn.LayerNorm(self.d_model)
        self.attn = nn.MultiheadAttention(
                    embed_dim = self.d_model,
                    num_heads = nhead,
                    dropout = 0.0,
                    batch_first = True
                )
        
        #self.lnout = nn.LayerNorm(self.d_model)
 
        #self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model),
                                 #nn.ReLU(),
                                 #nn.Linear(4 * d_model, d_model))
        
        # Linear layer to get the right ouput size
        self.fc = nn.Linear(self.d_model, out_features=output_dim)
        self.tanh = nn.Tanh()

    def forward(self, list_x, list_mask=None):
        for i, x in enumerate(list_x) :      
            B, N, _ = x.shape

            # Encoding of the positions of the neighbors
            neigh = self.neighbor_encoder(x)
            
            # Encoding of the ego value (0,0)
            ego = self.ego_query.expand(B, 1, -1)

            # Concatenation of the encoded values
            tokens = cat([ego, neigh], dim=1)

            # Ego is never masked
            if list_mask is not None:
                full_mask = cat([zeros(B, 1, dtype=bool, device=list_mask.device), list_mask], dim=1)
            
            # Attention blocks
            xkv = tokens
            xq = ego
            if list_mask is not None:
                attn_out, _ = self.attn(xq, xkv, xkv, key_padding_mask=full_mask)
            else:
                attn_out, _ = self.attn(xq, xkv, xkv)

            xq = attn_out

            # Linear layer to get the right size for the output
            x = xq.squeeze(1)
            return self.tanh(self.fc(x))


class SelfAttentionMechanism(nn.Module):
    def __init__(self, output_dim, d_model=64, nhead=4, input_dim=2):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Encoder for the positions
        self.neighbor_encoder = nn.Sequential(nn.Linear(input_dim, self.d_model),
                                                nn.Tanh(),
                                                nn.Linear(self.d_model, self.d_model),
                                                nn.Tanh())

        # Attention module
        self.ln = nn.LayerNorm(self.d_model)
        self.attn = nn.MultiheadAttention(
                    embed_dim = self.d_model,
                    num_heads = nhead,
                    dropout = 0.0,
                    batch_first = True
                )
        
        # Linear layer to get the right ouput size
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(self.d_model, out_features=output_dim)
        
    def forward(self, list_x, list_mask=None):
        # x should be a numpy array, mask too if not None
        for i, x in enumerate(list_x) :      
            B, N, _ = x.shape

            # Encoding of the positions of the neighbors
            positions = self.neighbor_encoder(x)
            
            # Attention blocks
            x = positions
            attn_out, _ = self.attn(x, x, x)
            
            # Linear layer
            x = attn_out.mean(dim=1)
            return self.tanh(self.fc(x))

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
       self.attention_actor = mlp_args.attention_actor
       self.attention_critic = mlp_args.attention_critic
       padding_actor=mlp_args.padding
       velocities_critic = mlp_args.velocities_critic
       if self.critic and self.omniscient_critic:
           self.dim_actor = 3
       self.num_obstacles = mlp_args.num_obstacles
       if mlp_args.num_landmarks == 0:
           num_landmarks_features = 6
       else:
           num_landmarks_features = mlp_args.num_landmarks*2
       flattened_size = mlp_args.num_agents*2 + num_landmarks_features
       input_size = flattened_size + mlp_args.nb_additional_data*2
       if "local" in self.experiment_name:
            input_size = flattened_size + mlp_args.nb_additional_data
       if "coverage" in self.experiment_name:
            #only the local case for now
            flattened_size -= num_landmarks_features
            if "local" in self.experiment_name:
                if self.attention_actor:
                    flattened_size -= 2
                input_size = flattened_size + mlp_args.nb_additional_data
                self.dim_actor = 2
                input_size = 22
                if "global" in self.experiment_name:
                    self.dim_actor = 3
                    input_size += 2
                if self.num_obstacles != 0:
                    self.dim_actor = 3
                    input_size = 34
            else:
                input_size = flattened_size + 2*mlp_args.nb_additional_data
                self.dim_actor = 3
                if self.num_obstacles != 0:
                    dim_actor = 4
                    input_size += 2*self.num_obstacles
                    input_size += 4 # For the walls
       if self.omniscient_critic and self.critic:
            if "rvr" in self.experiment_name:
                self.cnn1 = SimplSparseSpreadCNN((mlp_args.grid_resolution_critic, mlp_args.grid_resolution_critic), 12, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel, input_channels=3)
                self.cnn2 = SimplSparseSpreadCNN((mlp_args.grid_resolution_critic, mlp_args.grid_resolution_critic), 5, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel, input_channels=1)
                input_size = 17
            else:
                self.dim_actor = 0
                if self.attention_critic:
                    if velocities_critic:
                        self.attn = SelfAttentionMechanism(mlp_args.num_agents*2, d_model=mlp_args.d_model, input_dim=4)
                    else:
                        self.attn = SelfAttentionMechanism(mlp_args.num_agents*2, d_model=mlp_args.d_model)
                    input_size = mlp_args.num_agents*2 + 2
                    if self.num_obstacles != 0:
                        self.attn_obs = SelfAttentionMechanism(mlp_args.num_obstacles*2 + 8, d_model=mlp_args.d_model)
                        input_size += mlp_args.num_obstacles*2
                        input_size += 8
                        self.dim_actor += 1
                else:
                    if not self.attention_actor:
                        self.cnn1 = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), flattened_size, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel, input_channels=3)
                        self.dim_actor = 2
                        input_size = flattened_size + 2
                        if self.num_obstacles != 0:
                            self.cnn2 = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), mlp_args.num_obstacles * 2 + 8, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel, input_channels=1)
                            self.dim_actor += 1
                            input_size += mlp_args.num_obstacles * 2
                            input_size += 8
                    else :
                        input_size = mlp_args.num_agents*2 + 2
                self.dim_actor += 1
                input_size -= 2
       else:
            if "rvr" in self.experiment_name:
                self.cnn1 = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), 12, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel, input_channels=3, padding_size=padding_actor)
                self.cnn2 = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), 5, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel, input_channels=1)
                input_size = 19
            else:
                if self.attention_actor and not self.critic:
                    self.attn = EgoAttentionMechanism(20, d_model=mlp_args.d_model)
                    if self.num_obstacles != 0:
                        if "global" in self.experiment_name:
                            nb_features = 4
                        else:
                            nb_features = 2
                        self.attn_obs = EgoAttentionMechanism(12, d_model=mlp_args.d_model, nb_features=nb_features)
                elif self.attention_critic:
                    self.attn = EgoAttentionMechanism(flattened_size, d_model=mlp_args.d_model)
                else:
                    self.cnn1 = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), flattened_size, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel, input_channels=1)
                    if self.num_obstacles != 0:
                        self.cnn2 = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), 12, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel, input_channels=3)
                        self.dim_actor = 6
              
       self.nb_additional_data = mlp_args.nb_additional_data
       
       if not self.omniscient_critic and obs_shape[0]/(self.dim_actor) > 1:
            input_size *= mlp_args.num_agents

       if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_size)

       self.mlp = MLPLayer(input_size, mlp_args.hidden_size, mlp_args.layer_N, mlp_args.use_orthogonal, mlp_args.use_ReLU)

    def forward(self, x, mask=None):
        # Séparer le tenseur en trois parties autant de fois que nécessaire
        x_inter_list = []
        for i in range(len(x)//(self.dim_actor)):
            if "local" in self.experiment_name:
                if self.critic and self.omniscient_critic and "rvr" in self.experiment_name:
                    x1 = self.cnn1(x[:3])
                    x2 = self.cnn2([x[3]])
                    x_inter = cat((x1, x2), dim=1)
                    #x_inter = x1
                elif "coverage" in self.experiment_name:
                    if self.critic and self.omniscient_critic:
                        if self.attention_critic:
                            x1 = self.attn([x[0]])
                            if self.num_obstacles != 0:
                                x2 = self.attn_obs([x[1]])
                                x_inter = cat((x1, x2), dim=1)
                            else:
                                x_inter = x1
                        elif self.attention_actor:
                            x1 = x[0].reshape(x[0].size(0), x[0].size(1) * x[0].size(2))
                            x_inter = x1
                        else:
                            x_inter = self.cnn1(x[:3])
                    elif self.attention_critic and not self.attention_actor:
                        action = x[i*self.dim_actor + 0]
                        positions = x[i*self.dim_actor + 1].reshape(x[i*self.dim_actor + 1].size(0), x[i*self.dim_actor + 1].size(1) * x[i*self.dim_actor + 1].size(2))
                        x_inter = cat((action, positions), dim=1)
                    else:
                        velocity = x[i*self.dim_actor + 0]
                        if self.attention_actor:
                            if "global" in self.experiment_name:
                                x1 = self.attn([x[i*self.dim_actor + 2]], list_mask=mask[i*self.dim_actor + 0])
                                position = x[i*self.dim_actor + 1]
                                x_inter = cat((velocity, position, x1), dim=1)
                            else:
                                x1 = self.attn([x[i*self.dim_actor + 1]], list_mask=mask[i*self.dim_actor + 0])
                                if self.num_obstacles != 0:
                                    x2 = self.attn_obs([x[i*self.dim_actor + 2]], list_mask=mask[i*self.dim_actor + 1])
                                    x_inter = cat((velocity, x1, x2), dim=1)
                                else:
                                    x_inter = cat((velocity, x1), dim=1)
                        else:
                            x1 = self.cnn1([x[i*self.dim_actor + 1]])
                            x_inter = cat((velocity, x1), dim=1)
                else:
                    velocity = x[i*self.dim_actor + 0]
                    if "rvr" in self.experiment_name:
                        x1 = self.cnn1(x[1:4])
                        x2 = self.cnn2(x[4])
                        x_inter = cat((velocity, x1, x2), dim=1)
                    else:
                        x1 = self.cnn1([x[i*self.dim_actor + 1]])
                        x_inter = cat((velocity, x1), dim=1)
            else:
                if self.critic and self.omniscient_critic and self.num_obstacles != 0:
                    if self.attention_critic:
                        x1 = self.attn([x[0]])
                        x2 = self.attn_obs([x[1]])
                    else:
                        x1 = self.cnn1(x[:3])
                        x2 = self.cnn2([x[3]])
                    x_inter = cat((x1, x2), dim=1)
                elif self.critic and not self.attention_critic:
                    x1 = self.cnn1(x[:3])
                    if self.num_obstacles != 0:
                        x2 = self.cnn2([x[3]])
                        x_inter = cat((x1, x2), dim=1)
                    else:
                        x_inter = x1
                else:
                    position = x[i*self.dim_actor + 0]
                    velocity = x[i*self.dim_actor + 1]
                    if self.attention_critic:
                        x1 = self.attn([x[i*self.dim_actor + 2]])
                        if self.num_obstacles != 0:
                            x2 = self.attn_obs([x[i*self.dim_actor + 3]])
                            x_inter = cat((velocity, position, x1, x2), dim=1)
                        else:
                            x_inter = cat((velocity, position, x1), dim=1)
                    else:
                        x1 = self.cnn1([x[i*self.dim_actor + 2]])
                        if self.num_obstacles != 0:
                            x2 = self.cnn2(x[i*self.dim_actor + 3:i*self.dim_actor + 6])
                            x_inter = cat((velocity, position, x1, x2), dim=1)
                        else:
                            x_inter = cat((velocity, position, x1), dim=1)
            x_inter_list.append(x_inter)
        # Concatenate the output of the CNN with position and velocity
        x = cat(x_inter_list, dim=1)
        # Give x to the MLP
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x = self.mlp(x)
        return x
