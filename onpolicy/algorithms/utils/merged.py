import torch.nn as nn
import spconv.pytorch as spconv
from torch import cat, sparse_coo_tensor, float32, zeros, bool
from .util import init
from math import ceil

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class EgoAttentionMechanism(nn.Module):
    def __init__(self, output_dim, d_model=64, nhead=4):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Encoder for the positions
        self.neighbor_encoder = nn.Sequential(nn.Linear(2, self.d_model),
                                                nn.ReLU(),
                                                nn.Linear(self.d_model, self.d_model))

        # Attention module
        self.lnkv = nn.LayerNorm(self.d_model)
        self.lnq = nn.LayerNorm(self.d_model)
        self.attn = nn.MultiheadAttention(
                    embed_dim = self.d_model,
                    num_heads = nhead,
                    dropout = 0.0,
                    batch_first = True
                )
        self.lnout = nn.LayerNorm(self.d_model)

        # Linear layer to get the right ouput size
        self.fc = nn.Linear(self.d_model, out_features=output_dim)
        self.tanh = nn.Tanh()

    def forward(self, list_x, list_mask=None):
        for i, x in enumerate(list_x) :      
            B, N, _ = x.shape

            # Encoding of the positions of the neighbors
            neigh = self.neighbor_encoder(x)
            
            # Encoding of the ego value (0,0)
            ego_in = zeros(B, 1, 2, device=x.device)
            ego = self.neighbor_encoder(ego_in)

            # Concatenation of the encoded values
            tokens = cat([ego, neigh], dim=1)

            # Ego is never masked
            if list_mask[i] is not None:
                full_mask = cat([zeros(B, 1, dtype=bool, device=list_mask[i].device), list_mask[i]], dim=1)
            
            # Attention blocks
            xkv = self.lnkv(tokens)
            xq = self.lnq(ego)
            attn_out, _ = self.attn(xq, xkv, xkv, key_padding_mask=full_mask)
            xq = xq + attn_out
            xq = self.lnout(xq)

            # Linear layer to get the right size for the output
            x = xq.squeeze(1)
            return self.tanh(self.fc(x))


class SelfAttentionMechanism(nn.Module):
    def __init__(self, output_dim, num_agents, d_model=64, nhead=4, attn_layers=1):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Encoder for the positions
        self.neighbor_encoder = nn.Sequential(nn.Linear(2, self.d_model),
                                                nn.ReLU(),
                                                nn.Linear(self.d_model, self.d_model))

        # Attention module
        self.attn_blocks = nn.ModuleList()
        for _ in range(attn_layers):
            block = nn.ModuleDict({
                "ln1": nn.LayerNorm(self.d_model),
                "attn": nn.MultiheadAttention(
                    embed_dim = self.d_model,
                    num_heads = nhead,
                    dropout = 0.0,
                    batch_first = True
                ),
                "ln2": nn.LayerNorm(self.d_model)
            })
            self.attn_blocks.append(block)
        
        # Linear layer to get the right ouput size
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(self.d_model * num_agents, out_features=output_dim)
        
    def forward(self, list_x, list_mask=None):
        # x should be a numpy array, mask too if not None
        for i, x in enumerate(list_x) :      
            B, N, _ = x.shape

            # Encoding of the positions of the neighbors
            positions = self.neighbor_encoder(x)
            
            # Attention blocks
            x = positions
            for blk in self.attn_blocks:
                xn = blk["ln1"](x)
                attn_out, _ = blk["attn"](xn, xn, xn)
                x = x + attn_out
                x = blk["ln2"](x)
            
            # Linear layer 
            x = x.view(x.size(0), -1)
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
       if "coverage" in self.experiment_name:
            #only the local case for now
            flattened_size -= num_landmarks_features
            if "local" in self.experiment_name:
                input_size = flattened_size + mlp_args.nb_additional_data
                self.dim_actor = 2
            else:
                input_size = flattened_size + 2*mlp_args.nb_additional_data
                self.dim_actor = 3
       if self.omniscient_critic and self.critic:
            if "rvr" in self.experiment_name:
                self.cnn1 = SimplSparseSpreadCNN((mlp_args.grid_resolution_critic, mlp_args.grid_resolution_critic), 12, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel, input_channels=3)
                self.cnn2 = SimplSparseSpreadCNN((mlp_args.grid_resolution_critic, mlp_args.grid_resolution_critic), 5, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel, input_channels=1)
                input_size = 17
            else:
                if self.attention_critic:
                    self.attn = SelfAttentionMechanism(flattened_size, mlp_args.num_agents)
                else:
                    self.cnn1 = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), flattened_size, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
                self.dim_actor = 1
                input_size -= 2
       else:
            # Actor case
            if "rvr" in self.experiment_name:
                self.cnn1 = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), 12, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel, input_channels=3, padding_size=padding_actor)
                self.cnn2 = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), 5, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel, input_channels=1)
                input_size = 19
            else:
                if self.attention_actor:
                    self.attn = EgoAttentionMechanism(flattened_size)
                else:
                    self.cnn1 = SimplSparseSpreadCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), flattened_size, mlp_args.use_orthogonal, mlp_args.use_ReLU, stride=mlp_args.stride, kernel_size=mlp_args.kernel)
              
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
                        else:
                            x1 = self.cnn1([x[0]])
                        x_inter = x1
                    else:
                        velocity = x[i*self.dim_actor + 0]
                        if self.attention_actor:
                            x1 = self.attn([x[i*self.dim_actor + 1]], list_mask=mask)
                        else:
                            x1 = self.cnn1([x[i*self.dim_actor + 1]])
                        x_inter = cat((velocity, x1), dim=1)
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
                position = x[i*self.dim_actor + 0]

                velocity = x[i*self.dim_actor + 1]

                x1 = self.cnn1([x[i*self.dim_actor + 2]])

                x_inter = cat((velocity, position, x1), dim=1)
            x_inter_list.append(x_inter)
        # Concatenate the output of the CNN with position and velocity
        x = cat(x_inter_list, dim=1)
        # Give x to the MLP
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x = self.mlp(x)
        return x
