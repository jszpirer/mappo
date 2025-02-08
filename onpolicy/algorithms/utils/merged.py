import torch.nn as nn
from .util import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNLayer(nn.Module):
    def __init__(self, obs_shape, output_size, use_orthogonal, use_ReLU, kernel_size=2, stride=1):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=1,
                            kernel_size=kernel_size,
                            stride=stride)
                  ),
            active_func,
            Flatten(),
            init_(nn.Linear((input_width - kernel_size + stride) * (input_height - kernel_size + stride),
                            output_size)
                  ),
            active_func
        )

    def forward(self, x):
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
            x = self.fc2i
        return x

class MergedModel(nn.Module):
    def __init__(self, mlp_args, obs_shape):
       (MergedModel, self).__init__()

        self.cnn = CNNLayer(obs_shape, 10, mlp_args.use_orthogonal, mlp_args.use_ReLU)
        flattened_size = mlp_args.output_size
        self.mlp = MLPLayer(flattened_size, mlp_args.hidden_size, mlp_args.layer_N, mlp_args.use_orthogonal, mlp_args.use_ReLU)

    def forward(self, x):
        # Extract positon and velocity from x
        additional_data = x[:, -4:]
        x = x[:, :-4]

        # Give x to the CNN
        x = self.cnn(x)

        # Concatenate the output of the CNN with position and velocity
        x = torch.cat((x, additional_data), dim=1)

        # Give x to the MLP
        x = self.mlp(x)
        return x
