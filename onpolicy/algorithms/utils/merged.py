import torch.nn as nn
from torch import cat, chunk, div, add, abs, max
from .util import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class SimpleCNN2(nn.Module):
    def __init__(self, obs_shape, output_size, use_orthogonal, use_ReLU, kernel_size=2, stride=1, input_channels=1, output_channels=1, sigmoid=False):
        super(SimpleCNN2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1, groups=input_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1, groups=input_channels)
        size = 7
        self.fc_red = nn.Linear(in_features=size * size, out_features=output_size//3)
        self.fc_green = nn.Linear(in_features=size * size, out_features=output_size//3)
        self.fc_blue = nn.Linear(in_features=size * size, out_features=output_size//3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # flatten
        x = x.view(x.size(0), -1)
        # chunk the 1D vector to separate the colors
        red, green, blue = chunk(x, 3, dim=1)
        # dense layers
        red_out = self.fc_red(red)
        green_out = self.fc_green(green)
        blue_out = self.fc_blue(blue)
        # Concatenation of the colors
        x = cat((red_out, green_out, blue_out), dim=1)
        x = self.sigmoid(x)
        return x

class SimpleCNN3(nn.Module):
    def __init__(self, obs_shape, output_size, use_orthogonal, use_ReLU, kernel_size=2, stride=1, input_channels=4, output_channels=1, sigmoid=True):
        super(SimpleCNN3, self).__init__()

        self.conv1_message1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=stride, bias=sigmoid)
        self.conv1_message2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=stride, bias=sigmoid)
        self.conv1_message3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=stride, bias=sigmoid)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.relu = nn.ReLU()
        size = (obs_shape[1] - (kernel_size - 1) - 1) // stride + 1
        size = (size - (2 - 1) - 1) // 1 + 1
        self.fc_red = nn.Linear(in_features=size * size, out_features=output_size // 3, bias=sigmoid)
        self.fc_green = nn.Linear(in_features=size * size, out_features=output_size // 3, bias=sigmoid)
        self.fc_blue = nn.Linear(in_features=size * size, out_features=output_size // 3, bias=sigmoid)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        messages = x[:, :3, :, :]  # Les trois premiers canaux pour les messages
        position = x[:, 3:, :, :]  # Le quatrième canal pour la position des robots

        # Concaténation du canal de position avec chaque canal de message
        message1 = cat((messages[:, 0:1, :, :], position), dim=1)
        message2 = cat((messages[:, 1:2, :, :], position), dim=1)
        message3 = cat((messages[:, 2:3, :, :], position), dim=1)

        # Convolutions séparées
        message1 = self.relu(self.maxpool(self.conv1_message1(message1)))
        message2 = self.relu(self.maxpool(self.conv1_message2(message2)))
        message3 = self.relu(self.maxpool(self.conv1_message3(message3)))

        # flatten
        message1 = message1.view(message1.size(0), -1)
        message2 = message2.view(message2.size(0), -1)
        message3 = message3.view(message3.size(0), -1)

        # dense layers
        red_out = self.fc_red(message1)
        green_out = self.fc_green(message2)
        blue_out = self.fc_blue(message3)

        # Concatenation des caractéristiques
        x = cat((red_out, green_out, blue_out), dim=1)
        x = self.softmax(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, obs_shape, output_size, use_orthogonal, use_ReLU, kernel_size=2, stride=1, input_channels=1, output_channels=1, sigmoid=True):
        super(SimpleCNN, self).__init__()

        self.comm = not sigmoid
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=kernel_size, stride=stride, groups=input_channels, bias = sigmoid)
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        size = (obs_shape[1] - (kernel_size - 1) - 1) // stride + 1
        self.fc_red = nn.Linear(in_features=size * size, out_features=output_size//3, bias = sigmoid)
        self.fc_green = nn.Linear(in_features=size * size, out_features=output_size//3, bias = sigmoid)
        self.fc_blue = nn.Linear(in_features=size * size, out_features=output_size//3, bias=sigmoid)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        #if self.comm:
            #x = self.prelu(x)
        #else:
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
        if self.comm:
            x = self.sig(cat((red_out, green_out, blue_out), dim=1))
        else:
            x = self.tanh(cat((red_out, green_out, blue_out), dim=1))
        return x

class SimpleCNN4(nn.Module):
    def __init__(self, obs_shape, output_size, use_orthogonal, use_ReLU, kernel_size=2, stride=1, input_channels=1, output_channels=1, sigmoid=True):
        super(SimpleCNN4, self).__init__()

        self.comm = not sigmoid
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=kernel_size, stride=stride, groups=input_channels, bias = sigmoid)
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        size = (obs_shape[1] - (kernel_size - 1) - 1) // stride + 1
        self.fc_red = nn.Linear(in_features=size * size, out_features=output_size//3, bias = sigmoid)
        self.fc_green = nn.Linear(in_features=size * size, out_features=output_size//3, bias = sigmoid)
        self.fc_blue = nn.Linear(in_features=size * size, out_features=output_size//3, bias=sigmoid)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # flatten
        x = x.view(x.size(0), -1)
        # chunk the 1D vector to separate the colors
        red, green, blue = chunk(x, 3, dim=1)
        # dense layers
        red_out = self.fc_red(red)
        green_out = self.fc_green(green)
        blue_out = self.fc_blue(blue)
        # Concatenation of the colors
        if self.comm:
            x = abs(cat((red_out, green_out, blue_out), dim=1))
            max_values = max(x, dim=1).values
            max_values = max_values.unsqueeze(1).repeat(1, 3)
            x = div(x, add(max_values, 0.00000001))
        else:
            x = self.tanh(cat((red_out, green_out, blue_out), dim=1))
        return x

class CNNLayer(nn.Module):
    def __init__(self, obs_shape, output_size, use_orthogonal, use_ReLU, kernel_size=2, stride=1, input_channels=1, output_channels=1, sigmoid=False):
        super(CNNLayer, self).__init__()


        #TODO: Maybe change the active_func
        active_func = nn.ReLU()
        if sigmoid:
            print("Sigmoid for comm")
            last_active_func = nn.Sequential(nn.Sigmoid(), nn.Threshold(0.5, 0))
        else:
            last_active_func = nn.ReLU()
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

        print("Stride is " + str(stride))
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
            last_active_func
        )

    def forward(self, x):
        if not self.multiple_channels:
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
       print("In the MergedModel initialization")
       super(MergedModel, self).__init__()
       self.experiment_name = mlp_args.experiment_name

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
           self.dim_actor = 1 + 6*mlp_args.grid_resolution
           if "step2.5" in self.experiment_name:
               self.dim_actor += mlp_args.grid_resolution
           sigmoid = True if mlp_args.sigmoid==1 else False
           if "goalcolor" in self.experiment_name:
               if self.agent_ID == 1 or obs_shape[0]/(self.dim_actor) > 1:
                   print("Movable agent or critic")
                   if "stepbystep" not in self.experiment_name:
                       if "step1" not in self.experiment_name:
                           if mlp_args.simpleCNN == 1:
                               print("Simple for comm")
                               self.cnn1 = SimpleCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), output_comm, mlp_args.use_orthogonal, mlp_args.use_ReLU, input_channels=3, output_channels=nb_output_channels, stride=mlp_args.stride_comm,kernel_size=mlp_args.kernel_comm, sigmoid=False)
                           elif mlp_args.simpleCNN2 == 1:
                               self.cnn1 = SimpleCNN2((mlp_args.grid_resolution, mlp_args.grid_resolution), output_comm, mlp_args.use_orthogonal, mlp_args.use_ReLU, input_channels=3, output_channels=nb_output_channels, stride=mlp_args.stride_comm,kernel_size=mlp_args.kernel_comm, sigmoid=False)
                           elif mlp_args.simpleCNN3 == 1:
                               self.cnn1 = SimpleCNN3((mlp_args.grid_resolution, mlp_args.grid_resolution), output_comm, mlp_args.use_orthogonal, mlp_args.use_ReLU, input_channels=3, output_channels=nb_output_channels, stride=mlp_args.stride_comm,kernel_size=mlp_args.kernel_comm, sigmoid=False)
                           elif mlp_args.simpleCNN4 == 1:
                               self.cnn1 = SimpleCNN4((mlp_args.grid_resolution, mlp_args.grid_resolution), output_comm, mlp_args.use_orthogonal, mlp_args.use_ReLU, input_channels=3, output_channels=nb_output_channels, stride=mlp_args.stride_comm,kernel_size=mlp_args.kernel_comm, sigmoid=False)
                           else:
                               self.cnn1 = CNNLayer((mlp_args.grid_resolution, mlp_args.grid_resolution), output_comm, mlp_args.use_orthogonal, mlp_args.use_ReLU, input_channels=3, output_channels=nb_output_channels, stride=mlp_args.stride_comm,kernel_size=mlp_args.kernel_comm, sigmoid=True)
                       if "step2" not in self.experiment_name:
                           if mlp_args.simpleCNN == 1:
                               print("Simple CNN")
                               self.cnn2 = SimpleCNN((mlp_args.grid_resolution, mlp_args.grid_resolution), 6, mlp_args.use_orthogonal, mlp_args.use_ReLU, input_channels=3, output_channels=nb_output_channels, stride=mlp_args.stride, kernel_size=mlp_args.kernel, sigmoid=True)
                           else:    
                               self.cnn2 = CNNLayer((mlp_args.grid_resolution, mlp_args.grid_resolution), 6, mlp_args.use_orthogonal, mlp_args.use_ReLU, input_channels=3, output_channels=nb_output_channels, stride=mlp_args.stride)
               else:
                   input_size = 3
           else:
               if not "goalcomm" in self.experiment_name:
                   print("Not goalcomm")
                   self.cnn1 = CNNLayer((mlp_args.grid_resolution, mlp_args.grid_resolution), output_comm, mlp_args.use_orthogonal, mlp_args.use_ReLU, input_channels=3, output_channels=nb_output_channels)
               self.cnn2 = CNNLayer((mlp_args.grid_resolution, mlp_args.grid_resolution), 6, mlp_args.use_orthogonal, mlp_args.use_ReLU, input_channels=3, output_channels=nb_output_channels)
       
       self.grid_resolution = mlp_args.grid_resolution
       self.nb_additional_data = mlp_args.nb_additional_data
       
       if obs_shape[0]/(self.dim_actor) > 1:
           if "goalcolor" in self.experiment_name:
               input_size = 6 + output_comm + 2 + 3
           else:
               input_size *= mlp_args.num_agents

       self.mlp = MLPLayer(input_size, mlp_args.hidden_size, mlp_args.layer_N, mlp_args.use_orthogonal, mlp_args.use_ReLU)

    def forward(self, x):
        # Séparer le tenseur en trois parties autant de fois que nécessaire
        x_inter_list = []

        if "step2.5" in self.experiment_name:
            if x.size()[1]//(self.dim_actor) > 1:
                tensor2 = x[:, 1:3*self.grid_resolution+1, :]
                reshaped_tensor = tensor2.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                result = reshaped_tensor[:, :, 0, 0]
                goal_color = result.reshape(-1, 3)

                tensor1 = x[:, self.dim_actor:1+self.dim_actor, :]
                result = tensor1[:, :, :2]
                velocity = result.reshape(-1, 2)

                tensor2 = x[:, 1+self.dim_actor:4*self.grid_resolution+1+self.dim_actor, :]
                reshaped_tensor = tensor2.reshape(-1, 4, self.grid_resolution, self.grid_resolution)
                x2 = self.cnn1(reshaped_tensor)

                tensor3 = x[:, 4*self.grid_resolution+1+self.dim_actor:2*self.dim_actor, :]
                reshaped_tensor = tensor3.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                result = reshaped_tensor[:, :, 0, :2]
                x3 = result.reshape(-1, 6)

                x_inter = cat((goal_color, velocity, x2, x3), dim=1)
                x_inter_list.append(x_inter)
            else:
                if self.agent_ID == 1:
                    tensor1 = x[:, 0:1, :]
                    result = tensor1[:, :, :2]
                    velocity = result.reshape(result.shape[0], 2)

                    tensor2 = x[:, 1:4*self.grid_resolution+1, :]
                    reshaped_tensor = tensor2.reshape(-1, 4, self.grid_resolution, self.grid_resolution)
                    x2 = self.cnn1(reshaped_tensor)

                    tensor3 = x[:, 4*self.grid_resolution+1:1*self.dim_actor, :]
                    reshaped_tensor = tensor3.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                    result = reshaped_tensor[:, :, 0, :2]
                    x3 = result.reshape(-1, 6)

                    x_inter = cat((velocity, x2, x3), dim=1)
                    x_inter_list.append(x_inter)
                else:
                    tensor2 = x[:, 1:3*self.grid_resolution+1, :]
                    reshaped_tensor = tensor2.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                    result = reshaped_tensor[:, :, 0, 0]
                    goal_color = result.reshape(-1, 3)

                    x_inter = goal_color
                    x_inter_list.append(x_inter)

        elif "step" in self.experiment_name:
            if x.size()[1]//(self.dim_actor) > 1:
                tensor2 = x[:, 1:3*self.grid_resolution+1, :]
                reshaped_tensor = tensor2.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                result = reshaped_tensor[:, :, 0, 0]
                goal_color = result.reshape(-1, 3)

                tensor1 = x[:, self.dim_actor:1+self.dim_actor, :]
                result = tensor1[:, :, :2]
                velocity = result.reshape(-1, 2)

                tensor2 = x[:, 1+self.dim_actor:3*self.grid_resolution+1+self.dim_actor, :]
                reshaped_tensor = tensor2.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                if "step2" not in self.experiment_name:
                    result = reshaped_tensor[:, :, 0, 0]
                    x2 = result.reshape(-1, 3)
                else:
                    x2 = self.cnn1(reshaped_tensor)

                tensor3 = x[:, 3*self.grid_resolution+1+self.dim_actor:2*self.dim_actor, :]
                reshaped_tensor = tensor3.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                if "step1" not in self.experiment_name:
                    result = reshaped_tensor[:, :, 0, :2]
                    x3 = result.reshape(-1, 6)
                else:
                    x3 = self.cnn2(reshaped_tensor)

                x_inter = cat((goal_color, velocity, x2, x3), dim=1)
                x_inter_list.append(x_inter)
            else:
                if self.agent_ID == 1:
                    tensor1 = x[:, 0:1, :]
                    result = tensor1[:, :, :2]
                    velocity = result.reshape(result.shape[0], 2)

                    tensor2 = x[:, 1:3*self.grid_resolution+1, :]
                    reshaped_tensor = tensor2.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                    if "step2" not in self.experiment_name:
                        result = reshaped_tensor[:, :, 0, 0]
                        x2 = result.reshape(-1, 3)
                    else:
                        x2 = self.cnn1(reshaped_tensor)

                    tensor3 = x[:, 3*self.grid_resolution+1:1*self.dim_actor, :]
                    reshaped_tensor = tensor3.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                    if "step1" not in self.experiment_name:
                        result = reshaped_tensor[:, :, 0, :2]
                        x3 = result.reshape(-1, 6)
                    else:
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
        
        elif "goalcolor" in self.experiment_name:
            if x.size()[1]//(self.dim_actor) > 1:
                tensor2 = x[:, 1:3*self.grid_resolution+1, :]
                reshaped_tensor = tensor2.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                result = reshaped_tensor[:, :, 0, 0]
                goal_color = result.reshape(-1, 3)

                tensor1 = x[:, self.dim_actor:1+self.dim_actor, :]
                result = tensor1[:, :, :2]
                velocity = result.reshape(-1, 2)
    
                tensor2 = x[:, 1+self.dim_actor:3*self.grid_resolution+1+self.dim_actor, :]
                reshaped_tensor = tensor2.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                x2 = self.cnn1(reshaped_tensor)
    
                tensor3 = x[:, 3*self.grid_resolution+1+self.dim_actor:2*self.dim_actor, :]
                reshaped_tensor = tensor3.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                x3 = self.cnn2(reshaped_tensor)
    
                x_inter = cat((goal_color, velocity, x2, x3), dim=1)
                x_inter_list.append(x_inter)
            else:
                if self.agent_ID == 1:
                    tensor1 = x[:, 0:1, :]
                    result = tensor1[:, :, :2]
                    velocity = result.reshape(result.shape[0], 2)
    
                    tensor2 = x[:, 1:3*self.grid_resolution+1, :]
                    reshaped_tensor = tensor2.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                    x2 = self.cnn1(reshaped_tensor)
    
                    tensor3 = x[:, 3*self.grid_resolution+1:1*self.dim_actor, :]
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
                elif "goalcolor" in self.experiment_name:
                    if x.size()[1]//(self.dim_actor) > 1:
                        print("test shared observations")
                        print(x[0])
                    if self.agent_ID == 1:
                        tensor1 = x[:, i*self.dim_actor:1+i*self.dim_actor, :]
                        result = tensor1[:, :, :2]
                        velocity = result.reshape(result.shape[0], 2)
    
                        tensor2 = x[:, 1+i*self.dim_actor:3*self.grid_resolution+1+i*self.dim_actor, :]
                        reshaped_tensor = tensor2.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                        x2 = self.cnn1(reshaped_tensor)
    
                        tensor3 = x[:, 3*self.grid_resolution+1+i*self.dim_actor:(i+1)*self.dim_actor, :]
                        reshaped_tensor = tensor3.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                        x3 = self.cnn2(reshaped_tensor)
    
                        x_inter = cat((velocity, x2, x3), dim=1)
                        x_inter_list.append(x_inter)
                    else:
                        tensor2 = x[:, 1+i*self.dim_actor:3*self.grid_resolution+1+i*self.dim_actor, :]
                        reshaped_tensor = tensor2.reshape(-1, 3, self.grid_resolution, self.grid_resolution)
                        result = reshaped_tensor[:, :, 0, 0]
                        goal_color = result.reshape(-1, 3)
    
                        x_inter = goal_color
                        x_inter_list.append(x_inter)
                elif "goalcomm" in self.experiment_name:
                    tensor1 = x[:, i*self.dim_actor:1+i*self.dim_actor, :]
                    result = tensor1[:, :, :2]
                    velocity = result.reshape(result.shape[0], 2)
    
                    tensor2 = x[:, 1+i*self.dim_actor:97+i*self.dim_actor, :]
                    reshaped_tensor = tensor2.reshape(-1, 3, 32, 32)
                    result = reshaped_tensor[:, :, 0, 0]
                    goal_color = result.reshape(-1, 3)
    
                    tensor3 = x[:, 97+i*self.dim_actor:(i+1)*self.dim_actor, :]
                    reshaped_tensor = tensor3.reshape(-1, 3, 32, 32)
                    x3 = self.cnn2(reshaped_tensor)
    
                    x_inter = cat((velocity, goal_color, x3), dim=1)
                    x_inter_list.append(x_inter)
                elif "speaker" in self.experiment_name:
                    tensor1 = x[:, i*self.dim_actor:1+i*self.dim_actor, :]
                    result = tensor1[:, :, :2]
                    velocity = result.reshape(result.shape[0], 2)
    
                    tensor2 = x[:, 1+i*self.dim_actor:97+i*self.dim_actor, :]
                    reshaped_tensor = tensor2.reshape(-1, 3, 32, 32)
                    x2 = self.cnn1(reshaped_tensor)
    
                    tensor3 = x[:, 97+i*self.dim_actor:(i+1)*self.dim_actor, :]
                    reshaped_tensor = tensor3.reshape(-1, 3, 32, 32)
                    x3 = self.cnn2(reshaped_tensor)
    
                    x_inter = cat((velocity, x2, x3), dim=1)
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
        x = self.mlp(x)
        return x
