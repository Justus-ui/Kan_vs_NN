import torch
import torch.nn as nn
from Lipschitz_Linear import Lipschitz_Linear

class KAN_Layer(nn.Module):
    """
        Defines KAN with univariate approximation via GRUs
    """
    def __init__(self, in_dim, out_dim, h, device = None):
        """ 
        in_dim:Dimension of Agent information, i.e cartesian coordinates R^2
        device: if true uses cuda
        """
        super(KAN_Layer, self).__init__()
        # Problem Attributes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h = h

        self.device_train = torch.device("cpu") ## if device marker is set --> cuda
        if device:
            assert torch.cuda.is_available()
            self.device_train = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device_eval = torch.device("cpu")
        self.device_eval = torch.device("cpu")
        self.Network_stack = nn.ModuleList() ## Read linear_Network_stack["to Neuron"]["from Neuron"]
        self.linear_Network_stack = nn.ModuleList()
        self.activation = nn.ReLU()
        # Constraint
        
        #init model
        self.init_layers()

    def train(self, mode = True):
        super(KAN_Layer, self).train(mode)
        self.to_device()

    def eval(self):
        super(KAN_Layer, self).eval()
        self.to_device()

    def to_device(self):
        """
            Loads modules to device
        """
        device = self.device_train if self.training else self.device_eval
        #print(f"loaded to device {device}")
        self.to(device)
    
    def init_layers(self):
        """
            defines the univariate grus if we estimate f: R^ -> R
        """
        Networks = nn.ModuleList()
        for _ in range(self.out_dim):
            for _ in range(self.in_dim):
                #print([1] + self.h)
                Networks.append(Lipschitz_Linear([1] + self.h))
            self.Network_stack.append(Networks)
            self.linear_Network_stack.append(Lipschitz_Linear([self.h[-1] * self.in_dim, 1]))
        

    def forward(self, x):
        """
        When Networktype is uni
        x: Inital States [Batch_size, in_dim]
        return
        x: Inital States [Batch_size, out_dim]
        """
        outs = torch.zeros(x.shape[0], self.out_dim)
        for i in range(self.out_dim):
            output_list = []
            for j in range(self.in_dim):
                output_list.append(self.Network_stack[i][j](x[:,j].unsqueeze(1)))
            outs[:,i] = self.linear_Network_stack[i](self.activation(torch.cat(output_list, dim=1))).flatten()
        return outs
    
class Neural_Kan(nn.Module):
    """
    Class:
    shape: list, describing tuple (n_1,...,n_N)
    h: shape of univariate Neural Networks. 
    """
    def __init__(self, shape, h, device = None):
        super(Neural_Kan, self).__init__()
        self.layers = nn.Sequential()
        for i in range(len(shape) - 1):
            #print(shape[i], shape[i + 1])
            self.layers.append(KAN_Layer(in_dim = shape[i], out_dim = shape[i + 1], h = h, device = device))

    def forward(self,x):
        return self.layers(x)
    
if __name__ == '__main__':
    model = Neural_Kan(shape = [5,4,3], h = [8,16])
    print(model(torch.randn(100,5)).shape)

