import torch
import torch_geometric
from torch_scatter import scatter


class EGNNLayer(torch_geometric.nn.MessagePassing):
    def __init__(self, h_dim, e_dim, hid_dim):
        super().__init__()
        self.n_step, self.dist_min_max, self.gamma = 64, [0, 10], 10
        self.h_dim, self.e_dim = h_dim, e_dim 
        self.fc_m = torch.nn.Linear(h_dim * 2 + e_dim + self.n_step, hid_dim)
        self.fc_pos = torch.nn.Linear(hid_dim, 1)
        self.fc_x = torch.nn.Linear(h_dim + hid_dim, h_dim)

    def forward(self, x, pos, edge_index, edge_attr):
        m_ij = self._make_message(x, pos, edge_index, edge_attr)
        alpha_ij = self.fc_pos(m_ij)
        pos = self._update_pos(pos, edge_index, alpha_ij)
        x = self._update_x(x, edge_index, m_ij)
        return x, pos

    def _make_message(self, x, pos, edge_index, edge_atttr):
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]
        pos_i = x[edge_index[0]]
        pos_j = x[edge_index[1]]
        r_ij = (pos_i - pos_j).norm(dim=1)
        r_ij = self._dist_soft_one_hot(r_ij, self.n_step, self.dist_min_max, self.gamma)
        m_ij = torch.cat([x_i, x_j, r_ij, edge_attr], dim=1)
        m_ij = self.fc_m(m_ij)
        return m_ij

    def _get_attention_from_message(self, m_ij):
        alpha_ij = self.fc_pos(m_ij).squeeze()
        alpha_ij = torch.sigmoid(alpha_ij)
        return alpha_ij

    def _update_pos(self, pos, edge_index, alpha_ij):
        pos_vect = pos[edge_index[0]] - pos[edge_index[1]]
        pos = pos + scatter(pos_vect, edge_index[0], dim=0, dim_size=pos.size(0), reduce='mean')
        return pos

    def _update_x(self, x, edge_index, m_ij):
        m_aggr = scatter(m_ij, edge_index[0], dim=0, dim_size=x.size(0), reduce='sum')
        x = self.fc_x(torch.cat([x, m_aggr], dim=1))
        return x 

    def _dist_soft_one_hot(self, r_ij, n_step, dist_min_max, gamma): # [num_e] -> [num_e, step] as soft one hot
        dist_min, dist_max = dist_min_max
        c = torch.Tensor([dist_min * (n_step - i - 1) / (n_step - 1) + dist_max * i / (n_step - 1) for i in range(n_step)])
        c = c.unsqueeze(0)
        r_ij = r_ij.unsqueeze(1).repeat(1, n_step)
        r_ij = torch.exp(-gamma * torch.pow(r_ij - c, 2))
        return r_ij 


class EGNNBlock(torch.nn.Module):
    def __init__(self, h_dim, e_dim, hid_dim, n_layer, \
            activate_last=True, residual_connection='none'):
        super().__init__()
        assert residual_connection in ['none', 'add']
        self.activate_last = activate_last 
        self.residual_connection = residual_connection
        self.h_dim, self.e_dim, self.hid_dim = h_dim, e_dim, hid_dim 
        self.layer_list = torch.nn.ModuleList([EGNNLayer(h_dim, e_dim, hid_dim) for _ in range(n_layer)])

    def forward(self, x, pos, edge_idnex, edge_attr):
        x_ori, pos_ori = x, pos
        for layer in self.layer_list:
            x, pos = layer(x, pos, edge_index, edge_attr)
            x = torch.nn.functional.leaky_relu(x)
        if self.residual_connection == 'none':
            pass 
        elif self.residual_connection == 'add':
            x = x_ori + x
        if self.activate_last:
            x = torch.nn.functional.leaky_relu(x)
        return x, pos


class MPNNLayer(torch_geometric.nn.MessagePassing):
    def __init__(self, h_dim, e_dim):
        super().__init__(aggr='add')
        self.h_dim, self.e_dim = h_dim, e_dim
        self.fc_m = torch.nn.Linear(h_dim * 2 + e_dim, h_dim)
        # self.fc_alpha = torch.nn.Linear(h_dim, 1)

    def forward(self, x, edge_index, edge_attr): # x[N, h_dim]
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x

    def message(self, x_j, x_i, edge_attr):
        m_ij = torch.cat([x_j, x_i, edge_attr], dim=1)
        m_ij = self.fc_m(m_ij) # [e hd]
        return m_ij


class MPNNBlock(torch.nn.Module):
    def __init__(self, h_dim, e_dim, n_layer, \
            activate_last=True, residual_connection='none'):
        super().__init__()
        assert residual_connection in ['none', 'add'], 'residual connnection error'
        self.activate_last = activate_last
        self.residual_connection = residual_connection
        self.h_dim, self.e_dim = h_dim, e_dim
        self.layer_list = torch.nn.ModuleList([MPNNLayer(h_dim, e_dim) for _ in range(n_layer)])

    def forward(self, x, edge_index, edge_attr):
        x_ori = x
        for layer in self.layer_list:
            x = layer(x, edge_index, edge_attr)
            x = torch.nn.functional.leaky_relu(x)
        if self.residual_connection == 'none':
            pass 
        elif self.residual_connection == 'add':
            x = x + x_ori
        if self.activate_last:
            x = torch.nn.functional.leaky_relu(x)
        return x


if __name__ == '__main__':
    x = torch.rand(5, 11)
    pos = torch.rand(5, 3)
    edge_index = torch.Tensor([[0,0], [0, 1], [1, 0], [2, 3], [3, 2]]).long().t()
    edge_attr = torch.rand(5, 6)
    
    mpnn_layer = MPNNLayer(h_dim=11, e_dim=6)
    mpnn_layer(x, edge_index, edge_attr)

    egnn_layer = EGNNBlock(h_dim=11, e_dim=6, hid_dim=128, n_layer=3)
    x, pos = egnn_layer(x, pos, edge_index, edge_attr)
    print(pos.size())



