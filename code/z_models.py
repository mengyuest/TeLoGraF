import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv
from torch_geometric.utils import scatter
import utils
import math

class ChildSumLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='sum')
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        return x_j

# https://github.com/pyg-team/pytorch_geometric/issues/121
# https://arxiv.org/pdf/1503.00075
# Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
class TreeLSTMLayer(MessagePassing):
    def __init__(self, x_dim, hidden_dim):
        super().__init__(aggr='add')
        # TODO two types, childsum or nary
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        output_dim = hidden_dim
        self.child_sum = ChildSumLayer(in_channels=hidden_dim, out_channels=hidden_dim)
        
        # input repr
        self.Wi = torch.nn.Linear(x_dim, output_dim)
        self.Ui = torch.nn.Linear(hidden_dim, output_dim)
        
        # forget gate
        self.Wf = torch.nn.Linear(x_dim, output_dim)
        self.Uf = torch.nn.Linear(hidden_dim, output_dim)
        
        # output repr
        self.Wo = torch.nn.Linear(x_dim, output_dim)
        self.Uo = torch.nn.Linear(hidden_dim, output_dim)
        
        # control how much from input 
        self.Wu = torch.nn.Linear(x_dim, output_dim)
        self.Uu = torch.nn.Linear(hidden_dim, output_dim)
        
        self.activation = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
    
    def forward(self, x, edge_index, hidden=None, cell=None, orders=None):
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_dim).to(x.device)
            
        if cell is None:
            cell = torch.zeros(x.shape[0], self.hidden_dim).to(x.device)
        
        h_tilde = self.child_sum(hidden, edge_index)
        i_data = self.activation(self.Wi(x) + self.Ui(h_tilde))
        o_data = self.activation(self.Wo(x) + self.Uo(h_tilde))
        u_data = self.tanh(self.Wu(x) + self.Uu(h_tilde))
        
        # f_jk = sigma(Wf x_j + Uf h_k + bf) 
        # c_j = i_j \odot u_j + \sum\limits_{k\in C(j)} f_jk \odot c_k
        c_final = i_data * u_data
        for order in orders:
            mask = order[edge_index[0]].bool()
            if edge_index[:, mask].shape[1]!=0:
                c_other_partial = self.propagate(edge_index[:, mask], x=x, h=hidden, c=cell)
                c_final = c_final + c_other_partial
        h_final = o_data * self.tanh(c_final)
        return h_final, c_final

    def message(self, x_i, x_j, h_i, h_j, c_i, c_j):  
        f_ij = self.activation(self.Wf(x_i) + self.Uf(h_j))
        output = f_ij * c_j
        return output
    
class TreeLSTMNet(torch.nn.Module):
    def __init__(self, x_dim, ego_state_dim, hidden_dim, hidden_layers, args):
        super(TreeLSTMNet, self).__init__()
        self.args = args
        self.layers = []
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        for i in range(hidden_layers):
            self.layers.append(TreeLSTMLayer(x_dim=x_dim, hidden_dim=hidden_dim))
        self.layers = torch.nn.ModuleList(self.layers)
        
        self.hid_proj = nn.Linear(hidden_dim, args.condition_dim)
        self.ego_state_dim = ego_state_dim
        self.mlp = utils.build_relu_nn(input_dim=ego_state_dim, output_dim=ego_state_dim, hiddens=args.mlp_hiddens)
    
    def forward(self, ego_states, data):
        x, edge_index, depths = data.x, data.edge_index, data.depths
        max_layer_n = int(torch.max(depths).item())
        orders = [(depths==max_layer_n-i).long() for i in range(max_layer_n+1)]
        hidden = torch.zeros(x.shape[0], self.hidden_dim).to(x.device)
        cell = torch.zeros(x.shape[0], self.hidden_dim).to(x.device)
                
        for i in range(self.hidden_layers):
            hidden, cell = self.layers[i](x, edge_index, hidden=hidden, cell=cell, orders=orders)
        
        hidden_graph = scatter(hidden, data.batch, dim=0 ,reduce="mean")
        stl_feat = self.hid_proj(hidden_graph)
        if ego_states is None:
            return stl_feat
        else:
            ego_feat = self.mlp(ego_states)
            conditions = torch.cat([stl_feat, ego_feat], dim=-1)
            return conditions


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, raw_input_dim, ego_state_dim, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, args=None):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers, batch_first=True)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.ninp = ninp
        self.args = args
        self.ego_state_dim = ego_state_dim
        self.raw_input_dim = raw_input_dim
        self.input_proj = nn.Linear(raw_input_dim, ninp)
        self.hid_proj = nn.Linear(nhid, self.args.condition_dim)
        self.mlp = utils.build_relu_nn(input_dim=ego_state_dim, output_dim=ego_state_dim, hiddens=self.args.mlp_hiddens)
        # self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, ego_states, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        if self.raw_input_dim==1:
            src = self.input_proj(src[:, :, None])
        else:
            src = self.input_proj(src)

        src = self.pos_encoder(src)       
        
        output = self.encoder(src, mask=self.src_mask)
        
        stl_feat = self.hid_proj(output[:, -1, :])
        if ego_states is None:
            return stl_feat
        else:
            ego_feat = self.mlp(ego_states)
            conditions = torch.cat([stl_feat, ego_feat], dim=-1)
            return conditions


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(MLP, self).__init__()
        self.args = args
        self.mlp = utils.build_relu_nn(input_dim=input_dim, output_dim=output_dim+input_dim, hiddens=args.mlp_hiddens)

    def forward(self, x):
        return self.mlp(x)
    
# Define the MLP for GINConv
class GINMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GINMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class GCN(torch.nn.Module):
    def __init__(self, input_dim, ego_state_dim, args):
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        self.ego_state_dim = ego_state_dim
        self.output_dim = args.condition_dim  # gnn_feat_dim + ego_state_dim
        _conv_list = []
        all_dims = [self.input_dim] + args.hiddens + [self.output_dim]
        self.lin = []
        for layer_in_dim, layer_out_dim in zip(all_dims[:-1], all_dims[1:]):
            if args.gat:
                _conv_list.append(GATv2Conv(layer_in_dim, layer_out_dim))
            elif args.gin_conv:
                _conv_list.append(GINConv(GINMLP(layer_in_dim, layer_out_dim, layer_out_dim)))
            else:
                if args.gcn_no_self_loops:
                    _conv_list.append(GCNConv(layer_in_dim, layer_out_dim, add_self_loops=False))
                else:
                    _conv_list.append(GCNConv(layer_in_dim, layer_out_dim))
            if args.residual:
                if layer_in_dim==layer_out_dim:
                    self.lin.append(nn.Identity())
                else:
                    self.lin.append(nn.Linear(layer_in_dim, layer_out_dim))
        if args.residual:
            self.lin = torch.nn.ModuleList(self.lin)
        self.conv_list = torch.nn.ModuleList(_conv_list)
        self.mlp = utils.build_relu_nn(input_dim=self.ego_state_dim, output_dim=self.ego_state_dim, hiddens=args.mlp_hiddens)

        if args.with_predict_head:
            self.predict_head = nn.Linear(args.condition_dim, 4)
        
        if args.predict_score:
            n_dim = 64
            self.predict_score_head = utils.build_relu_nn(input_dim=args.horizon * 2, output_dim=args.condition_dim, hiddens=[n_dim])
        
    def predict(self, embedding):
        logits = self.predict_head(embedding)
        return logits
        
    def forward(self, ego_states, data):
        x, edge_index = data.x, data.edge_index
        for layer_i in range(len(self.conv_list)):
            if self.args.residual:
                identity = self.lin[layer_i](x)
            x = self.conv_list[layer_i](x, edge_index)
            if self.args.residual and self.args.post_residual==False:
                x = x + identity
            if layer_i != len(self.conv_list)-1:
                x = F.relu(x)
            if self.args.residual and self.args.post_residual==True:
                x = x + identity

        if self.args.aggr_type==0:
            x = scatter(x, data.batch, dim=0 ,reduce="mean")
        elif self.args.aggr_type==1:
            x = scatter(x, data.batch, dim=0 ,reduce="max")
        elif self.args.aggr_type==2:
            x0 = scatter(x, data.batch, dim=0 ,reduce="mean")
            x2 = scatter(x, data.batch, dim=0 ,reduce="max")
            x = (x0+x2)/2
        elif self.args.aggr_type==3:
            x0 = scatter(x, data.batch, dim=0 ,reduce="mean")
            x1 = scatter(x, data.batch, dim=0 ,reduce="min")
            x2 = scatter(x, data.batch, dim=0 ,reduce="max")
            x = (x0+x1+x2)/3
        elif self.args.aggr_type==4:
            root_indices = torch.where(data.batch[:-1]!=data.batch[1:])[0]+1
            root_indices = F.pad(root_indices, [1,0], mode='constant', value=0)
            x = x[root_indices]

        if ego_states is None:
            return x
        else:
            ego_feat = self.mlp(ego_states)
            x = torch.cat([x, ego_feat], dim=-1)
            return x


class GRUEncoder(torch.nn.Module):
    def __init__(self, input_dim, ego_state_dim, hidden_dim, feature_dim, num_layers=1, bidirectional=False, args=None):
        super(GRUEncoder, self).__init__()
        self.gru = torch.nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.hidden_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = torch.nn.Linear(self.hidden_dim, feature_dim)  # Projection to feature_dim
        self.args = args
        self.ego_state_dim = ego_state_dim
        self.mlp = utils.build_relu_nn(input_dim=ego_state_dim, output_dim=ego_state_dim, hiddens=args.mlp_hiddens)
    
    def forward(self, ego_states, x):
        """
            x: Input tensor of shape [N, T, k].
            Encoded features of shape [N, m].
        """
        if self.args.bfs_encoding or self.args.smart_encoding or self.args.hashimoto:
            x_unsqu = x
        else:
            x_unsqu = x[:, :, None]
        _, hidden = self.gru(x_unsqu)  # hidden: [num_layers * num_directions, N, h]
        hidden = hidden[-1]  # [N, h]
        features = self.fc(hidden)  # [N, m]
        if ego_states is None:
            return features
        else:
            ego_feat = self.mlp(ego_states)
            features = torch.cat([features, ego_feat], dim=-1)
            return features


class ScorePredictor(torch.nn.Module):
    def __init__(self, encoder, feat_dim, nt, state_dim, args):
        super(ScorePredictor, self).__init__()
        self.encoder = encoder
        self.args = args
        self.state_dim = state_dim
        self.nt = nt
        self.traj_mlp = utils.build_relu_nn(input_dim=state_dim*nt, output_dim=feat_dim, hiddens=args.traj_hiddens)
        self.score_mlp = utils.build_relu_nn(input_dim=feat_dim, output_dim=1, hiddens=args.score_hiddens)
    
    def get_stl_embedding(self, stl_data):
        return self.encoder(None, stl_data)
    
    def get_traj_embedding(self, trajs):
        return self.traj_mlp(trajs)
    
    def pred_score(self, stl_feat, traj_feat):
        fuse_feat = stl_feat + traj_feat
        score = self.score_mlp(fuse_feat).squeeze(-1)
        return score
    
    def forward(self, ego_states, stl_data, trajs):
        stl_feat = self.encoder(None, stl_data)  # (BS, *)
        
        traj_feat = self.traj_mlp(trajs)  # (BS, *)
        fuse_feat = stl_feat + traj_feat
        score = self.score_mlp(fuse_feat).squeeze(-1)
        return score # (BS, *)
    
    def dual_forward(self, ego_states, stl_data, trajs, mini_batch, stl_feat=None):
        if stl_feat is None:
            stl_feat = self.encoder(None, stl_data)  # (BS, *)
        
        traj_feat = self.traj_mlp(trajs)  # (2*BS, *)
        BS = stl_feat.shape[0]
        doub_traj_feat = traj_feat.reshape(2,BS,traj_feat.shape[-1])[:,:mini_batch]
        doub_stl_feat = torch.stack([stl_feat[:mini_batch], stl_feat[:mini_batch]], dim=0)
        fuse_feat = doub_stl_feat + doub_traj_feat
        score = self.score_mlp(fuse_feat).squeeze(-1)
        return score