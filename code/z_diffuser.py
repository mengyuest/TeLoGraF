import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import utils
from collections import namedtuple
Sample = namedtuple('Sample', 'trajectories values chains')

from generate_scene_v1 import generate_trajectories, generate_trajectories_dubins
from generate_panda_scene import get_trajectories, panda_postproc

# TODO(Yue)
# https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        # return self.dropout(x)
        return x


#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

#-----------------------------------------------------------------------------#
#--------------------------------- attention ---------------------------------#
#-----------------------------------------------------------------------------#

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)

#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x


#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        
        return weighted_loss, {'a0_loss': a0_loss}

class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(),
                utils.to_np(targ).squeeze()
            )[0,1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
}


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5, dropout=None):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        if self.dropout is not None:
            out = self.dropout(self.blocks[0](x) + self.time_mlp(t))
            out = self.dropout(self.blocks[1](out))
            return out + self.residual_conv(x)
        else:
            out = self.blocks[0](x) + self.time_mlp(t)
            out = self.blocks[1](out)
            return out + self.residual_conv(x)

class MockNet(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
    ):
        super().__init__()
        self.mlp = nn.Linear(transition_dim, transition_dim)
    
    def forward(self, x, cond, time):
        '''
            x : [ batch x horizon x transition ]
        '''
        x = self.mlp(x)
        return x


class MLPNet(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
        dropout=None,
    ):
        super().__init__()
        
        time_dim = dim
        self.horizon = horizon
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        
        # TODO (yue)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        
        self.mlp = utils.build_relu_nn(horizon * transition_dim + dim, horizon * transition_dim, 
                                       hiddens=[xx*dim for xx in list(dim_mults)[1::-1]]+[xx*dim for xx in list(dim_mults)])
    
    def forward(self, x, cond, time):
        x = einops.rearrange(x, 'b h t -> b (h t)')

        t = self.time_mlp(time)
        
        # TODO(yue)
        tt = self.cond_mlp(cond)
        t = tt + t
        
        x = torch.cat([x, t], -1)

        x = self.mlp(x)

        x = einops.rearrange(x, 'b (h t) -> b h t', h=self.horizon)
        return x
        

class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
        dropout=None,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # TODO (yue)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon, dropout=dropout),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon, dropout=dropout),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon, dropout=dropout)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon, dropout=dropout)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon, dropout=dropout),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon, dropout=dropout),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        h = []
        
        # TODO(yue)
        tt = self.cond_mlp(cond)
        t = tt + t
        #

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x


class ValueFunction(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4
        ##
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down1 = Downsample1d(mid_dim_2)
        horizon = horizon // 2
        ##
        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down2 = Downsample1d(mid_dim_3)
        horizon = horizon // 2
        ##
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        ## mask out first conditioning timestep, since this is not sampled by the model
        # x[:, :, 0] = 0

        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        ##
        x = self.mid_block1(x, t)
        x = self.mid_down1(x)
        ##
        x = self.mid_block2(x, t)
        x = self.mid_down2(x)
        ##
        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out




class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class TransformerBackbone(nn.Module):
    def __init__(self, horizon, transition_dim, cond_dim, hidden_dim, num_heads, num_layers, args):
        super().__init__()

        dim = hidden_dim//4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, hidden_dim),
        )

        # TODO (yue)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, hidden_dim),
        )
        
        self.pos_enc = PositionalEncoding(hidden_dim)
        
        self.input_projection = nn.Linear(transition_dim, hidden_dim)
        
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=0.0),
            num_layers=num_layers
        )
        
        # Output layer to project back to the feature dimension
        self.output_projection = nn.Linear(hidden_dim, transition_dim)


    def forward(self, x, cond, time):
        '''
            x : [ batch x horizon x transition ]
        '''

        t = self.time_mlp(time)
                
        # TODO(yue)
        tt = self.cond_mlp(cond)
        
        ttt = (tt + t).unsqueeze(0)
        # print(t.shape, tt.shape, ttt.shape, x.shape)
        x = einops.rearrange(x, 'b t k -> t b k')
        # print(t.shape, tt.shape, ttt.shape, x.shape)
        
        # Project input features
        x = self.input_projection(x)
        
        x = self.pos_enc(x) + ttt

        
        # Pass through the Transformer encoder
        x = self.transformer_encoder(x)
        
        # Project back to the original feature dimension
        x = self.output_projection(x)
        
        # Permute back to (B, T, k)        
        x = einops.rearrange(x, 't b k -> b t k')
        return x


@torch.no_grad()
def default_sample_fn(model, x, cond, t, **sample_kwargs):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GradNN(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, encoder=None,
        clip_value_min=-3, clip_value_max=3, tokenizer=None,
        trans_model=None, transition_dim=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        if transition_dim is None:
            self.transition_dim = observation_dim + action_dim
        else:
            self.transition_dim = transition_dim
        self.model = model
        self.encoder = encoder
    
    def forward(self, cond):
        batch_size = cond.shape[0]
        x = torch.zeros(batch_size, self.horizon, self.transition_dim).to(cond.device)
        t = make_timesteps(batch_size, 1, cond.device)
        trajs = self.model(x, cond, t)
        values = torch.zeros(len(x), device=x.device)
        res = Sample(trajs, values, None)
        return res
    

class GaussianVAE(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, encoder=None,
        clip_value_min=-3, clip_value_max=3, tokenizer=None,
        trans_model=None, transition_dim=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        if transition_dim is None:
            self.transition_dim = observation_dim + action_dim
        else:
            self.transition_dim = transition_dim
        self.model = model
        self.encoder = encoder
        
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.clip_value_min = clip_value_min
        self.clip_value_max = clip_value_max
        self.predict_epsilon = predict_epsilon
        
        betas = cosine_beta_schedule(n_timesteps)
        self.register_buffer('betas', betas)
        
        self.tokenizer = tokenizer
        self.trans_model = trans_model
    
    def encode(self, x, cond):
        device = self.betas.device
        batch_size = x.shape[0]
        t = make_timesteps(batch_size, 1, device)
        return None
    
    def sample(self, x, cond):
        device = self.betas.device
        batch_size = x.shape[0]
        t = make_timesteps(batch_size, 100, device)
        return None


class GaussianFlow(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, encoder=None,
        clip_value_min=-3, clip_value_max=3, tokenizer=None,
        trans_model=None, transition_dim=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        if transition_dim is None:
            self.transition_dim = observation_dim + action_dim
        else:
            self.transition_dim = transition_dim
        self.model = model
        self.encoder = encoder
        
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.clip_value_min = clip_value_min
        self.clip_value_max = clip_value_max
        self.predict_epsilon = predict_epsilon
        
        betas = cosine_beta_schedule(n_timesteps)
        self.register_buffer('betas', betas)
        
        self.tokenizer = tokenizer
        self.trans_model = trans_model
    
    
    def get_flow_pattern(self, t_list):
        timesteps = len(t_list)-1
        total_len = 100
        dts = [(t_list[iiiii+1]-t_list[iiiii])/total_len for iiiii in range(timesteps)]
        return timesteps, dts
    
    def get_flow_pattern_new(self, t_list):
        timesteps = len(t_list)
        total_len = 100
        dts = [(t_list[iiiii+1]-t_list[iiiii])/total_len for iiiii in range(timesteps-1)]
        dts.insert(0, (t_list[0]-0)/total_len)
        return timesteps, dts
 
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        chain = [x] if return_chain else None
        
        if "in_painting" in sample_kwargs and sample_kwargs["in_painting"]:
            print("do in painting")
            do_nothing=False
        
        dts = None
        if "guidance_data" in sample_kwargs and sample_kwargs["guidance_data"] is not None:
            guidance_data = sample_kwargs["guidance_data"]
            args = guidance_data['args']
            denorm = guidance_data["denorm"]
            norm_func = guidance_data["norm_func"]
            loss_func = guidance_data["loss_func"]
            # dyna_func = guidance_data['dyna_func']
            real_stl_list = guidance_data["real_stl_list"]
            if args.env=="panda":
                CA = guidance_data["CA"]
            mini_batch_size = args.batch_size
            test_muls = args.test_muls if args.test_muls is not None else 1
            timesteps = self.n_timesteps
            t_list = list(range(0, self.n_timesteps))
            dt = 1 / self.n_timesteps
            
            if args.cls_guidance:
                score_predictor = guidance_data["score_predictor"]
                stl_embeds = guidance_data["stl_embeds"]
                if args.encoder == "goal":
                    stl_embeds_gnn = guidance_data["stl_embeds_gnn"]
            
        else:
            guidance_data = None
            args = sample_kwargs["args"]
            
            if args.flow_pattern is None:
                t_list = list(range(0, self.n_timesteps))
                timesteps = self.n_timesteps
                dt = 1 / self.n_timesteps
            elif args.flow_pattern==0:
                t_list = [0]*5+list(range(0, self.n_timesteps))
                timesteps = 5+self.n_timesteps
                dt = 1 / self.n_timesteps
            elif args.flow_pattern==-1:
                t_list = list(range(1, self.n_timesteps))
                timesteps = self.n_timesteps - 1
                dt = 1 / self.n_timesteps
            elif args.flow_pattern==1:
                t_list=[0,9,19,29,39,49,59,69,79,89,99]
                timesteps = self.n_timesteps//10
                dt = 1 / self.n_timesteps * 10
            elif args.flow_pattern==2:
                t_list=[0,9,19,29,39,49,99]
                timesteps = 6
                dts = [1 / self.n_timesteps * 10] * 5 + [1 / self.n_timesteps * 50]
            elif args.flow_pattern==3:
                t_list=[0,9,19,29,99]
                timesteps = 4
                dts = [1 / self.n_timesteps * 10] * 3 + [1 / self.n_timesteps * 70]
            elif args.flow_pattern==4:
                t_list=[0]+list(range(19,100,20))
                timesteps, dts = self.get_flow_pattern(t_list)
            elif args.flow_pattern==5:
                t_list=[0]+list(range(9,100,10))
                timesteps, dts = self.get_flow_pattern(t_list)
            elif args.flow_pattern==6:
                t_list=[0]+list(range(4,100,5))
                timesteps, dts = self.get_flow_pattern(t_list)
            elif args.flow_pattern==7:
                t_list=[0]+list(range(3,100,3))
                timesteps, dts = self.get_flow_pattern(t_list)
            elif args.flow_pattern==8:
                t_list=[0]+list(range(1,100,2))
                timesteps, dts = self.get_flow_pattern(t_list)
            elif args.flow_pattern==9:
                t_list=list(range(0,100))
                timesteps, dts = self.get_flow_pattern(t_list)
            elif args.flow_pattern==10:
                t_list=[0]+list(range(24,100,25))
                timesteps, dts = self.get_flow_pattern(t_list)
            elif args.flow_pattern==11:
                t_list=[0]+list(range(49,100,50))
                timesteps, dts = self.get_flow_pattern(t_list)
            elif args.flow_pattern==12:
                t_list=list(range(0,100,99))
                timesteps, dts = self.get_flow_pattern(t_list)
            elif args.flow_pattern==13:
                t_list=[99]
                timesteps = 1
                dt = 1.0
            elif args.flow_pattern==14:
                t_list=list(range(1,100,1))
                timesteps, dts = self.get_flow_pattern_new(t_list)
            elif args.flow_pattern==15:
                t_list=list(range(1,100,2))
                timesteps, dts = self.get_flow_pattern_new(t_list)
            elif args.flow_pattern==16:
                t_list=list(range(3,100,3))
                timesteps, dts = self.get_flow_pattern_new(t_list)
            elif args.flow_pattern==17:
                t_list=list(range(3,100,4))
                timesteps, dts = self.get_flow_pattern_new(t_list)
            elif args.flow_pattern==18:
                t_list=list(range(4,100,5))
                timesteps, dts = self.get_flow_pattern_new(t_list)
            elif args.flow_pattern==19:
                t_list=list(range(9,100,10))
                timesteps, dts = self.get_flow_pattern_new(t_list)
            elif args.flow_pattern==20:
                t_list=list(range(19,100,20))
                timesteps, dts = self.get_flow_pattern_new(t_list)
            elif args.flow_pattern==21:
                t_list=list(range(29,100,30))
                timesteps, dts = self.get_flow_pattern_new(t_list)
            elif args.flow_pattern==22:
                t_list=list(range(49,100,50))
                timesteps, dts = self.get_flow_pattern_new(t_list)
            elif args.flow_pattern==23:
                t_list=list(range(99,100,100))
                timesteps, dts = self.get_flow_pattern_new(t_list)
        
        POST_GUIDANCE=False
        
        for i in reversed(range(0, timesteps)):
            t = make_timesteps(batch_size, t_list[i], device)
            if args.flow_pattern is not None and dts is not None:
                dt = dts[i]
            if POST_GUIDANCE:
                with torch.set_grad_enabled(False):
                    dx = self.model(x, cond, t)
                    x = x + dx * dt
            
            grad_guidance = 0.0
            if guidance_data is not None:
                if args.guidance_before is None or i<args.guidance_before:
                    # first denorm, then compute loss, finally update 
                    with torch.set_grad_enabled(True):
                        x_detach = x.detach().requires_grad_(True)
                        for inner_i in range(args.guidance_steps):
                            x_real = denorm(x_detach)
                            # should use dynamics to generate the trajectory
                            if args.env=="simple":
                                xs = x_real[..., 0, :2].detach()
                                us = x_real[..., 2:]
                                trajs = generate_trajectories(xs, us, 0.5)
                                x_real_aug = torch.cat([trajs[..., :-1, :], us], dim=-1)
                                # x_real_aug_norm = norm_func(x_real_aug)
                            elif args.env=="dubins":
                                xs = x_real[..., 0, :4].detach()
                                us = x_real[..., 4:]
                                trajs = generate_trajectories_dubins(xs, us, 0.5, v_max=2.0,)
                                x_real_aug = torch.cat([trajs[..., :-1, :], us], dim=-1)
                                # x_real_aug_norm = norm_func(x_real_aug)
                            elif args.env=="panda":
                                xs = x_real[..., 0, :7].detach()
                                us = x_real[..., 7:]
                                trajs = get_trajectories(xs, us, dof=7, dt=0.05)
                                x_real_aug = trajs[..., :-1, :]
                            elif args.env in ["pointmaze", "antmaze"]:
                                do_nothing_here = True
                            else:
                                raise NotImplementedError
                                                        
                            if args.cls_guidance:
                                x_detach_flat = x_detach.reshape(x_detach.shape[0], -1)
                                # x_real_aug_flat = x_real_aug.reshape(x_real_aug.shape[0], -1)
                                if args.encoder=="goal":
                                    score_est = score_predictor.forward(None, stl_embeds_gnn, x_detach_flat)
                                else:
                                    score_est = score_predictor.forward(None, stl_embeds, x_detach_flat)
                                # print(score_est)
                                # exit()
                                loss = torch.mean(torch.nn.ReLU()(0.05 - score_est))
                                # print(loss)
                            else:
                                x_real_4d = x_real_aug.reshape(mini_batch_size, test_muls, x_real_aug.shape[-2], x_real_aug.shape[-1])
                                loss_list=[]
                                for ii in range(mini_batch_size):
                                    loss_v, dbg_info = loss_func(x_real_4d[ii], real_stl_list[ii])
                                    if args.env=="dubins":
                                        loss_reg = torch.mean(torch.nn.ReLU()(us**2-4)) * 10
                                        loss_v = loss_v + loss_reg
                                    loss_list.append(loss_v)
                            
                                loss_list = torch.stack(loss_list, dim=0)
                                loss = torch.mean(loss_list)
                            grad_x = torch.autograd.grad(loss, x_detach)[0]
                            x_detach = x_detach - grad_x * args.guidance_lr
                    
                    # TODO (what if only work on us, not change xs? will x_detach xs differetn than x?)
                    grad_guidance = (x_detach - x) * args.guidance_scale                    
            
                    # print("guidance=",torch.sum(torch.abs(grad_guidance)))
            
            if POST_GUIDANCE:
                x = x + grad_guidance    
            else:
                with torch.set_grad_enabled(False):
                    dx = self.model(x, cond, t)
                    x = x + dx * dt + grad_guidance
            
            
            if return_chain: chain.append(x)
        
        values = torch.zeros(len(x), device=x.device)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        batch_size = len(cond)
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        return self.p_sample_loop(shape, cond, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#
    
    '''
    x_1 = Tensor(make_moons(256, noise=0.05)[0])
    x_0 = torch.randn_like(x_1)
    t = torch.rand(len(x_1), 1)
    x_t = (1 - t) * x_0 + t * x_1
    dx_t = x_1 - x_0
    optimizer.zero_grad()
    loss_fn(flow(x_t, t), dx_t).backward()
    optimizer.step()
    '''
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        tt = ((t+1).float() / (self.n_timesteps))[:, None, None]
        # torch.Size([256, 64, 14]) torch.Size([256, 64, 14]) torch.Size([256])
        x_noisy = tt * noise + (1-tt) * x_start
        return x_noisy

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)
    
    

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, encoder=None,
        clip_value_min=-3, clip_value_max=3, tokenizer=None,
        trans_model=None, transition_dim=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        if transition_dim is None:
            self.transition_dim = observation_dim + action_dim
        else:
            self.transition_dim = transition_dim
        self.model = model
        self.encoder = encoder

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.clip_value_min = clip_value_min
        self.clip_value_max = clip_value_max
        self.predict_epsilon = predict_epsilon
        
        self.tokenizer = tokenizer
        self.trans_model = trans_model

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(self.clip_value_min, self.clip_value_max)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        chain = [x] if return_chain else None

        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            
            if "in_painting" in sample_kwargs and sample_kwargs["in_painting"]:
                print("do in painting")
                do_nothing=False
            
            model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
            model_std = torch.exp(0.5 * model_log_variance)

            # no noise when t == 0
            noise = torch.randn_like(x)
            noise[t == 0] = 0
            x = model_mean + model_std * noise           

            # progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        # progress.stamp()
        values = torch.zeros(len(x), device=x.device)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond)
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)