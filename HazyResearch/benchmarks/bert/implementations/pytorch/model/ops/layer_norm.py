# Adapted from https://github.com/NVIDIA/apex/blob/master/apex/contrib/layer_norm/layer_norm.py
import torch
from torch.nn import init

from apex._autocast_utils import _cast_if_autocast_enabled
import dropout_layer_norm


def _dropout_add_layer_norm_forward(x0, x1, gamma, beta, dropout_p, epsilon):
    """ Assume that arguments are contiguous
    """
    hidden_size = gamma.numel()
    x0mat = x0.view((-1, hidden_size))
    x1mat = x1.view((-1, hidden_size))
    zmat, xmat, dmask, mu, rsigma = dropout_layer_norm.dropout_add_ln_fwd(
        x0mat, x1mat, gamma, beta, dropout_p, epsilon, None
    )
    return zmat, xmat, dmask, mu, rsigma


def _dropout_add_layer_norm_backward(dz, x, dmask, mu, rsigma, gamma, dropout_p):
    """ Assume that arguments are contiguous
    """
    hidden_size = gamma.numel()
    xmat = x.view((-1, hidden_size))
    dzmat = dz.view(xmat.shape)
    dx0mat, dx1mat, dgamma, dbeta, _, _ = dropout_layer_norm.dropout_add_ln_bwd(
        dzmat, xmat, dmask, mu, rsigma, gamma, dropout_p
    )
    return dx0mat, dx1mat, dgamma, dbeta


class DropoutAddLayerNormFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, x1, gamma, beta, dropout_p, epsilon):
        x0 = x0.contiguous()
        x1 = x1.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        zmat, xmat, dmask, mu, rsigma = _dropout_add_layer_norm_forward(x0, x1, gamma, beta,
                                                                       dropout_p, epsilon)
        ctx.save_for_backward(xmat.view(x0.shape), dmask, gamma, mu, rsigma)
        ctx.dropout_p = dropout_p
        return zmat.view(x0.shape)

    @staticmethod
    def backward(ctx, dz):
        # assert dz.is_contiguous()
        dz = dz.contiguous()  # this happens!
        x, dmask, gamma, mu, rsigma = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        dx0mat, dx1mat, dgamma, dbeta = _dropout_add_layer_norm_backward(dz, x, dmask, mu, rsigma,
                                                                         gamma, dropout_p)
        dx0 = dx0mat.view(x.shape)
        dx1 = dx1mat.view(x.shape)
        return dx0, dx1, dgamma, dbeta, None, None


# We duplicate code to return both the output and the dropout mask for testing.
# Returning both makes backward a bit slower, so we want to keep using the other version for speed.
class DropoutAddLayerNormDmaskFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, x1, gamma, beta, dropout_p, epsilon):
        x0 = x0.contiguous()
        x1 = x1.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        zmat, xmat, dmask, mu, rsigma = _dropout_add_layer_norm_forward(x0, x1, gamma, beta,
                                                                       dropout_p, epsilon)
        ctx.save_for_backward(xmat.view(x0.shape), dmask, gamma, mu, rsigma)
        ctx.dropout_p = dropout_p
        dmask = dmask.view(x0.shape)
        ctx.mark_non_differentiable(dmask)
        return zmat.view(x0.shape), dmask

    @staticmethod
    def backward(ctx, dz, ddmask_ignored_):
        # assert dz.is_contiguous()
        dz = dz.contiguous()  # this happens!
        x, dmask, gamma, mu, rsigma = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        dx0mat, dx1mat, dgamma, dbeta = _dropout_add_layer_norm_backward(dz, x, dmask, mu, rsigma,
                                                                         gamma, dropout_p)
        dx0 = dx0mat.view(x.shape)
        dx1 = dx1mat.view(x.shape)
        return dx0, dx1, dgamma, dbeta, None, None


def dropout_add_layer_norm(x0, x1, weight, bias, dropout_p, epsilon, return_dropout_mask=False):
    args = _cast_if_autocast_enabled(x0, x1, weight, bias, dropout_p, epsilon)
    with torch.cuda.amp.autocast(enabled=False):
        return (DropoutAddLayerNormFN.apply(*args) if not return_dropout_mask
                else DropoutAddLayerNormDmaskFN.apply(*args))


class DropoutAddLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, p=0.5, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.p = p
        self.epsilon = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x0, x1):
        return dropout_add_layer_norm(x0, x1, self.weight, self.bias,
                                       self.p if self.training else 0.0, self.epsilon)
