import math
import torch
from torch.optim.optimizer import Optimizer


def centralized_gradient(x,use_gc=True,gc_conv_only=False):
    if use_gc:
      if gc_conv_only:
        if len(list(x.size()))>3:
            x.add_(-x.mean(dim = tuple(range(1,len(list(x.size())))), keepdim = True))
      else:
        if len(list(x.size()))>1:
            x.add_(-x.mean(dim = tuple(range(1,len(list(x.size())))), keepdim = True))
    return x


class ACClip(Optimizer):
    r"""Implements AdamP algorithm.

    It has been proposed in `Slowing Down the Weight Norm Increase in
    Momentum-based Optimizers`__

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        delta: threhold that determines whether a set of parameters is scale
            invariant or not (default: 0.1)
        wd_ratio: relative weight decay applied on scale-invariant parameters
            compared to that applied on scale-variant parameters (default: 0.1)
        nesterov: enables Nesterov momentum (default: False)


    Note:
        No Reference code
        Cosine similarity from AdamP
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), eps=1e-5, weight_decay=1e-5, fixed_decay=False,
                 clip_grad_norm=True, max_grad_norm=1.0, alpha=1.0, mod=1) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, alpha=alpha, mod=mod)
        self._fixed_decay = fixed_decay
        self._clip_grad_norm = clip_grad_norm
        self.max_grad_norm = max_grad_norm
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 1.0 <= alpha <= 2.0:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))
        super(ACClip, self).__init__(params, defaults)

    @staticmethod
    def _channel_view(x):
        return x.view(x.size(0), -1)

    @staticmethod
    def _layer_view(x):
        return x.view(1, -1)

    @staticmethod
    def _cosine_similarity(x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        x_norm = x.norm(dim=1).add_(eps)
        y_norm = y.norm(dim=1).add_(eps)
        dot = (x * y).sum(dim=1)

        return dot.abs() / x_norm / y_norm

    def step(self, closure=None):
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        combined_scale = 1.0
        if self._clip_grad_norm and self.max_grad_norm > 0:
            parameters = []
            for group in self.param_groups:
                parameters += [p for p in group['params'] if p.grad is not None]
            global_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
            if math.isfinite(global_grad_norm) and global_grad_norm > self.max_grad_norm:
                combined_scale = (global_grad_norm + 1e-6) / self.max_grad_norm

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if self._clip_grad_norm:
                    grad.div_(combined_scale)
                beta1, beta2 = group['betas']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # the clipping value, i.e., \tao_0^{\alpha}
                    state['tau'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Second-order momentum, v_t
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq, tau = state['exp_avg'], state['exp_avg_sq'], state['tau']
                alpha = group['alpha']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                tau.mul_(beta2).add_(grad.abs().pow(alpha), alpha=1 - beta2)  # alpha = 1
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # step_size = group['lr']
                step_size = group['lr'] / bias_correction1

                # truncate large gradient - ACClip
                denom = tau.pow(1 / alpha).div(exp_avg.abs().add(group['eps'])).clamp(min=0.0, max=1.0)

                # Adaptive Learning Rates : Work like Adam
                if group['mod'] == 1:
                    # denom.div_(exp_avg_sq.mul(beta2).sqrt().add(group['eps']))
                    denom.div_(exp_avg_sq.div(bias_correction2).mul(beta2).sqrt().add(group['eps']))

                update = denom * exp_avg

                # # gradient centralization
                # update = centralized_gradient(update, use_gc=True, gc_conv_only=False)

                # Update
                p.data.add_(update, alpha=-step_size)

                if not self._fixed_decay:
                    p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                else:
                    p.data.mul_(1.0 - group['weight_decay'])

        return loss

