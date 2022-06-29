import torch
import mlp_cuda
from torch import nn
from apex import amp

#implements fused GEMM+bias in forward pass using mlp_cuda from apex
class FusedMlpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight)
        output = mlp_cuda.forward(True, 0, (input, weight, bias))
        return output[0]

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)
        grad_bias = torch.sum(grad_output, dim=0)
        return grad_input, grad_weight, grad_bias

mlp_function = amp.half_function(FusedMlpFunction.apply)

class FusedMlp(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(FusedMlp, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return mlp_function(input, self.weight, self.bias)

