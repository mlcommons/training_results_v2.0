# Adapted from https://github.com/NVIDIA/apex/blob/master/apex/fused_dense/fused_dense.py
# On the backward pass, we don't use the fused kernel from cublasLt since that's a bit slower.
# Instead we use the regular backward from F.linear.
# We also make it work with pytorch amp.
# TD [2022-02-27] The fused backward is also less accurate, and it might silently fail to compute
# grad_bias (when it takes the cublas gemm code path instead of the cublasLt code path)
# TD [2022-04-19] The fused backward seems fine now. Also fused dense gelu dense works.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

# import fused_dense_cuda  # from apex
import fused_dense_lib as fused_dense_cuda
# from src.ops.triton.triton_matmul import matmul_dgelu


# implements fused GEMM+bias in forward pass using mlp_cuda from apex
class FusedDenseFuncTD(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        output = fused_dense_cuda.linear_bias_forward(x.reshape(batch_dim, n), weight, bias)
        return output.reshape(*batch_shape, output.shape[-1])

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        grad_input, grad_weight, grad_bias = fused_dense_cuda.linear_bias_backward(
            x.reshape(batch_dim, n), weight, grad_output.reshape(batch_dim, grad_output.shape[-1])
        )
        # print((grad_bias - grad_output.view(-1, grad_output.shape[-1]).sum(dim=0)).abs().max())
        return grad_input.reshape_as(x), grad_weight, grad_bias
        # grad_input, grad_weight = None, None
        # grad_output_reshaped = grad_output.reshape(batch_dim, grad_output.shape[-1])
        # if ctx.needs_input_grad[0]:
        #     grad_input = (grad_output_reshaped @ weight.conj()).reshape(*batch_shape, n)
        # if ctx.needs_input_grad[1]:
        #     grad_weight = grad_output_reshaped.t() @ x.conj().reshape(batch_dim, n)
        # # We don't need to compute grad_bias explicitly, when we return grad_out Pytorch
        # # will sum over the batch dimension to get grad_bias.
        # return grad_input, grad_weight, grad_output


fused_dense_function_td = FusedDenseFuncTD.apply


class FusedDenseTD(nn.Linear):

    def forward(self, x):
        if x.is_cuda and self.bias is not None:
            return fused_dense_function_td(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)


class FusedDenseResidualFunc(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        output = fused_dense_cuda.linear_bias_forward(x.reshape(batch_dim, n), weight, bias)
        return output.reshape(*batch_shape, output.shape[-1]), x

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, grad_input):
        x, weight = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        grad_input, grad_weight, grad_bias = fused_dense_cuda.linear_bias_residual_backward(
            x.reshape(batch_dim, n), weight, grad_output.reshape(batch_dim, grad_output.shape[-1]),
            grad_input.reshape(batch_dim, n)
        )
        return grad_input.reshape_as(x), grad_weight, grad_bias


fused_dense_residual_function = FusedDenseResidualFunc.apply


class FusedDenseResidual(nn.Linear):
    """Similar to FusedDense, but we return both the output and the input.
    This is so that in the backward pass, we can combine the input gradient from the residual branch
    with the input gradient from the matrix multiply, without having to do a separate addition.
    """

    def forward(self, x):
        if x.is_cuda and self.bias is not None:
            return fused_dense_residual_function(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias), x


class FusedDenseGeluDenseFuncTD(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, weight1, bias1, weight2, bias2, checkpoint_activation=False):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        # output1, output2, gelu_in = fused_dense_cuda.linear_gelu_linear_forward(
        #     x.reshape(batch_dim, n), weight1, bias1, weight2, bias2
        # )
        # gelu_in = fused_dense_cuda.linear_bias_forward(x.reshape(batch_dim, n), weight1, bias1)
        # output1 = F.gelu(gelu_in, approximate='tanh')
        output1, gelu_in = fused_dense_cuda.linear_gelu_forward(x.reshape(batch_dim, n), weight1,
                                                                bias1)
        output2 = fused_dense_cuda.linear_bias_forward(output1, weight2, bias2)
        ctx.checkpoint_activation = checkpoint_activation
        if not checkpoint_activation:
            ctx.save_for_backward(x, weight1, weight2, gelu_in, output1)
        else:
            ctx.save_for_backward(x, weight1, weight2, bias1)
        return output2.reshape(*batch_shape, output2.shape[-1])

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        checkpoint_activation = ctx.checkpoint_activation
        x, weight1, weight2, *rest = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        if not checkpoint_activation:
            gelu_in, output1 = rest
        else:
            bias1, = rest
            output1, gelu_in = fused_dense_cuda.linear_gelu_forward(x.reshape(batch_dim, n),
                                                                    weight1, bias1)
        grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2 = fused_dense_cuda.linear_gelu_linear_backward(
            x.reshape(batch_dim, n), gelu_in, output1, weight1, weight2,
            grad_output.reshape(batch_dim, grad_output.shape[-1])
        )
        # grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        # # grad_output1, grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_backward(output1, weight2, grad_output)
        # grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_wgrad(output1, grad_output)
        # grad_gelu = matmul_dgelu(grad_output, weight2, gelu_in)
        # grad_input, grad_weight1, grad_bias1 = fused_dense_cuda.linear_bias_backward(
        #     x.reshape(batch_dim, n), weight1, grad_gelu
        # )
        return grad_input.reshape_as(x), grad_weight1, grad_bias1, grad_weight2, grad_bias2, None


fused_dense_gelu_dense_function_td = FusedDenseGeluDenseFuncTD.apply


class FusedDenseGeluDenseTD(nn.Module):

    def __init__(self, in_features, intermediate_features, out_features, bias=True,
                 checkpoint_activation=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        assert bias == True, "DenseGeluDense module without bias is currently not supported"
        self.checkpoint_activation = checkpoint_activation
        self.fc1 = nn.Linear(in_features, intermediate_features, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(intermediate_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x):
        return fused_dense_gelu_dense_function_td(x, self.fc1.weight, self.fc1.bias,
                                                  self.fc2.weight, self.fc2.bias,
                                                  self.checkpoint_activation)


class FusedDenseResGeluDenseFunc(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, weight1, bias1, weight2, bias2, checkpoint_activation=False):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        # output1, output2, gelu_in = fused_dense_cuda.linear_gelu_linear_forward(
        #     x.reshape(batch_dim, n), weight1, bias1, weight2, bias2
        # )
        # gelu_in = fused_dense_cuda.linear_bias_forward(x.reshape(batch_dim, n), weight1, bias1)
        # output1 = F.gelu(gelu_in, approximate='tanh')
        output1, gelu_in = fused_dense_cuda.linear_gelu_forward(x.reshape(batch_dim, n), weight1,
                                                                bias1)
        output2 = fused_dense_cuda.linear_bias_forward(output1, weight2, bias2)
        ctx.checkpoint_activation = checkpoint_activation
        if not checkpoint_activation:
            ctx.save_for_backward(x, weight1, weight2, gelu_in, output1)
        else:
            ctx.save_for_backward(x, weight1, weight2, bias1)
        return output2.reshape(*batch_shape, output2.shape[-1]), x

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, grad_input):
        checkpoint_activation = ctx.checkpoint_activation
        x, weight1, weight2, *rest = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        if not checkpoint_activation:
            gelu_in, output1 = rest
        else:
            bias1, = rest
            output1, gelu_in = fused_dense_cuda.linear_gelu_forward(x.reshape(batch_dim, n),
                                                                    weight1, bias1)
        grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2 = fused_dense_cuda.linear_residual_gelu_linear_backward(
            x.reshape(batch_dim, n), gelu_in, output1, weight1, weight2,
            grad_output.reshape(batch_dim, grad_output.shape[-1]),
            grad_input.reshape(batch_dim, n)
        )
        # grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        # # grad_output1, grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_backward(output1, weight2, grad_output)
        # grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_wgrad(output1, grad_output)
        # grad_gelu = matmul_dgelu(grad_output, weight2, gelu_in)
        # grad_input, grad_weight1, grad_bias1 = fused_dense_cuda.linear_bias_residual_backward(
        #     x.reshape(batch_dim, n), weight1, grad_gelu,
        #     grad_input.reshape(batch_dim, n)
        # )
        return grad_input.reshape_as(x), grad_weight1, grad_bias1, grad_weight2, grad_bias2, None


fused_dense_res_gelu_dense_function_td = FusedDenseResGeluDenseFunc.apply


class FusedDenseResGeluDense(FusedDenseGeluDenseTD):

    def forward(self, x):
        return fused_dense_res_gelu_dense_function_td(x, self.fc1.weight, self.fc1.bias,
                                                      self.fc2.weight, self.fc2.bias,
                                                      self.checkpoint_activation)
