#!/usr/bin/env python3

import math
from typing import Optional

import torch
# from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.functions import MaternCovariance
from gpytorch.settings import trace_mode
from gpytorch.kernels.kernel import Kernel
# from ..functions import MaternCovariance
# from ..settings import trace_mode
# from .kernel import Kernel


class SemanticMaternKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Matern kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

       \begin{equation*}
          k_{\text{Matern}}(\mathbf{x_1}, \mathbf{x_2}) = \frac{2^{1 - \nu}}{\Gamma(\nu)}
          \left( \sqrt{2 \nu} d \right)^{\nu} K_\nu \left( \sqrt{2 \nu} d \right)
       \end{equation*}

    where

    * :math:`d = (\mathbf{x_1} - \mathbf{x_2})^\top \Theta^{-2} (\mathbf{x_1} - \mathbf{x_2})`
      is the distance between
      :math:`x_1` and :math:`x_2` scaled by the lengthscale parameter :math:`\Theta`.
    * :math:`\nu` is a smoothness parameter (takes values 1/2, 3/2, or 5/2). Smaller values are less smooth.
    * :math:`K_\nu` is a modified Bessel function.

    There are a few options for the lengthscale parameter :math:`\Theta`:
    See :class:`gpytorch.kernels.Kernel` for descriptions of the lengthscale options.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    :param nu: (Default: 2.5) The smoothness parameter.
    :type nu: float (0.5, 1.5, or 2.5)
    :param ard_num_dims: (Default: `None`) Set this if you want a separate lengthscale for each
        input dimension. It should be `d` if x1 is a `... x n x d` matrix.
    :type ard_num_dims: int, optional
    :param batch_shape: (Default: `None`) Set this if you want a separate lengthscale for each
         batch of input data. It should be `torch.Size([b1, b2])` for a `b1 x b2 x n x m` kernel output.
    :type batch_shape: torch.Size, optional
    :param active_dims: (Default: `None`) Set this if you want to
        compute the covariance of only a few input dimensions. The ints
        corresponds to the indices of the dimensions.
    :type active_dims: Tuple(int)
    :param lengthscale_prior: (Default: `None`)
        Set this if you want to apply a prior to the lengthscale parameter.
    :type lengthscale_prior: ~gpytorch.priors.Prior, optional
    :param lengthscale_constraint: (Default: `Positive`) Set this if you want
        to apply a constraint to the lengthscale parameter.
    :type lengthscale_constraint: ~gpytorch.constraints.Interval, optional
    :param eps: (Default: 1e-6) The minimum value that the lengthscale can take (prevents divide by zero errors).
    :type eps: float, optional

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        >>> # Non-batch: ARD (different lengthscale for each input dimension)
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=5))
        >>> covar = covar_module(x)  # Output: LazyVariable of size (10 x 10)
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.MaternKernel(nu=0.5, batch_shape=torch.Size([2])
        >>> covar = covar_module(x)  # Output: LazyVariable of size (2 x 10 x 10)
    """

    has_lengthscale = True

    def __init__(self, embeddingModel, tokenizer, nu: Optional[float] = 2.5,**kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(SemanticMaternKernel, self).__init__(**kwargs)
        self.embeddingModel = embeddingModel
        self.nu = nu
        self.tokenizer = tokenizer

    def embed(self, x):
        if x.dim() == 2:
            x_3 = x.unsqueeze(0)
        elif x.dim() == 3:
            x_3 = x
        x_tensor = torch.empty(x_3.size(0), x_3.size(1), 768)
        for i in range(x_3.size(0)):
            for j in range(x_3.size(1)):
                text = self.tokenizer.decode(torch.round(x_3[i][j]).int())
                x_tensor[i][j] = self.embeddingModel.encode(text)
                
        if x.dim() == 2:
            x_tensor = x_tensor.squeeze(0)
        if x.shape[:-1] != x_tensor.shape[:-1]:
            print("x.shape", x.shape)
            print("x_tensor.shape", x_tensor.shape)
            raise Exception("x.shape != x_tensor.shape")

        print(f"embed x_tensor.shape {x_tensor.shape}")
        return x_tensor
    
    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):

            # print("forward My ***x1", x1.size())
            # print("forward My***x2", x2.size())
            # print(self.embed(x1))
            x1e = self.embed(x1)
            x2e = self.embed(x2)

            mean = x1e.reshape(-1, x1e.size(-1)).mean(0)[(None,) * (x1e.dim() - 1)]
            lengthscale = self.lengthscale.new_full((self.lengthscale.shape[0:-1][0], x1e.shape[-1]), float(self.lengthscale[0, 1]))

            x1_ = (x1e - mean).div(lengthscale)
            x2_ = (x2e - mean).div(lengthscale)

            distance = self.covar_dist(x1_, x2_, diag=diag, **params)
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            elif self.nu == 2.5:
                constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
            # print("forward My ***constant_component", constant_component.size())
            return constant_component * exp_component
        return MaternCovariance.apply(
            x1e, x2e, self.lengthscale, self.nu, lambda x1e, x2e: self.covar_dist(x1e, x2e, **params)
        )



