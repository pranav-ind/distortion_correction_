
import torch as th
import torch.nn.functional as F
import numpy as np

"""
Create a one dimensional gaussian kernel matrix
"""
def gaussian_kernel_1d(sigma, asTensor=False, dtype=th.float32, device='cpu'):

    kernel_size = int(2*np.ceil(sigma*2) + 1)

    x = np.linspace(-(kernel_size - 1) // 2, (kernel_size - 1) // 2, num=kernel_size)

    kernel = 1.0/(sigma*np.sqrt(2*np.pi))*np.exp(-(x**2)/(2*sigma**2))
    kernel = kernel/np.sum(kernel)

    if asTensor:
        return th.tensor(kernel, dtype=dtype, device=device)
    else:
        return kernel


"""
Create a two dimensional gaussian kernel matrix
"""
def gaussian_kernel_2d(sigma, asTensor=False, dtype=th.float32, device='cpu'):

    y_1 = gaussian_kernel_1d(sigma[0])
    y_2 = gaussian_kernel_1d(sigma[1])

    kernel = np.tensordot(y_1, y_2, 0)
    kernel = kernel / np.sum(kernel)

    if asTensor:
        return th.tensor(kernel, dtype=dtype, device=device)
    else:
        return kernel

"""
Create a three dimensional gaussian kernel matrix
"""
def gaussian_kernel_3d(sigma, asTensor=False, dtype=th.float32, device='cpu'):

    kernel_2d = gaussian_kernel_2d(sigma[:2])
    kernel_1d = gaussian_kernel_1d(sigma[-1])

    kernel = np.tensordot(kernel_2d, kernel_1d, 0)
    kernel = kernel / np.sum(kernel)

    if asTensor:
        return th.tensor(kernel, dtype=dtype, device=device)
    else:
        return kernel


"""
    Create a Gaussian kernel matrix
"""
def gaussian_kernel(sigma, dim=1, asTensor=False, dtype=th.float32, device='cpu'):

    assert dim > 0 and dim <=3

    if dim == 1:
        return gaussian_kernel_1d(sigma, asTensor=asTensor, dtype=dtype, device=device)
    elif dim == 2:
        return gaussian_kernel_2d(sigma, asTensor=asTensor, dtype=dtype, device=device)
    else:
        return gaussian_kernel_3d(sigma, asTensor=asTensor, dtype=dtype, device=device)