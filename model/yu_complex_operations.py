import torch 
import torch.nn as nn
import numpy as np
# import coremltools as ct  


def view_as_complex(real_tensor):
    """
    Convert a real tensor with the last dimension of size 2 into a complex tensor.
    :param real_tensor: A tensor where the last dimension represents real and imaginary parts.
    :return: A complex tensor.
    """
    # Check if the input tensor has the right dimension, disabled to remove model conversion warning.
    #if real_tensor.size(-1) != 2:
    #    raise ValueError("The last dimension of the input tensor should be of size 2.")

    real_part = real_tensor[..., 0]
    imag_part = real_tensor[..., 1]

    return torch.complex(real_part, imag_part)

def complex_add(a, b):
    """
    Perform element-wise addition for two complex numbers.
    tensor([17.+12.j,  3.+4.j]) + tensor([0.5000+1.j, 1.5000+2.j]) = tensor([17.5000+13.j,  4.5000+6.j])
    Args:
        a (torch.tensor): Tensor containing complex numbers.
        b (torch.tensor): Tensor containing complex numbers.

    Returns:
        torch.tensor: Tensor containing the addition result of two complex numbers.
    """
    # Extract real and imaginary parts of a and b
    real_a, imag_a = torch.real(a), torch.imag(a)
    real_b, imag_b = torch.real(b), torch.imag(b)

    real_result = real_a + real_b
    imag_result = imag_a + imag_b

    # Combine real and imaginary parts to create complex tensor
    result = torch.complex(real_result, imag_result)
   
    return result

def complex_scalar_add(a, scalar):
    """
    Perform element-wise division for two complex numbers.

    Args:
        a (torch.tensor): Tensor containing complex numbers.
        b : a number.

    Returns:
        torch.tensor: Tensor containing the result of element-wise division.
    """
    # Extract real and imaginary parts of a and b
    real_a, imag_a = torch.real(a), torch.imag(a)

    real_result = real_a + scalar
    imag_result = imag_a

    # Combine real and imaginary parts to create complex tensor
    result = torch.complex(real_result, imag_result)

    return result

def complex_minus(a, b):
    """
    Perform element-wise minus for two complex numbers.
    tensor([17.+12.j,  3.+4.j]) - tensor([0.5000+1.j, 1.5000+2.j]) = tensor([16.5000+11.j,  1.5000+2.j])
    
    Args:
        a (torch.tensor): Tensor containing complex numbers.
        b (torch.tensor): Tensor containing complex numbers.

    Returns:
        torch.tensor: Tensor containing the result of element-wise minus.
    """
    # Extract real and imaginary parts of a and b
    real_a, imag_a = torch.real(a), torch.imag(a)
    real_b, imag_b = torch.real(b), torch.imag(b)

    real_result = real_a - real_b
    imag_result = imag_a - imag_b

    # Combine real and imaginary parts to create complex tensor
    result = torch.complex(real_result, imag_result)
   
    return result
    

def complex_multiple(a, b):
    """
    Perform element-wise multiplication for two complex numbers.
    tensor([17.+12.j,  3.+4.j]) * tensor([0.5000+1.j, 1.5000+2.j]) = tensor([-3.5000+23.j, -3.5000+12.j])

    Args:
        a (torch.tensor): Tensor containing complex numbers.
        b (torch.tensor): Tensor containing complex numbers.

    Returns:
        torch.tensor: Tensor containing the result of element-wise division.
    """
    # Extract real and imaginary parts of a and b
    real_a, imag_a = torch.real(a), torch.imag(a)
    real_b, imag_b = torch.real(b), torch.imag(b)

    real_result = real_a * real_b - imag_a * imag_b
    imag_result = real_a * imag_b + imag_a * real_b
    
    # Combine real and imaginary parts to create complex tensor
    result = torch.complex(real_result, imag_result)
   
    return result
    

def complex_division(a, b):
    """
    Perform element-wise division for two complex numbers.
    tensor([1.+2.j, 3.+4.j]) / tensor([0.5000+1.j, 1.5000+2.j]) = tensor([2.+0.j, 2.+0.j])
    
    Args:
        a (torch.tensor): Tensor containing complex numbers.
        b (torch.tensor): Tensor containing complex numbers.

    Returns:
        torch.tensor: Tensor containing the result of element-wise division.
    """
    # Extract real and imaginary parts of a and b
    real_a, imag_a = torch.real(a), torch.imag(a)
    real_b, imag_b = torch.real(b), torch.imag(b)

    # Compute the denominator for division
    #denominator = real_b**2 + imag_b**2
    denominator = real_b*real_b + imag_b*imag_b

    # Compute real and imaginary parts of the result
    real_result = (real_a * real_b + imag_a * imag_b) / denominator
    imag_result = (imag_a * real_b - real_a * imag_b) / denominator
    # Combine real and imaginary parts to create complex tensor
    result = torch.complex(real_result, imag_result)
   
    return result

def complex_unsqueeze(complex_m, dim):
    real, imag = torch.real(complex_m), torch.imag(complex_m)

    real_result = real.unsqueeze(dim)
    imag_result = imag.unsqueeze(dim)

    # Combine real and imaginary parts to create complex tensor
    result = torch.complex(real_result, imag_result)

    return result

def complex_torch_sum(input_tensor, dim=None, keepdim=False):
    """
    Compute the sum of a complex number tensor along the specified dimension.

    Args:
        input_tensor (torch.Tensor): The input complex number tensor.
        dim (int, optional): The dimension along which to compute the sum.
        keepdim (bool, optional): Whether to keep the resulting dimension or not.

    Returns:
        torch.Tensor: The sum of the input tensor along the specified dimension.
    """

    real, imag = torch.real(input_tensor), torch.imag(input_tensor)
    if dim is None:
        # Sum all elements if dim is not specified
        real_sum = torch.sum(real)
        imag_sum = torch.sum(imag)
        return torch.complex(real_sum, imag_sum)
    else:
        # Sum along the specified dimension
        real_sum = torch.sum(real, dim=dim, keepdim=keepdim)
        imag_sum = torch.sum(imag, dim=dim, keepdim=keepdim)
        return torch.complex(real_sum, imag_sum)

def complex_scalar_multiple(a, scalar):
    """
    Perform the multiplication for a complex number and a scalar.
    tensor([17.+12.j,  3.+4.j]) * 5 = tensor([85.+60.j, 15.+20.j])
    
    Args:
        a (torch.tensor): Tensor containing complex numbers.
        b (torch.tensor): Tensor containing complex numbers.

    Returns:
        torch.tensor: Tensor containing the multiplication result of a complex number and a scalar.
    """
    # Extract real and imaginary parts of a and b
    real_a, imag_a = torch.real(a), torch.imag(a)

    real_result = real_a * scalar
    imag_result = imag_a * scalar
    
    # Combine real and imaginary parts to create complex tensor
    result = torch.complex(real_result, imag_result)
   
    return result
    

def complex_array_scalar_multiplication(complex_array, scalar_array):
    """
    implement 
    [17.0 + 12.0j, 3.0 + 4.0j] x [[9], 
                                  [1.5], 
                                  [4], 
                                  [89], 
                                  [3]]
    """
    # Extract real and imaginary parts of complex_array
    real_part = torch.real(complex_array)
    imag_part = torch.imag(complex_array)

    # Element-wise multiplication with real and imaginary parts separately
    real_result = real_part * scalar_array
    imag_result = imag_part * scalar_array

    # Combine real and imaginary parts into a single tensor
    result = torch.complex(real_result, imag_result)

    return result

def scalar_complex_div(scalar, a):
    """
    Perform element-wise division for a scalar and a complex number.
    5 / tensor([17.+12.j,  3.+4.j]) = tensor([0.1963-0.1386j, 0.6000-0.8000j])

    Args:
        a (torch.tensor): Tensor containing complex numbers.
        b (torch.tensor): Tensor containing complex numbers.

    Returns:
        torch.tensor: Tensor containing the division result of a scalar and a complex number.
    """
    # Extract real and imaginary parts of a and b
    real_a, imag_a = torch.real(a), torch.imag(a)
    denominator = real_a*real_a + imag_a*imag_a

    real_result = scalar * real_a / denominator
    imag_result = scalar * imag_a / denominator
    
    # Combine real and imaginary parts to create complex tensor
    #result = view_as_complex(torch.stack((real_result, imag_result), dim=-1))
    result = torch.complex(real_result, imag_result)
   
    return result

def complex_pow_scalar_array(a, e):
    """
    Perform element-wise division for two complex numbers.

    Args:
        a (torch.tensor): Tensor containing complex numbers.
        e (torch.tensor): Tensor array like torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]).

    Returns:
        torch.tensor: Tensor containing the result of a power b.
        
    a = tensor(0.9239-0.3827j)
    exponent = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])

    output =tensor([ 1.0000e+00-0.0000e+00j,  9.2388e-01-3.8268e-01j,
                     7.0711e-01-7.0711e-01j,  3.8268e-01-9.2388e-01j,
                    -4.3711e-08-1.0000e+00j, -3.8268e-01-9.2388e-01j,
                    -7.0711e-01-7.0711e-01j, -9.2388e-01-3.8268e-01j,
                    -1.0000e+00+8.7423e-08j])
    """
    # Extract real and imaginary parts of a and b
    real_a, imag_a = torch.real(a), torch.imag(a)
    magnitude = (real_a**2 + imag_a**2)**(e / 2)

    # Compute the pha
    phase = torch.atan2(imag_a, real_a)
    # Apply the exponent to magnitude
    magnitude = magnitude ** e

    # Compute the real and imaginary parts
    real_result = magnitude * torch.cos(phase * e)
    imag_result = magnitude * torch.sin(phase * e)

    # Combine real and imaginary parts to create complex tensor
    result = torch.complex(real_result, imag_result)

    return result

def complex_conj (complex_m):
    real, imag = torch.real(complex_m), torch.imag(complex_m)

    real_result = real
    imag_result = imag *(-1)

    # Combine real and imaginary parts to create complex tensor
    result = torch.complex(real_result, imag_result)

    return result
           
def main():
    a = torch.tensor([17.0 + 12.0j, 3.0 + 4.0j])
    b = torch.tensor([0.5 + 1.0j, 1.5 + 2.0j])

    c = a + b
    print (f"{a} + {b} = {c}")
    
    
        
if __name__ == '__main__':
    main()

