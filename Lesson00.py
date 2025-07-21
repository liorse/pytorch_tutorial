# going over lesson 00 from https://www.learnpytorch.io/00_pytorch_fundamentals/

#%%
import torch
# %%
scalar = torch.tensor(7)
scalar.ndim
scalar.item()
# %%

vector = torch.tensor([7, 3])
vector.ndim
vector.shape
# %%
#my password to library Jcoiq983
# %%
matrix = torch.tensor([[7, 3], 
                       [2, 5]])
matrix.shape
# %%
tensor = torch.tensor([[[7, 3, 2],
                        [2, 5, 4],
                        [1, 4, 6]]])
tensor.shape

# %%
# create a random tensor with a size of (3, 4)
random_tensor = torch.rand(3, 4)
random_tensor.shape
random_tensor.ndim
random_tensor
# %%
random_tensor = torch.rand(size=(224, 224, 3))
random_tensor.shape
#random_tensor.ndim
#random_tensor
# %%
zeros = torch.zeros(size=(3, 4))
zeros.shape
zeros.ndim
zeros.dtype
# %%
ones = torch.ones(size=(3, 4))
ones.dtype
ones
# %%
range_of_numbers = torch.arange(start=1, end=101, step=3)
range_of_numbers
zeros = torch.ones_like(range_of_numbers)
zeros
# %%
float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=None, device=None, requires_grad=False)
float_32_tensor.dtype, float_32_tensor.device, float_32_tensor.shape, float_32_tensor.device

# %%
float_16_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=torch.float16 )
float_16_tensor.dtype, float_16_tensor.device, float_16_tensor.shape, float

# print information about float_16_tensor_16_tensor.device
print(f"tensor shape {float_16_tensor.shape}")
print("Float 16 Tensor:", float_16_tensor)



# %%
# tensor manipulation
tensor = torch.tensor([1, 2, 3])
tensor_plus = tensor + 1
tensor_plus_multiply = tensor_plus * 10
tensor_plus_multiply = torch.multiply(tensor_plus, 10)
tensor_plus_multiply

element_wise_multiply = tensor * tensor
element_wise_multiply
# %%
tensor_mul = tensor * tensor
tensor_mul
tensor_mat_mul = tensor @ tensor
tensor_mat_mul
# %%
# create matmul error
tensor1 = torch.rand(2,3)
tensor2 = torch.rand(2,3)
tensor3 = tensor1 @ tensor2.T
print(tensor3)

# %%
torch.manual_seed(42)
tensor_A = torch.rand(2,3)
linear = torch.nn.Linear(in_features=3, out_features=6)
x = tensor_A 
output = linear(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")   
print(f"Output: {output}")
# %%
# aggregate tensors
x = torch.arange(10, 100, 10)
print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
print(f"Mean: {x.float().mean()}")
print(f"Sum: {x.sum()}")

# return the index of the minimum value
print(f"Index of minimum value: {x.argmin()}")
# return the index of the maximum value
print(f"Index of maximum value: {x.argmax()}")
# %%
# changing the data type of a tensor
tensor = torch.arange(1., 10, 1)
tensor.dtype
tensor_float16 = tensor.type(torch.float16)
tensor_float16.dtype
tensor_float16
# %%
x = torch.arange(1, 8)
x_reshaped = x.reshape(1, 7)
x_reshaped.shape
x_reshaped
z = x.view(1, 7)
z.shape
z[:, 0] = 5
z, x

x_stacked = torch.stack([x, x, x], dim=0)
x_stacked

x_squeezed = x_reshaped.squeeze()
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
x_unsqueezed
# %%
# permute the dimensions of a tensor
x = torch.rand(224, 224, 3)
print(x.shape)
x_permuted = x.permute(2, 0, 1)  # change the order of dimensions
print(x_permuted.shape)
# %%
# indexing tensors, selecting elements from tensors
x = torch.arange(1, 10).reshape(1, 3, 3)
x, x.shape
x[0,:, 2]
# %%
# working with numpy
import numpy as np
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array).type(torch.float32)
array, tensor

# Tensor to numpy
tensor = torch.ones(7)
numpy_array = tensor.numpy()
tensor, numpy_array
tensor = tensor + 1
numpy_array, tensor
# %%
# working with random nubmers
rx = torch.rand(3, 4)
ry = torch.rand(3, 4)

print(f"Random Tensor 1:\n{rx}\n")
print(f"Random Tensor 2:\n{ry}\n")
rx == ry

SEED_NUMBER = 42
torch.random.manual_seed(SEED_NUMBER)
rx = torch.rand(3, 4)
torch.random.manual_seed(SEED_NUMBER)
ry = torch.rand(3, 4)

print(f"Random Tensor 1:\n{rx}\n")
print(f"Random Tensor 2:\n{ry}\n")
rx == ry

# %%
torch.backends.mps.is_available()
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
# %%
# let's try to put a tensor on the GPU
tensor = torch.tensor([1, 2, 3])
print(tensor, tensor.device)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu, tensor_on_gpu.device)
# %%
# if a tensor is on the GPU we can't transfrom it to a numpy array
# tensor_on_gpu.numpy() # this will raise an error
tensor_on_gpu.cpu().numpy()  # this will raise an error
# %%
#Create a random tensor with shape (7, 7).
x = torch.rand(7, 7)
x
#Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7) (hint: you may have to transpose the second tensor).
y = torch.rand(1, 7)
x @ y.T
#Set the random seed to 0 and do exercises 2 & 3 over again.
torch.random.manual_seed(0)
x = torch.rand(7, 7)
torch.random.manual_seed(0)
y = torch.rand(1, 7)
x @ y.T
#Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent? (hint: you'll need to look into the documentation for torch.cuda for this one). If there is, set the GPU random seed to 1234.
torch.mps.manual_seed(1234)  # for MPS (Apple Silicon)
#Create two random tensors of shape (2, 3) and send them both to the GPU (you'll need access to a GPU for this). Set torch.manual_seed(1234) when creating the tensors (this doesn't have to be the GPU random seed).
torch.manual_seed(1234)
x = torch.rand(2, 3).to(device)
torch.manual_seed(1234)
y = torch.rand(2, 3).to(device)
#Perform a matrix multiplication on the tensors you created in 6 (again, you may have to adjust the shapes of one of the tensors).
z = x @ y.T
#Find the maximum and minimum values of the output of 7.
print(f"Max: {z.max()}, Min: {z.min()}")
#Find the maximum and minimum index values of the output of 7.
print(f"Max index: {z.argmax()}, Min index: {z.argmin()}")
#Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with all the 1 dimensions removed to be left with a tensor of shape (10). Set the seed to 7 when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape.
torch.manual_seed(7)
x = torch.rand(1, 1, 1, 10)
print(f"Tensor with 1s:\n{x}\nShape: {x.shape}")
x_squeezed = x.squeeze()
print(f"Tensor with 1s removed:\n{x_squeezed}\nShape: {x_squeezed.shape}")
#Create a tensor with shape (1, 2, 3, 4) and
# permute the dimensions so it has shape (4, 3, 2, 1). Set the seed to 8 when you create it and print out the original tensor and the permuted tensor.
torch.manual_seed(8)
x = torch.rand(1, 2, 3, 4)
print(f"Original tensor:{x} Shape: {x.shape}")   
#x_permuted = x.permute(3, 2, 1, 0)
# %%

# %%
