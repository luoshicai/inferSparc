import torch
import sten

input_tensor = torch.randn(6,5)
print("input_tensor: ", input_tensor)

class MLP(torch.nn.Module):
    def __init__(self, channel_sizes):
        super().__init__()
        self.layers = torch.nn.Sequential()
        in_out_pairs = list(zip(channel_sizes[:-1], channel_sizes[1:]))
        for idx, (in_channels, out_channels) in enumerate(in_out_pairs):
            if idx != 0:
                self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Linear(in_channels, out_channels))

    def forward(self, input):
        return self.layers(input)


model = MLP([5, 4, 3, 2])
output = model(input_tensor)
print("out_put: ", output)

class SparseLinear(torch.nn.Module):
    def __init__(self, input_features, output_features, weight_sparsity):
        super().__init__()
        self.weight_sparsity = weight_sparsity
        self.weight = sten.SparseParameterWrapper(
            sten.random_fraction_sparsifier_dense_csc(
                sten.RandomFractionSparsifier(self.weight_sparsity),
                torch.randn(output_features, input_features),
                (
                    sten.KeepAll(),
                    torch.Tensor,
                    sten.RandomFractionSparsifier(self.weight_sparsity),
                    sten.CscTensor,
                )
            )
        )
        self.bias = torch.nn.Parameter(torch.rand(output_features))
        
        print("self.weight: ", self.weight._wrapped_tensor_container[0].to_dense())
        print("self.bias: ", self.bias)

    def forward(self, input):
        sparse_op = sten.sparsified_op(
            orig_op=torch.nn.functional.linear,
            out_fmt=tuple(
                [(sten.KeepAll(), torch.Tensor,
                  sten.KeepAll(), torch.Tensor)]
            ),
            grad_out_fmt=tuple(
                [(sten.KeepAll(), torch.Tensor,
                  sten.KeepAll(), torch.Tensor)]
            ),
        )

        return sparse_op(input, self.weight, self.bias)
    
class SparseMLP(torch.nn.Module):
    def __init__(self, channel_sizes, weight_sparsity):
        super().__init__()
        self.layers = torch.nn.Sequential()
        in_out_pairs = list(zip(channel_sizes[:-1], channel_sizes[1:]))
        for idx, (in_channels, out_channels) in enumerate(in_out_pairs):
            if idx != 0:
                self.layers.append(torch.nn.ReLU())
            self.layers.append(SparseLinear(
                in_channels, out_channels, weight_sparsity))

    def forward(self, input):
        return self.layers(input)
    
sparse_model = SparseMLP([5, 4, 3, 2], 0.8)
sparse_output = sparse_model(input_tensor)
print("sparse_model output: ", sparse_output)
print("test sten1 success")