import spconv.pytorch as spconv
import torch
import torch.cuda.amp
import torch.nn as nn
from spconv.pytorch import SparseModule


class SubMConvBlock(SparseModule):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        norm_fn,
        activation_fn,
        stride=1,
        padding=1,
        algo=spconv.ConvAlgo.Native,
        bias=False,
    ):
        super().__init__()

        self.sequence = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                algo=algo,
            ),
            norm_fn(output_channels),
            activation_fn(),
        )

    def forward(self, input):
        return self.sequence(input)


class EncoderBlock(SparseModule):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        norm_fn,
        activation_fn,
        stride=2,
        padding=1,
        key=None,
        algo=spconv.ConvAlgo.Native,
        bias=False,
    ):
        super().__init__()

        self.sequence = spconv.SparseSequential(
            spconv.SparseConv3d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                stride=stride,
                indice_key=key,
                algo=algo,
                bias=bias,
                padding=padding,
            ),
            norm_fn(output_channels),
            activation_fn(),
        )

    def forward(self, input):
        return self.sequence(input)


class DecoderBlock(SparseModule):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        norm_fn,
        activation_fn,
        key=None,
        algo=spconv.ConvAlgo.Native,
        bias=False,
    ):
        super().__init__()

        self.sequence = spconv.SparseSequential(
            spconv.SparseInverseConv3d(
                input_channels,
                output_channels,
                kernel_size,
                indice_key=key,
                algo=algo,
                bias=bias,
            ),
            norm_fn(output_channels),
            activation_fn(),
        )

    def forward(self, input):
        return self.sequence(input)


class ResBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        norm_fn,
        activation_fn,
        algo=spconv.ConvAlgo.Native,
        bias=False,
    ):
        super().__init__()

        if input_channels == output_channels:
            self.identity = spconv.SparseSequential(nn.Identity())
        else:
            self.identity = spconv.SparseSequential(
                spconv.SubMConv3d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    padding=1,
                    bias=False,
                    algo=algo,
                )
            )

        self.sequence = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channels, output_channels, kernel_size, bias=False, algo=algo
            ),
            norm_fn(output_channels),
            activation_fn(),
            spconv.SubMConv3d(
                output_channels, output_channels, kernel_size, bias=False, algo=algo
            ),
            norm_fn(output_channels),
        )

        self.activation_fn = spconv.SparseSequential(activation_fn())

    def forward(self, input):
        identity = spconv.SparseConvTensor(
            input.features, input.indices, input.spatial_shape, input.batch_size
        )
        output = self.sequence(input)
        output = output.replace_feature(
            output.features + self.identity(identity).features
        )
        return self.activation_fn(output)


class UBlock(nn.Module):
    def __init__(
        self,
        n_planes,
        norm_fn,
        activation_fn,
        kernel_size=3,
        key_id=1,
        algo=spconv.ConvAlgo.Native,
        bias=False,
    ):
        super().__init__()

        self.n_planes = n_planes
        self.Head = ResBlock(
            n_planes[0],
            n_planes[0],
            kernel_size,
            norm_fn,
            activation_fn,
            algo=algo,
            bias=bias,
        )

        if len(n_planes) > 1:
            self.Encode = EncoderBlock(
                n_planes[0],
                n_planes[1],
                kernel_size,
                norm_fn,
                activation_fn,
                stride=2,
                key=key_id,
                algo=algo,
                bias=bias,
            )
            self.U = UBlock(
                n_planes[1:],
                norm_fn,
                activation_fn,
                kernel_size,
                key_id + 1,
                algo,
                bias,
            )
            self.Decode = DecoderBlock(
                n_planes[1],
                n_planes[0],
                kernel_size,
                norm_fn,
                activation_fn,
                key=key_id,
                algo=algo,
                bias=bias,
            )
            self.Tail = ResBlock(
                n_planes[0] * 2,
                n_planes[0],
                kernel_size,
                norm_fn,
                activation_fn,
                algo=algo,
                bias=bias,
            )

    def forward(self, input):
        output = self.Head(input)

        identity = spconv.SparseConvTensor(
            output.features,
            output.indices,
            output.spatial_shape,
            output.batch_size,
        )

        if len(self.n_planes) > 1:
            output = self.Encode(output)
            output = self.U(output)
            output = self.Decode(output)
            output = output.replace_feature(
                torch.cat((identity.features, output.features), dim=1)
            )
            output = self.Tail(output)

        return output


class SparseFC(nn.Module):
    def __init__(
        self,
        n_planes,
        norm_fn,
        activation_fn=None,
        kernel_size=1,
        algo=spconv.ConvAlgo.Native,
        bias=False,
    ):
        super().__init__()

        self.sequence = spconv.SparseSequential()
        for i in range(len(n_planes) - 2):
            self.sequence.add(
                spconv.SubMConv3d(
                    n_planes[i],
                    n_planes[i + 1],
                    kernel_size=kernel_size,
                    bias=False,
                    algo=algo,
                    padding=0,
                )
            )
            self.sequence.add(norm_fn(n_planes[i + 1]))
            self.sequence.add(activation_fn())

        self.sequence.add(
            spconv.SubMConv3d(
                n_planes[-2],
                n_planes[-1],
                kernel_size=kernel_size,
                bias=False,
                algo=algo,
                padding=0,
            )
        )

    def forward(self, input):
        return self.sequence(input)


class MLP(nn.Module):
    def __init__(
        self,
        n_planes,
        norm_fn,
        activation_fn=None,
        bias=False,
    ):
        super().__init__()

        self.sequence = spconv.SparseSequential()

        for i in range(len(n_planes) - 2):
            self.sequence.add(
                nn.Linear(
                    n_planes[i],
                    n_planes[i + 1],
                    bias=bias,
                )
            )
            self.sequence.add(norm_fn(n_planes[i + 1]))
            self.sequence.add(activation_fn())

        self.sequence.add(
            nn.Linear(
                n_planes[-2],
                n_planes[-1],
                bias=bias,
            )
        )

    def forward(self, input):
        return self.sequence(input)
