import torch
import torch.nn as nn
import constants
import open3d.ml.torch as ml3d

class BaselineModel(nn.Module):

    def __init__(self, input_channels, n_class) -> None:
        super().__init__()
        
        # Set number of channels for the hidden layers to a constant of 3 for now
        hidden_channels = 3
        # Set kernel size to a constant size of 3 x 3 x 3 for now
        kernel_size = [3,3,3]
        
        # TODO: Check if the activation works, otherwise change it to None and add an explicit relu layer
        self.encoder_layer1 = ml3d.layers.SparseConv(in_channels=input_channels, \
            filters=hidden_channels, kernel_size=kernel_size, use_bias=True, activation='relu')
        self.encoder_layer2 = ml3d.layers.SparseConv(in_channels=hidden_channels, \
            filters=hidden_channels, kernel_size=kernel_size, use_bias=True, activation='relu')
        self.encoder_layer3 = ml3d.layers.SparseConv(in_channels=hidden_channels, \
            filters=hidden_channels, kernel_size=kernel_size, use_bias=True, activation='relu')

        self.decoder_layer1 = ml3d.layers.SparseConvTranspose(in_channels=hidden_channels, \
            filters=hidden_channels, kernel_size=kernel_size, use_bias=True, activation='relu')
        self.decoder_layer2 = ml3d.layers.SparseConvTranspose(in_channels=hidden_channels, \
            filters=hidden_channels, kernel_size=kernel_size, use_bias=True, activation='relu')
        self.decoder_layer3 = ml3d.layers.SparseConvTranspose(in_channels=hidden_channels, \
            filters=hidden_channels, kernel_size=kernel_size, use_bias=True, activation='relu')

        self.one_conv = nn.Conv2d(hidden_channels, n_class, kernel_size=1)

        self.voxel_size_encoder = [0.02, 0.02*2, 0.02*4]
        self.voxel_size_decoder = [0.02*4, 0.02*2, 0.02]

    def forward(self, coords, pixel_vals):
        pass

    def __call__(self, coords, pixel_vals):
        return self.forward(coords, pixel_vals)

    