import open3d.ml.torch as ml3d
import os
import pickle
import numpy as np
import pandas as pd
import sys
import torch
from model import *

import open3d as o3d
o3d.visualization.webrtc_server.enable_webrtc()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data = np.load(os.path.join('data', 'data.npy'))
# data = data[:, [0,1,2]]

# v = ml3d.vis.Visualizer()
# v.visualize_dataset(data, 'all', range(100))

if __name__ == '__main__':

    experiment_name = sys.argv[1]

    cube_red = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    cube_red.compute_vertex_normals()
    cube_red.paint_uniform_color((1.0, 0.0, 0.0))
    o3d.visualization.draw(cube_red)

    model = BaselineModel(input_channels=3, n_class=13)
    model_path = os.path.join('experiment_data', experiment_name, 'best_model.pth')
    print (f"Loading model state dict from {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.to(device = DEVICE)
    model.eval()

    # data = np.load(os.path.join('data', 'data.npy'))
    with open(os.path.join('data', 'data.pkl'), 'rb') as pickle_file:
        data = pickle.load(pickle_file).to_numpy()
    coords = data[:, [0, 1, 2]]
    pixel_vals = data[:, [3, 4, 5]]
    labels = data[:, 6]

    coords = torch.from_numpy(coords).type(torch.FloatTensor).to(device=DEVICE)
    print (coords.shape)
    pixel_vals = torch.from_numpy(pixel_vals).type(torch.FloatTensor).to(device=DEVICE)
    labels = torch.from_numpy(labels).type(torch.LongTensor).to(device=DEVICE)

    out = model(coords, pixel_vals)
    pred = torch.nn.Softmax(dim=1)(out).argmax(axis=1)

    v = ml3d.vis.Visualizer()

    # label lookup table needed for plotting and assigning colors
    lut = ml3d.vis.LabelLUT()

    # Setting colors for all 13 possible labels, even though some don't exist in the dataset
    possible_labels = torch.unique(labels, sorted=True)
    for i in possible_labels:
        lut.add_label(str(i), i)

    # Use same lut for both labels and pred
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)

    vis_d = [{
        "name": "experiment",
        "points": coords,
        "labels": labels
    }]

    v.visualize(vis_d)