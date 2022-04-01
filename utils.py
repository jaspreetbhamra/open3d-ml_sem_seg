import os
import json
import numpy as np

def read_file(path):
    if os.path.isfile(path):
        with open(path) as json_file:
            data = json.load(json_file)
        return data
    else:
        raise Exception("file doesn't exist: ", path)



def read_file_in_dir(root_dir, file_name):
    path = os.path.join(root_dir, file_name)
    return read_file(path)



def write_to_file(path, data):
    with open(path, "w") as outfile:
        json.dump(data, outfile)



def write_to_file_in_dir(root_dir, file_name, data):
    path = os.path.join(root_dir, file_name)
    write_to_file(path, data)



def log_to_file(path, log_str):
    with open(path, 'a') as f:
        f.write(log_str + '\n')



def log_to_file_in_dir(root_dir, file_name, log_str):
    path = os.path.join(root_dir, file_name)
    log_to_file(path, log_str)



def iou(pred, labels, n_classes):
    ious = []
    pred = pred.view(-1)
    target = labels.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(n_classes):  # last class is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()#.data.cpu()[0]  # Cast to long to prevent overflows
        union = pred_inds.long().sum().item()+ target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious)