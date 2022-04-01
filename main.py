from model import *
from dataset_factory import *
from constants import *
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import copy
import os
from matplotlib import pyplot as plt
import json
from sklearn.utils import class_weight
from utils import *
from datetime import datetime

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Experiment(object):

    def __init__(self, config_file) -> None:
        # Read the config file
        if os.path.isfile(config_file+'.json'):
            with open (config_file+'.json') as f:
                self.__config_data = json.load(f)
        else:
            raise FileNotFoundError(f"Config file {config_file}.json doesn't exist")

        self.__name = self.__config_data['experiment_name']
        # Used to store models and experiment specific outputs
        self.__experiment_dir = os.path.join('experiment_data', self.__name)
        
        # Read dataset
        self.__train_loader, self.__val_loader = get_datasets(self.__config_data)

        # Setup Experiment
        self.__epochs = self.__config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None

        # Init model
        self.__model = self.get_model(self.__config_data)

        # Weight initialization
        for p in self.__model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        labels_reshaped = np.array(self.__train_loader.dataset.labels).reshape(-1,1)
        class_weights = torch.Tensor(class_weight.compute_class_weight('balanced', \
            classes=np.unique(labels_reshaped), y=labels_reshaped.flatten())).to(DEVICE)

        # Cross Entropy applies softmax internally
        # self.__criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.__criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=self.__config_data['experiment']['learning_rate'])

        self.__init_model()

        # Load experiment data, if available
        self.__load_experiment()



    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir) and not os.listdir(self.__experiment_dir):
            print ("Empty experiment directory found")
        elif os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])
        else:
            os.makedirs(self.__experiment_dir)



    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()



    def get_model(self, config_data):
        return BaselineModel(3, 13)



    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs): 
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss, self.__mean_iou_score = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    

    def __train(self):
        best_iou_score = 0.0
        self.__model.train()
        training_loss = 0

        for i, (coords, pixel_vals, labels) in enumerate(self.__train_loader):

            scaler = torch.cuda.amp.grad_scaler.GradScaler()

            coords = coords.type(torch.FloatTensor).to(device=DEVICE)
            pixel_vals = pixel_vals.type(torch.FloatTensor).to(device=DEVICE)
            labels = labels.type(torch.LongTensor).to(device=DEVICE)

            self.__optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.__model(coords, pixel_vals) # N x Q x vocab_size
                loss = self.__criterion(outputs, labels)
                
                batch_loss = loss.sum().item() / labels.shape[0]
                # print (f"Batch Label Shape: {labels.shape[0]}")
                training_loss += loss.sum().item()

            scaler.scale(loss).backward(retain_graph=False)
            scaler.step(self.__optimizer)
            scaler.update()

            if i % 100 == 0:
                print("Batch {} Loss: {}".format(i, batch_loss))

        training_loss /= len(self.__train_loader)
        
        # TODO: Calculate IoU score

        return training_loss



    def __val(self):
        self.__model.eval()
        val_loss = 0
        losses = []
        mean_iou_scores = []
        accuracy = []

        with torch.no_grad():

            for i, (coords, pixel_vals, labels) in enumerate(self.__val_loader):
                
                coords = coords.type(torch.FloatTensor).to(device=DEVICE)
                pixel_vals = pixel_vals.type(torch.FloatTensor).to(device=DEVICE)
                labels = labels.type(torch.LongTensor).to(device=DEVICE)

                outputs = self.__model(coords, pixel_vals)
                
                loss = self.__criterion(outputs, labels)
                losses.append(loss.item())

                pred = nn.Softmax(dim=1)(outputs).argmax(axis=1)

                # TODO
                n_class = self.__config_data["model"]["n_classes"]
                mean_iou_scores.append(np.nanmean(iou(pred, labels, n_class)))
                # compute pixel accuracy???

                val_loss += loss.sum().item()

            val_loss /= len(self.__val_loader)

        print("Loss at epoch {ep} is {ml}".format(ep=self.__current_epoch, ml=np.mean(losses)))
        print("IoU at epoch {ep} is {iou}".format(ep=self.__current_epoch, iou=np.mean(mean_iou_scores)))

        if len(self.__val_losses) == 0:
            # New experiment
            self.__best_model = self.__model.state_dict()
            torch.save(self.__model.state_dict(), os.path.join(self.__experiment_dir, 'best_model.pth'))
        elif val_loss < min(self.__val_losses):
            # Better model than previously saved model
            self.__best_model = self.__model.state_dict()
            torch.save(self.__model.state_dict(), os.path.join(self.__experiment_dir, 'best_model.pth'))
        
        return val_loss, np.mean(mean_iou_scores)



    def __test(self):
        pass


    
    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)



    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)



    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)



    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Mean IoU: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, \
            self.__mean_iou_score, str(time_elapsed), str(time_to_completion))
        self.__log(summary_str, 'epoch.log')



    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()



if __name__ == "__main__":

    # The name of the json config file containing model hyperparameters is passed as arg
    config_file = 'config'    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    experiment = Experiment(config_file)
    experiment.run()

    # Test function not implemented
    # experiment.test()

    
    