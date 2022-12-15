import os
import torch
import copy
import tqdm
import sys
from torchvision.transforms.functional import to_pil_image
import matplotlib.pylab as plt
import random
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import random
from torchvision import models
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau



np.random.seed(2022)
random.seed(2022)
torch.manual_seed(2022)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_vids(path2ajpgs):
    listOfCats = os.listdir(path2ajpgs)
    ids = []
    labels = []
    for catg in listOfCats:
        path2catg = os.path.join(path2ajpgs, catg)
        listOfSubCats = os.listdir(path2catg)
        random.seed(2020)
        random.shuffle(listOfSubCats)
        path2subCats = []
        for i,los in enumerate(listOfSubCats):
            path2subCats.append(os.path.join(path2catg,los))
            
        ids.extend(path2subCats)
        labels.extend([catg]*len(listOfSubCats))
    return ids, labels, listOfCats 

def denormalize(x_, mean, std):
    x = x_.clone()
    for i in range(3):
        x[i] = x[i]*std[i]+mean[i]
    x = to_pil_image(x)        
    return x

def train_val(model, params):
    num_epochs=params["num_epochs"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]
    
    loss_history={
        "train": [],
        "val": [],
    }
    
    metric_history={
        "train": [],
        "val": [],
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf')
    
    for epoch in range(num_epochs):
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,sanity_check,opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl,sanity_check)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
        
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)
        

        print("train loss: %.6f, dev loss: %.6f, training accuracy: %.2f, accuracy: %.2f" %(train_loss,val_loss,100*train_metric, 100*val_metric))
        print("-"*10) 
    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history

# get learning rate 
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    with torch.no_grad():
        metric_b = metrics_batch(output,target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b
    
def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    running_metric=0.0
    len_data = len(dataset_dl.dataset)
    for xb, yb in tqdm.notebook.tqdm(dataset_dl):
        xb=xb.to(device)
        yb=yb.to(device)
        output=model(xb)
        loss_b,metric_b=loss_batch(loss_func, output, yb, opt)
        running_loss+=loss_b
        
        if metric_b is not None:
            running_metric+=metric_b
        if sanity_check is True:
            break
    loss=running_loss/float(len_data)
    metric=running_metric/float(len_data)
    return loss, metric

def plot_loss(loss_hist, metric_hist):
    
    num_epochs= len(loss_hist["train"])
    plt.title("Train-Val Loss")
    plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
    plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()

    plt.title("Train-Val Accuracy")
    plt.plot(range(1,num_epochs+1), metric_hist["train"],label="train")
    plt.plot(range(1,num_epochs+1), metric_hist["val"],label="val")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()

def collate_fn_r3d_18(batch):
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.tensor(1)
    labels_tensor = torch.tensor(1)
    
    
    imgs_tensor = torch.stack(imgs_batch)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
  
    return imgs_tensor,labels_tensor





class VideoDataset(Dataset):
    def __init__(self, ids, labels, transform, label_dict, total_frames = 32, start_skip = 0, end_skip = 0):      
        self.transform = transform
        self.ids = ids
        self.labels = labels
        self.total_frames = total_frames
        self.start_skip = start_skip
        self.end_skip = end_skip
        self.labels_dict = label_dict

    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        path2imgs=glob.glob(self.ids[idx]+"/*.jpg")
        ##################################################################################################
        path2imgs = path2imgs[self.start_skip:self.total_frames-self.end_skip]
        ##################################################################################################
        label = self.labels_dict[self.labels[idx]]
        frames = []
        for p2i in path2imgs:
            frame = Image.open(p2i)
            frames.append(frame)
        
        seed = np.random.randint(1e9)        
        frames_tr = []
        for frame in frames:
            random.seed(seed)
            np.random.seed(seed)
            frame = self.transform(frame)
            frames_tr.append(frame)
            
        frames_tr = frames_tr
        if len(frames_tr)>0:
            frames_tr = torch.stack(frames_tr)
        return frames_tr, label


def start_training(path2data, hyper_params):

    num_classes = hyper_params['num_classes']
    batch_size = hyper_params['batch_size']
    num_epochs = hyper_params['num_epochs']
    learning_rate = hyper_params['learning_rate']
    test_ratio = hyper_params['test_ratio']
    start_skip = hyper_params['start_skip']
    end_skip = hyper_params['end_skip']
    total_frames = hyper_params['total_frames']
    mean = hyper_params['mean']
    std = hyper_params['std']
    h = hyper_params['h']
    w = hyper_params['w']


   
    path2ajpgs = path2data 
    all_vids, all_labels, catgs = get_vids(path2ajpgs) 

    labels_dict = {}
    ind = 0
    for uc in catgs:
        labels_dict[uc] = ind
        ind+=1


    unique_ids = [id_ for id_, label in zip(all_vids,all_labels) if labels_dict[label]<num_classes]
    unique_labels = [label for id_, label in zip(all_vids,all_labels) if labels_dict[label]<num_classes]

    
    sss = StratifiedShuffleSplit(n_splits=2, test_size=test_ratio, random_state=0)
    train_indx, test_indx = next(sss.split(unique_ids, unique_labels))

    train_ids = [unique_ids[ind] for ind in train_indx]
    train_labels = [unique_labels[ind] for ind in train_indx]

    test_ids = [unique_ids[ind] for ind in test_indx]
    test_labels = [unique_labels[ind] for ind in test_indx]




    train_transformer = transforms.Compose([
                transforms.RandomAffine(degrees=20, translate=(0.1,0.1)),    
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])     

    test_transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ]) 
            

    train_ds = VideoDataset(ids= train_ids, labels= train_labels, 
                            transform= train_transformer,label_dict=labels_dict, 
                            total_frames = total_frames, start_skip = start_skip, 
                            end_skip = end_skip)

    test_ds = VideoDataset(ids= test_ids, labels= test_labels,
                            transform= test_transformer, label_dict=labels_dict,
                            total_frames = total_frames, start_skip = start_skip,
                            end_skip = end_skip)


    train_dl = DataLoader(train_ds, batch_size= batch_size, shuffle=True, collate_fn= collate_fn_r3d_18)
    test_dl  = DataLoader(test_ds, batch_size= 2*batch_size, shuffle=False, collate_fn= collate_fn_r3d_18)          


    model = models.video.r3d_18(pretrained=True, progress=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = model.to(device)

    os.makedirs("models",  exist_ok=True)

    path2weights = "models/weights.pt"
    torch.save(model.state_dict(), path2weights)
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=5,verbose=1)

    params_train={
        "num_epochs": num_epochs,
        "optimizer": opt,
        "loss_func": loss_func,
        "train_dl": train_dl,
        "val_dl": test_dl,
        "sanity_check": False,
        "lr_scheduler": lr_scheduler,
        "path2weights": "./models/weights_3dcnn.pt",
        'verbose':0
        }

    model,loss_hist,metric_hist = train_val(model,params_train)
    
    return model, loss_hist, metric_hist






if __name__ == "__main__":
    hyper_params = {
       "num_epochs": 120,
        "learning_rate": 5e-5,
        "batch_size": 64,
        "h": 100,
        "w": 100,
        "mean": [0.5, 0.5, 0.5],
        "std":  [0.5, 0.5, 0.5],
        "total_frames": 32,
        "start_skip": 6,
        "end_skip": 8,
        "batch_size": 64,
        "num_classes": 40,
        "test_ratio": 0.2
    }
    path2data = sys.argv[1]
    model, loss_hist, metric_hist = start_training(path2data, hyper_params)
