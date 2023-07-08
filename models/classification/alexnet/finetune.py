'''
Author: dpsfigo
Date: 2023-07-08 09:33:59
LastEditors: dpsfigo
LastEditTime: 2023-07-08 15:37:16
Description: 请填写简介
'''
import argparse
# from datasets.dataset import Dataset
from hparams import hparams
import torch
from torch.utils import data as data_utils
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, datasets, utils
import torch.nn as nn

from alexnet import AlexNet
import torchvision.models as models
# from utils import logging_util
from os.path import join
from tqdm import tqdm
import time
import os
import json

parser = argparse.ArgumentParser(description='Code to train the model')
parser.add_argument("--data_root", help="Root folder of the dataset", default="./data/flow_data/", type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default="./checkpoints/", type=str)
parser.add_argument('--checkpoint_path', help='Resume model from this checkpoint', default=None, type=str)
args = parser.parse_args()

global_step = 0
global_epoch = 0

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        #outputs = torch.sigmoid(output)
        outputs = F.softmax(output, 1)
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred)).reshape(-1).float().sum()# / pred.shape[1]
        return correct

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        # new_s[k.replace('module.', '')] = v
        new_s['module.' + k] = v  # ================================
        # new_s[k] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

def train(device, model, train_data_loader, train_dataset_len, test_data_loader, test_dataset_len, optimizer, loss_func,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
 
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_loss = 0., 0.
        model.train()
        
        for step, (x,  gt) in enumerate(train_data_loader):
            if step == 0:
                continue
            
            # Move data to CUDA device
            x = x.float().to(device)
            gt = gt.to(device)

            pred = model(x)
            loss = loss_func(pred, gt)
            optimizer.zero_grad()


            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step

            if global_step == 1 or global_step % checkpoint_interval == 0:
                # save_checkpoint(
                #     model, optimizer, global_step, checkpoint_dir, global_epoch)
                pass

            if global_step == 1 or global_step % hparams.checkpoint_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, test_dataset_len, global_step, device, model, checkpoint_dir, loss_func)
            # print("loss: ", loss.item())
            # prog_bar.set_description("loss: ".format(loss.item()))

        global_epoch += 1
        

def eval_model(test_data_loader, test_dataset_len, global_step, device, model, checkpoint_dir, loss_func):
    eval_steps = 700
    losses = []
    step = 0
    acc = 0.0
    for x, gt in test_data_loader:
        step += 1
        model.eval()

        # Move data to CUDA device
        x = x.float().to(device)
        gt = gt.to(device)
        pred = model(x)

        loss = loss_func(pred, gt)

        losses.append(loss.item())
        acc += accuracy(pred, gt).item()

        # if step > eval_steps: 
        #     averaged_loss = sum(losses) / len(losses)

        #     print('loss: {}'.format(averaged_loss))

        #     return averaged_loss
    acc = acc/test_dataset_len
    print("acc ",acc)

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

if __name__ == "__main__":
    torch.manual_seed(3407)
    data_transform = {
        "train":transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val":transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
    # image_path = os.path.join(data_root, "data", "flower_data")
    image_path = args.data_root

    # train_dataset = Dataset(args.data_root, args.file_list, "trainval.txt")
    # val_dataset = Dataset(args.data_root, args.file_list, "test.txt")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    train_num = len(train_dataset)

    label_list = train_dataset.class_to_idx
    class_dict = dict((value, key)for key, value in label_list.items())
    label_json = json.dumps(class_dict, indent=4)
    with open("class_indices.json", "w") as f:
        f.write(label_json)
    
    batch_size = hparams.batch_size
    number_worker = hparams.number_worker
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers= number_worker)
    
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    val_num = len(val_dataset)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=batch_size, shuffle=False,
                                               num_workers= number_worker)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # net = AlexNet(num_classes=5)
    net = models.alexnet(pretrained=True)
    #冻结参数梯度
    featue_extract = True
    set_parameter_requires_grad(net, featue_extract)
    #修改模型
    num_ftrs = net.classifier[6].in_features
    net.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=len(label_list), bias=True)
    optimizer = optim.Adam(net.parameters(), lr=hparams.initial_learning_rate)

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, net, optimizer, reset_optimizer=False)
        
    log_path = '{}/train'.format(args.checkpoint_dir)
    # logger = logging_util.LoggingUtil(log_path).getLogger(__name__)
    loss_func = torch.nn.CrossEntropyLoss()
    model = net.to(device)
    train(device, model, train_data_loader, train_num, val_data_loader, val_num, optimizer, loss_func,
          checkpoint_dir=args.checkpoint_dir,
          checkpoint_interval=hparams.checkpoint_interval,
          nepochs=hparams.nepochs)
