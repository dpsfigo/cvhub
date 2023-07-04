'''
Author: dpsfigo
Date: 2023-06-27 15:40:01
LastEditors: dpsfigo
LastEditTime: 2023-07-04 19:49:25
Description: 请填写简介
'''
import argparse
from datasets.dataset import Dataset
from scripts.hparams import hparams
import torch
from torch.utils import data as data_utils
from torch import optim

from models.classification.alexnet import AlenNet
from utils import logging_util
from os.path import join
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Code to train the model')
parser.add_argument("--data_root", help="Root folder of the dataset", default="./data/oxford-iiit-pet/images/", type=str)
parser.add_argument("--file_list", help="Root folder of the filelist", default="./data/oxford-iiit-pet/annotations/", type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default="./checkpoints/", type=str)
parser.add_argument('--checkpoint_path', help='Resume model from this checkpoint', default=None, type=str)
args = parser.parse_args()

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

def train(device, model, train_data_loader, test_data_loader, optimizer, loss_func,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
 
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_l1_loss = 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x,  gt) in prog_bar:
            model.train()
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device)
            gt = gt.to(device)

            pred = model(x)

            loss = loss_func(pred, gt)

            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step == 1 or global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir, loss_func)

        global_epoch += 1
        

def eval_model(test_data_loader, global_step, device, model, checkpoint_dir, loss_func):
    eval_steps = 700
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    step = 0
    while 1:
        for x, gt in test_data_loader:
            step += 1
            model.eval()

            # Move data to CUDA device
            x = x.to(device)
            gt = gt.to(device)

            pred = model(x)

            loss = loss_func(pred, gt)

            losses.append(loss.item())

            if step > eval_steps: 
                averaged_loss = sum(losses) / len(losses)

                print('loss: {}'.format(averaged_loss))

                return averaged_loss

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

if __name__ == "__main__":
    torch.manual_seed(3407)
    train_dataset = Dataset(args.file_list, "trainval.txt")
    val_dataset = Dataset(args.file_list, "test.txt")

    train_data_loader = data_utils.DataLoader(train_dataset, hparams.batch_size, shuffle=True)
    val_data_loader = data_utils.DataLoader(val_dataset, hparams.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlenNet(num_classes=2)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hparams.initial_learning_rate)

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)
        
    log_path = '{}/train'.format(args.checkpoint_dir)
    logger = logging_util.LoggingUtil(log_path).getLogger(__name__)
    loss_func = torch.nn.CrossEntropyLoss()
    train(device, model, train_data_loader, val_data_loader, optimizer, logger, loss_func, 
          checkpoint_dir=args.checkpoint_dir,
          checkpoint_interval=hparams.checkpoint_interval,
          nepochs=hparams.nepochs)
