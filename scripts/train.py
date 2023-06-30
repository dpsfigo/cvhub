import argparse
from datasets import Dataset
from hparams import hparams
import torch
from torch.utils import data as data_utils
from torch import optim

from models.classification.alexnet import AlenNet


parser = argparse.ArgumentParser(description='Code to train the model')
parser.add_argument("--data_root", help="Root folder of the dataset", default="./data/fp/preprocessed_288/", type=str)
parser.add_argument("--file_list", help="Root folder of the filelist", default="./data/fp/preprocessed_288/", type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default="./checkpoints/", type=str)
parser.add_argument('--checkpoint_path', help='Resume model from this checkpoint', default=None, type=str)
args = parser.parse_args()
def load_checkpoint_my_syncnet(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
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


def load_checkpoint_pretrained_wav2lip(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        # new_s[k.replace('module.', '')] = v
        new_s['module.' + k] = v
        # new_s[k] = v  # ================================
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
if __name__ == "__main__":
    torch.manual_seed(3407)
    train_dataset = Dataset(args.data_root, "train.txt")
    val_dataset = Dataset(args.data_root, "val.txt")

    train_data_loader = data_utils(train_dataset, hparams.batch_size, shuffle=True)
    val_data_loader = data_utils(val_dataset, hparams.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlenNet(num_classes=2)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hparams.initial)

    # if args.checkpoint_path is not None:
    #     load_checkpoint_pretrained_wav2lip(args.checkpoint_path, model, optimizer, reset_optimizer=False)
        
    # load_checkpoint_my_syncnet(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)
    log_path = '{}/wav2lip_train'.format(args.checkpoint_dir)
    logger = logging_util.LoggingUtil(log_path).getLogger(__name__)
    train(device, model, train_data_loader, val_data_loader, optimizer, logger, che)
