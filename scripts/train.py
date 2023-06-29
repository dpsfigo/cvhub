import argparse
import torch

parser = argparse.ArgumentParser(description='Code to train the model')
parser.add_argument("--data_root", help="Root folder of the dataset", default="./data/fp/preprocessed_288/", type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default="./checkpoints/", type=str)
parser.add_argument('--checkpoint_path', help='Resume model from this checkpoint', default=None, type=str)
args = parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(3407)