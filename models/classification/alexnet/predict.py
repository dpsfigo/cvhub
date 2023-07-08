'''
Author: dpsfigo
Date: 2023-07-08 10:44:02
LastEditors: dpsfigo
LastEditTime: 2023-07-08 16:13:47
Description: 请填写简介
'''
import argparse
# from datasets.dataset import Dataset
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets, utils

from alexnet import AlexNet
from tqdm import tqdm
import time
import os
import json
from PIL import Image
import cv2

parser = argparse.ArgumentParser(description='Code to train the model')
parser.add_argument('--checkpoint_path', help='Resume model from this checkpoint', default='./checkpoints/checkpoint_step000000001.pth', type=str)
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

def test_model(device, model, img):
    # Move data to CUDA device
    x = img.float().to(device)
    output = torch.squeeze(model(x)).cpu()
    predict = torch.softmax(output, dim=0)
    predict_class = torch.argmax(predict).numpy()
    score = predict[predict_class].detach().numpy()
    return (predict_class,score)


def _load(checkpoint_path):
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(model, path):
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	return model.eval()

def pretict(img):
    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)


    # 读取label文件
    cur_folder_path = os.path.dirname(os.path.realpath(__file__))
    label_indices_path = os.path.join(cur_folder_path, "class_indices.json")
    with open(label_indices_path, "r") as f:
        class_indicts = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet(num_classes=5).to(device)
    model = load_model(model, args.checkpoint_path)
    # model.load_state_dict(torch.load(args.checkpoint_path))
    
    model.eval()
    ret = test_model(device, model, img)
    print(ret)

if __name__ == "__main__":
    filename = './data/flower_data/train/daisy/5547758_eea9edfd54_n.jpg'
    img = cv2.imread(filename)
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 
    ret = pretict(img)