import os
import torch
import argparse
from alexnet import alexnet
import torch.nn as nn
import torchvision
from tqdm import tqdm
from PIL import Image
import seaborn as sns
from collections import OrderedDict as od
import matplotlib.pyplot as plt
import json
import cv2
from util import *

def generate_activated_image(network, layer, filter_number, args):
	img = nn.Parameter(torch.rand(1,3,224,224), requires_grad=True).to(args.device)
	pbar = tqdm(range(args.iters))
	optimizer = torch.optim.Adam([img], lr=args.lr, weight_decay=1e-6)
	for i in pbar:
		feats = network.forward_convs(img)
		loss = -feats[layer][:, i_filter, ...].mean()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		pbar.set_description('Layer:{},Filter:{}'.format(layer, i_filter))
		pbar.set_postfix(info='loss {:.4f}'.format(-loss.item()))
	return img
   
if __name__ == '__main__':
	dic = od()
	parser = argparse.ArgumentParser()
	parser.add_argument('pretrain_path', default='none', type=str)
	parser.add_argument('name', type=str, default='none')

	# parameters no need to modify
	parser.add_argument('--iters', default=30, type=int)
	parser.add_argument('--lr', default=1000.0, type=float)
	parser.add_argument('--root', default='../web_platform/src/', type=str)
	parser.add_argument('--layer', default='none', type=str)
	parser.add_argument('--filter_number', default='none', type=str)
	parser.add_argument('--relative_path', default='imgs/{}/filter_vis/', type=str)
	parser.add_argument('--json_path', default='paths/{}/filters_vis.json', type=str)
	args = parser.parse_args()
	args.relative_path = args.relative_path.format(args.name)
	args.json_path = args.json_path.format(args.name)

	path_manager = PathManager(args.root, args.relative_path, args.json_path)

	args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	network = alexnet(pretrained=args.pretrain_path).to(args.device)
	for para in network.parameters():
		para.requires_grad=False

	img = nn.Parameter(torch.randn(1,3,224,224), requires_grad=True).to(args.device)
	test_output = network.forward_convs(img)
	
	if args.layer != 'none':
		all_layers = [args.layer]
	else:
		all_layers = test_output.keys()
	if args.filter_number != 'none':
		all_filters = [int(args.filter_number)]
	else:
		all_filters = range(test_output[layer].shape[1])

	for layer in all_layers:
		for i_filter in all_filters:
			img = generate_activated_image(network, layer, i_filter, args)
			a = img[0].detach().numpy()
			path_manager.gen(layer, i_filter, 'last.jpg')
			cv2.imwrite(path_manager.abs, a.transpose(1,2,0)[...,::-1])
			if layer not in dic.keys():
				dic[layer] = []
			dic[layer].append(os.path.join(relative_path, 'last.jpg'))
	if not os.path.exists(os.path.dirname(args.json_path)):
		os.makedirs(os.path.dirname(args.json_path))
	with open(args.json_path, 'w') as f:
		if args.layer == 'none':
			json.dump(dic, f)

