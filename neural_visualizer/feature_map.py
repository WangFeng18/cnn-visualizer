import os
import torch
import argparse
from alexnet import alexnet
import torchvision
from tqdm import tqdm
from PIL import Image
import seaborn as sns
from collections import OrderedDict as od
import matplotlib.pyplot as plt
import json
import shutil
from util import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('pretrain_path', type=str, default='none')
	parser.add_argument('test_image', type=str, default='none')
	parser.add_argument('name', type=str, default='none')

	# parameters no need to modify
	parser.add_argument('--root', default='../web_platform/src/', type=str)
	parser.add_argument('--relative_path', default='imgs/{}/feature_maps/', type=str)
	parser.add_argument('--json_path', default='paths/{}/feature_map.json', type=str)
	parser.add_argument('--source_image_path', default='imgs/{}/source_image/', type=str)
	args = parser.parse_args()

	args.relative_path = args.relative_path.format(args.name)
	args.source_image_path = args.source_image_path.format(args.name)
	args.json_path = args.json_path.format(args.name)

	args.source_image_path = os.path.join(args.root, args.source_image_path)
	if not os.path.exists(args.source_image_path):
		os.makedirs(args.source_image_path)
	shutil.copy(args.test_image, os.path.join(args.source_image_path, os.path.basename(args.test_image)))

	path_manager = PathManager(args.root, args.relative_path, args.json_path)
	network = alexnet(pretrained=args.pretrain_path).to(args.device)
	args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	img = parse_single_image(args.test_image).to(args.device)
	feats = network.forward_convs(img)

	for key, val in feats.items():
		feat = val.detach().numpy()
		vmax = feat.max()
		vmin = feat.min()
		for i in tqdm(range(feat.shape[1])):
			path_manager.gen(key, i, 'last.svg')
			visualize(feat[0][i], path_manager.abs, vmax, vmin)
	path_manager.save_json()