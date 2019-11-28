import os
import torch
import torchvision
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict as od

def visualize(feat_map, output_path, vmax, vmin):
	sns.set()
	ax = sns.heatmap(feat_map, vmax=vmax, vmin=vmin, square=True, cbar=False)
	ax.set_xticks([])
	ax.set_yticks([])

	plt.savefig(output_path, papertype='letter', pad_inches=0, bbox_inches="tight")
	plt.clf()

def parse_single_image(img_path):
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	normalize = torchvision.transforms.Normalize(mean=mean,std=std)
	t = torchvision.transforms.Compose([
			torchvision.transforms.Resize(256),
			torchvision.transforms.CenterCrop(224),
			torchvision.transforms.ToTensor(),
			normalize,
		]) 
	with open(img_path, 'rb') as f:
		img = Image.open(f)
		img = img.convert('RGB')
		img = t(img)
	return img.unsqueeze(0)
 
class PathManager(object):
	def __init__(self, root, relative_path, json_path):
		self.root = root
		self.relative_path = relative_path
		self.output_path = os.path.join(self.root, self.relative_path)

		self.json_path = os.path.join(self.root, json_path)
		self.dic = od()

	def gen(self, i_layer, i_filter, fname):
		self.abs = os.path.join(self.output_path, str(i_layer), str(i_filter), fname)
		self.rel = os.path.join(self.relative_path, str(i_layer), str(i_filter), fname)
		if not os.path.exists(os.path.dirname(self.abs)):
			os.makedirs(os.path.dirname(self.abs))

		if i_layer not in dic.keys():
			dic[i_layer] = []
		dic[i_layer].append(self.rel)

	def save_json(self):
		if not os.path.exists(os.path.dirname(self.json_path)):
			os.makedirs(os.path.dirname(args.json_path))
		with open(self.json_path, 'w') as f:
			json.dump(dic, f)