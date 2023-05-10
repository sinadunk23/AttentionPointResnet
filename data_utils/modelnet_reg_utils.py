import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import h5py
import subprocess
import shlex
import json
import glob
import transform_functions
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    #point = point[]
    return centroids.astype(np.int32)

def download_modelnet40():
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	if not os.path.exists(DATA_DIR):
		os.mkdir(DATA_DIR)
	if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
		www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
		zipfile = os.path.basename(www)
		www += ' --no-check-certificate'
		os.system('wget %s; unzip %s' % (www, zipfile))
		os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
		os.system('rm %s' % (zipfile))

def load_data(train):
	if train: partition = 'train'
	else: partition = 'test'
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	all_data = []
	all_label = []
	for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
		f = h5py.File(h5_name)
		data = f['data'][:].astype('float32')
		label = f['label'][:].astype('int64')
		f.close()
		all_data.append(data)
		all_label.append(label)
	all_data = np.concatenate(all_data, axis=0)
	all_label = np.concatenate(all_label, axis=0)
	return all_data, all_label

def deg_to_rad(deg):
	return np.pi / 180 * deg

class ModelNet40Data(Dataset):
	def __init__(
		self,
		train=True,
		num_points=1024,
		download=True,
		randomize_data=False
	):
		super(ModelNet40Data, self).__init__()
		if download: download_modelnet40()
		self.data, self.labels = load_data(train)
		if not train: self.shapes = self.read_classes_ModelNet40()
		self.num_points = num_points
		self.randomize_data = randomize_data

	def __getitem__(self, idx):
		if self.randomize_data: current_points = self.randomize(idx)
		else: current_points = self.data[idx].copy()

		current_points = torch.from_numpy(current_points).float()
		label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

		return current_points, label

	def __len__(self):
		return self.data.shape[0]

	def randomize(self, idx):
		pt_idxs = farthest_point_sample(self.data[idx], self.num_points)
		return self.data[idx, pt_idxs].copy()

	def get_shape(self, label):
		return self.shapes[label]

	def read_classes_ModelNet40(self):
		BASE_DIR = os.path.dirname(os.path.abspath(__file__))
		DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
		file = open(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'), 'r')
		shape_names = file.read()
		shape_names = np.array(shape_names.split('\n')[:-1])
		return shape_names

class RegistrationData(Dataset):
	def __init__(self, data_class=ModelNet40Data(), is_testing=False, angle_range=90, translation_range=1, add_noise=False, shuffle_points=False):
		super(RegistrationData, self).__init__()
		self.is_testing = is_testing
		self.set_class(data_class)
		self.transforms = transform_functions.PCRNetTransform(angle_range=angle_range, translation_range=translation_range, add_noise=add_noise, shuffle_points=shuffle_points)

	def __len__(self):
		return len(self.data_class)

	def set_class(self, data_class):
		self.data_class = data_class

	def __getitem__(self, index):
		template, label = self.data_class[index]
		source = self.transforms(template)
		igt = self.transforms.igt
		if self.is_testing:
			return template, source, igt, self.transforms.igt_rotation, self.transforms.igt_translation
		else:
			return template, source, igt, self.transforms.igt_rotation, self.transforms.igt_translation