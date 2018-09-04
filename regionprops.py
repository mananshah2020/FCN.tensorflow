# Import libraries
import cv2
import numpy as np
import scipy
import sklearn
import skimage
from skimage import measure, io,feature
from skimage.morphology import reconstruction
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
import functools
import os

def main():
	path = None
	labels = {}
	tilesets = generate_tilesets(path)
	properties = get_properties(tilesets, labels)
	print(properties)

def walk(path):
	walk = sorted(list(os.walk(path)))
	columns = sorted(walk[0][1])
	columns = [int(columns[x]) for x in range(len(columns))]
	del walk[0]
	col_paths = [walk[x][0] for x in range(len(walk))]
	images = [sorted(walk[x][2]) for x in range(len(walk))]
	for col in range(len(images)):
		for img in range(len(images[col])):
			images[col][img] = int(images[col][img].split('.')[0])
	return columns, col_paths, images

def neighboring_columns(target_column, columns):
	left_col = target_column - 1
	right_col = target_column + 1
	if left_col not in columns:
		left_col = None
	if right_col not in columns:
		right_col = None
	return left_col, right_col

def group_tileset_paths(target_column, target, columns, col_paths, col_images):
	left_col, right_col = neighboring_columns(target_column,columns)
	target_col_path = col_paths[columns.index(target_column)]
	left_col_path = None
	right_col_path = None
	if right_col != None:
		right_col_path = col_paths[columns.index(right_col)]
	if left_col != None:
		left_col_path = col_paths[columns.index(left_col)]
	col_names = [left_col, target_column, right_col]
	col_paths = [left_col_path, target_col_path, right_col_path]
	
	t_m_b = [target - 1, target, target + 1]

	tileset_paths = []
	for col in range(3):
		if col_names[col] != None:
			col_list = col_images[columns.index(col_names[col])]
			for tile in t_m_b:
				if tile in col_list:
					im_path = col_paths[col] + '/{}.png'.format(tile)
					tileset_paths.append(im_path)
				else:
					tileset_paths.append(None)
		else:
			tileset_paths.extend((None,None,None))

	return tileset_paths

def combine(tiles):
	left = np.concatenate((tiles[0], tiles[1], tiles[2]), axis=0)
	mid = np.concatenate((tiles[3], tiles[4], tiles[5]), axis=0)
	right = np.concatenate((tiles[6], tiles[7], tiles[8]), axis=0)
	combined = np.concatenate((left, mid, right), axis=1)
	return combined

def create_tileset(tileset_paths):
	tiles = []
	for tile in range(len(tileset_paths)):
		if tileset_paths[tile] != None:
			tiles.append(cv2.imread(tileset_paths[tile]))
			tiles[tile] = tiles[tile][1:255,1:255]
			tiles[tile] = cv2.cvtColor(tiles[tile], cv2.COLOR_BGR2GRAY)
		else:
			tiles.append(np.zeros((254,254), dtype='uint8'))
	tileset = combine(tiles)
	return tileset

def label(img, input_label):
	img[img[:,:] != input_label] = 0
	labelled_img, num_labels = measure.label(img, return_num=True)
	return labelled_img, num_labels

def get_discrete_properties(labelled_img, num_labels):
	dims = labelled_img.shape
	discrete_properties = measure.regionprops(labelled_img)
	inside = []
	for l in range(num_labels):
		centroid = discrete_properties[l]['centroid']
		if centroid[0] > int(dims[0]/3) and centroid[0] < 2*int(dims[0]/3):
			if centroid[1] > int(dims[1]/3) and centroid[1] < 2*int(dims[1]/3):
				inside.append(l)
	for x in range(len(inside)):
		inside[x] = discrete_properties[inside[x]]
	return inside

def get_continuous_properties(labelled_img):
	dims = labelled_img.shape
	target = labelled_img[int(dims[0]/3):2*int(dims[0]/3), int(dims[1]/3):2*int(dims[1]/3)]
	labelled_target = measure.label(target)
	cont_properties = measure.regionprops(labelled_target)
	return cont_properties

def generate_tilesets(path):
	columns, col_paths, images_list = walk(path)
	tileset_paths_dict = {}
	for col in range(len(columns)):
		for i in range(len(images_list[col])):
			key = path + '/' + str(columns[col]) + '/' + str(images_list[col][i]) + '.png'
			tileset_paths_dict[key] = group_tileset_paths(columns[col],
				images_list[col][i], columns, col_paths, images_list)
	return tileset_paths_dict

def get_properties(tileset_paths_dict, label_dict):
	target_properties = {}
	for target in tileset_paths_dict.keys():
		properties = {}
		for l in label_dict.keys():
			labelled_img, num_labels = label(create_tileset(tileset_paths_dict[target]),
				label_dict[l][0])
			if label_dict[l][1] == 'discrete':
				properties[l] = get_discrete_properties(labelled_img, num_labels)
			else:
				properties[l] = get_continuous_properties(labelled_img)
		target_properties[target] = properties
	return target_properties

if __name__ == "__main__":
	main()