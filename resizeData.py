from glob import glob
from os import listdir
import numpy as np
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

def load_images(path, size=(256,256)):
	img_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		print(filename)
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		img_list.append(pixels)
	return img_list


def main():
	name = glob(r'D:\Kevin\University\EEE4022S\ScamResult\data2\*\\')
	path = r'D:\Kevin\University\EEE4022S\ScamResult\data2\\'
	path = path[:-1]
	size = int(len(name)/2)
	sign_list = []
	rsign_list = []
	for i in range(2):
		for k in range(size):
			if (i == 0):
				newPath = path + 'data' + str(k+1) + '\\'
				print(newPath)
				temp_list = load_images(newPath, size=(256,256))
				sign_list = sign_list + temp_list
			else:
				newPath = path + 'rSignData' + str(k+1) + '\\'
				print(newPath)
				temp_list = load_images(newPath, size=(256,256))
				rsign_list = rsign_list + temp_list
		
	# print(sign_list)
	# print(len(sign_list))
	sign_array = asarray(sign_list)
	rsign_array = asarray(rsign_list)
	# load dataset
	print('Loaded: ', sign_array.shape, rsign_array.shape)
	# save as compressed numpy array
	filename = 'data100.npz'
	savez_compressed(filename, sign_array, rsign_array)
	print('Saved dataset: ', filename)


if __name__ == '__main__':
	main()