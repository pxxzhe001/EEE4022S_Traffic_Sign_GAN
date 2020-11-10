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
	# name = glob(r'C:\Users\Kevin\Dropbox\UCTSHITS\4022S\Code\Kevin\ShapeDetection\data\*\\')
	path = r'D:\Kevin\University\EEE4022S\ScamResult\\'
	path = path[:-1]
	# size = int(len(name)/2)
	testA_list = []
	newPath = path + 'testData' + '\\'
	testA_list = load_images(newPath, size=(256,256))
	# print(sign_list)
	# print(len(sign_list))
	test_array = asarray(testA_list)
	# load dataset
	print('Loaded: ', test_array.shape)
	# save as compressed numpy array
	filename = 'testS.npz'
	savez_compressed(filename, test_array)
	print('Saved dataset: ', filename)


if __name__ == '__main__':
	main()