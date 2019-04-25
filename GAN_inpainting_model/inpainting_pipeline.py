from test import test_model
from mask_rcnn_testing import get_masks
from gluoncv import model_zoo, data, utils
from scipy.misc import imresize
from scipy.signal import convolve2d
import numpy as np
import cv2

def expand_masks(masks):
	maxval = np.amax(masks)
	kernel = np.ones((11, 11))
	expanded = convolve2d(masks, kernel, mode='same')
	return (expanded >= maxval) * 255

def erase_masks(fpath):
	x, im = downsize_file(fpath)
	masks = get_masks(x, im)
	if masks.ndim == 3:
		compiled_mask = np.amax(masks, axis=0)
	else:
		compiled_mask = masks
	expand_masks(compiled_mask) #convolve with a 11 x 11 kernel to expand masks for inpainting
	compiled_mask = np.array([compiled_mask for _ in range(3)])
	compiled_mask = np.moveaxis(compiled_mask, 0, -1)
	#compiled_mask = compiled_mask * 255. / np.amax(compiled_mask)
	compiled_mask = compiled_mask.astype(int)

	print(compiled_mask.shape)
	print(im.shape)
	cv2.imwrite("mask.png", compiled_mask)
	test_model(im, compiled_mask)

def downsize_file(fpath):
	x, orig_img = data.transforms.presets.rcnn.load_test("{}".format(fpath))
	return x, orig_img

erase_masks('test.jpg')


