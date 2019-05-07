import imageio
import mxnet
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
from PIL import Image
import sys
import numpy as np
from scipy.signal import convolve2d
# from inpainting_pipeline import expand_masks, erase_masks
import argparse
import cv2
import tensorflow as tf
import neuralgym as ng
from inpaint_model import InpaintCAModel

fname = 'baseballshort.gif'

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')


def expand_masks(masks, ksize):
	kernel = np.ones((ksize, ksize))
	expanded = convolve2d(masks, kernel, mode='same')
	return (expanded > 0) * 255

def erase_masks(fpath):
	x, im = downsize_file(fpath)
	masks = get_masks(x, im)
	if masks.ndim == 3:
		compiled_mask = np.amax(masks, axis=0)
	else:
		compiled_mask = masks
	compiled_mask = expand_masks(compiled_mask, 21) #convolve with a 11 x 11 kernel to expand masks for inpainting
	compiled_mask = np.array([compiled_mask for _ in range(3)])
	compiled_mask = np.moveaxis(compiled_mask, 0, -1)
	compiled_mask = compiled_mask * 255. / np.amax(compiled_mask)
	compiled_mask = compiled_mask.astype(int)

	print(compiled_mask.shape)
	print(im.shape)
	# cv2.imwrite("mask.png", compiled_mask)
	test_model(im, compiled_mask)




def test_model(image, mask, output_dir='output_images/output.png', checkpoint_dir='model_logs/release_places2_256'):
    ng.get_gpus(1)
    model = InpaintCAModel()

    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        # cv2.imwrite(output_dir, result[0][:, :, ::-1])
        # plt.imsave('out.jpg', result[0][:, :, ::-1])
        return result[0]





def get_masks(x, orig_img, net, class_to_remove):
    threshold = 0.5
    #x, orig_img = data.transforms.presets.rcnn.transform_test(image)
    ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

    # x is index, int(y[0]) is category id
    filtered_ids = np.array([(x,int(y[0])) for x,y in enumerate(ids) if scores[x] > threshold])


    # Prune scores, masks, boxes, and ids by selection
    # It's important to define these as np.array's
    scores = np.array([scores[item[0]] for item in filtered_ids if item[1]==class_to_remove])
    masks = np.array([masks[item[0]] for item in filtered_ids if item[1]==class_to_remove])
    bboxes = np.array([bboxes[item[0]] for item in filtered_ids if item[1]==class_to_remove])
    ids = np.array([item[1] for item in filtered_ids if item[1]==class_to_remove])

    if not masks:
        return []

    width, height = orig_img.shape[1], orig_img.shape[0]
    masks = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
    return masks




def process_video(fname):
    vid = imageio.get_reader(fname, 'ffmpeg')
    frames = []
    for idx, f in enumerate(vid):
        im = vid.get_data(idx)
        frame, orig_im = data.transforms.presets.ssd.transform_test(mxnet.nd.array(im),600)
        frames.append((frame, orig_im))

    finished_frames = []
    net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)
    print([(x,y) for (x,y) in enumerate(net.classes)])
    class_to_remove = input("Please enter a class index: ")

    print(len(frames))

    for count, frame in enumerate(frames):
        painted = process_frame(frame, net, class_to_remove)
        finished_frames.append(painted)
        print("Finished frame {}".format(count))
    imageio.mimsave('outgif.gif',frames)


def process_frame(frame, net, class_to_remove):
    masks = get_masks(frame[0], frame[1], net, class_to_remove)
    if masks == []:
        return frame[1]
    if masks.ndim == 3:
        compiled_mask = np.amax(masks, axis=0)
    else:
        compiled_mask = masks
    compiled_mask = expand_masks(compiled_mask, 21) #convolve with a 11 x 11 kernel to expand masks for inpainting
    compiled_mask = np.array([compiled_mask for _ in range(3)])
    compiled_mask = np.moveaxis(compiled_mask, 0, -1)
    compiled_mask = compiled_mask * 255. / np.amax(compiled_mask)
    compiled_mask = compiled_mask.astype(int)

    print(compiled_mask.shape)
    print(frame[1].shape)
    cv2.imwrite("mask.png", compiled_mask)
    return test_model(frame[1], compiled_mask)

process_video(fname)