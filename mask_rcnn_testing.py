from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from PIL import Image

net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)
x, orig_img = data.transforms.presets.rcnn.load_test("y/test.jpg")

ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

# paint segmentation mask on images directly
width, height = orig_img.shape[1], orig_img.shape[0]
masks = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
orig_img = utils.viz.plot_mask(orig_img, masks)

# identical to Faster RCNN object detection
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                         class_names=net.classes, ax=ax)
plt.show()