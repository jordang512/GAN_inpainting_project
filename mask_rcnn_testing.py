from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from PIL import Image
import sys

net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)
threshold = 0.9
x, orig_img = data.transforms.presets.rcnn.load_test("{}".format(sys.argv[1]))

ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

# paint segmentation mask on images directly
width, height = orig_img.shape[1], orig_img.shape[0]
masks = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
orig_img = utils.viz.plot_mask(orig_img, masks)

filtered_ids = [x for x in ids if x > threshold]
class_names = net.classes

print(set([class_names[item[0]] for item in filtered_ids]))

# identical to Faster RCNN object detection
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                         class_names=net.classes, ax=ax)
plt.show()