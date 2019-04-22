from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from PIL import Image
import sys

net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)
threshold = 0.5
x, orig_img = data.transforms.presets.rcnn.load_test("{}".format(sys.argv[1]))

ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]


# x is index, int(y[0]) is category id
filtered_ids = [(x,int(y[0])) for x,y in enumerate(ids) if scores[x] > threshold]
class_names = net.classes
print("I found these categories: {}".format(set([class_names[item[1]] for item in filtered_ids])))

# Idea: for all identified objects of category other than specified,
# set its score to 0. Then gluoncv will throw it out.

# Prune scores
scores = [scores[item[0]] for item in filtered_ids]
masks = [masks[item[0]] for item in filtered_ids]
bboxes = [bboxes[item[0]] for item in filtered_ids]

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