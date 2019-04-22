from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from PIL import Image
import sys
import numpy as np

net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)
threshold = 0.5
x, orig_img = data.transforms.presets.rcnn.load_test("{}".format(sys.argv[1]))
ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

# x is index, int(y[0]) is category id
filtered_ids = np.array([(x,int(y[0])) for x,y in enumerate(ids) if scores[x] > threshold])
class_names = net.classes

# Prompt user to select a category
print("I found these categories: ")
unique_classes = list(set([class_names[item[1]] for item in filtered_ids]))
for idx,item in enumerate(unique_classes):
    print("{}: {}".format(idx,item))
print("Please select one category by entering the number next to it")
# To get the category id, convert input->class->index of class
selection = net.classes.index(unique_classes[int(input("My choice is: "))])

# Prune scores, masks, boxes, and ids by selection
# It's important to define these as np.array's
scores = np.array([scores[item[0]] for item in filtered_ids if item[1]==selection])
masks = np.array([masks[item[0]] for item in filtered_ids if item[1]==selection])
bboxes = np.array([bboxes[item[0]] for item in filtered_ids if item[1]==selection])
ids = np.array([item[1] for item in filtered_ids if item[1]==selection])

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