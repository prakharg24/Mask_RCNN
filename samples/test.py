import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import time
from utils import read_json, dump_json

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
print(COCO_MODEL_PATH)

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
		utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
		# Set batch size to 1 since we'll be running inference on
		# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1
		# DETECTION_MIN_CONFIDENCE = 0.1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
							 'bus', 'train', 'truck', 'boat', 'traffic light',
							 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
							 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
							 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
							 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
							 'kite', 'baseball bat', 'baseball glove', 'skateboard',
							 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
							 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
							 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
							 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
							 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
							 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
							 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
							 'teddy bear', 'hair drier', 'toothbrush']

img_path = sys.argv[1]

compl_data = read_json(img_path)
iter_arr = compl_data['data']

result_data = []

img_batch = []
img_name = []

for idx, data_point in enumerate(iter_arr):
	filepath = data_point['image_name']
	print(filepath)

	image = skimage.io.imread("../" + filepath)
	st = time.time()

	results = model.detect([image], verbose=1)

	# print(results[0]['rois'])
	# print(results[0]['class_ids'])
	# print(results[0]['scores'])

	temp_dict = {}
	temp_dict['image_name'] = filepath

	all_dets = []
	for reg, cid, scr in zip(results[0]['rois'], results[0]['class_ids'], results[0]['scores']):
		if(cid==17):
			print(reg, 'dog', scr)
			all_dets.append([int(reg[1]), int(reg[0]), int(reg[3]), int(reg[2]), 'dog', int(100*scr)])
		if(cid==20):
			print(reg, 'cow', scr)
			all_dets.append([int(reg[1]), int(reg[0]), int(reg[3]), int(reg[2]), 'cow', int(100*scr)])
	
	temp_dict['pred_anno'] = all_dets
	result_data.append(temp_dict)

	if(idx%16==0):
		eval_data = {}
		eval_data['data'] = result_data
		dump_json('eval_data.json', eval_data)		
	print('Elapsed time = {}'.format(time.time() - st))

	# Visualize results
	# r = results[0]
	# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
	# 														class_names, r['scores'])


eval_data = {}
eval_data['data'] = result_data
dump_json('eval_data.json', eval_data)