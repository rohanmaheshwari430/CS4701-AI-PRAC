import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import cv2
import numpy as np
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from util.directory_utils import create_directory
from util.file_utils import download_file, clone_repository
from util.protobuf_utils import extract_protobuf_archive
from util.labelmap_utils import generate_label_map
from util.tfrecord_utils import generate_tfrecord_script
from util.pipeline_config_utils import (
    load_pipeline_config, update_pipeline_config, save_pipeline_config
)
from util.detection_utils import (
    load_detection_model, detect_objects, visualize_detections
)

CUSTOM_MODEL_NAME = 'my_ssdnet_mobnet'
BASE_MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
BASE_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

WORKSPACE_PATH = os.path.join('Tensorflow', 'workspace')
SCRIPTS_PATH = os.path.join(WORKSPACE_PATH, 'scripts')
APIMODEL_PATH = os.path.join('Tensorflow', 'models')
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH, 'annotations')
IMAGE_PATH = os.path.join(WORKSPACE_PATH, 'images')
IMPROVED_MODEL_PATH = os.path.join(WORKSPACE_PATH, 'improved-models')
BASE_MODEL_PATH = os.path.join(WORKSPACE_PATH, 'base-models')
CHECKPOINT_PATH = os.path.join(IMPROVED_MODEL_PATH, CUSTOM_MODEL_NAME)
OUTPUT_PATH = os.path.join(CHECKPOINT_PATH, 'export')
TFJS_PATH = os.path.join(CHECKPOINT_PATH, 'tfjsexport')
TFLITE_PATH = os.path.join(CHECKPOINT_PATH, 'tfliteexport')
PROTOC_PATH = os.path.join('Tensorflow', 'protoc')

paths = {
    'WORKSPACE_PATH': WORKSPACE_PATH,
    'SCRIPTS_PATH': SCRIPTS_PATH,
    'APIMODEL_PATH': APIMODEL_PATH,
    'ANNOTATION_PATH': ANNOTATION_PATH,
    'IMAGE_PATH': IMAGE_PATH,
    'IMPROVED_MODEL_PATH': IMPROVED_MODEL_PATH,
    'BASE_MODEL_PATH': BASE_MODEL_PATH,
    'CHECKPOINT_PATH': CHECKPOINT_PATH,
    'OUTPUT_PATH': OUTPUT_PATH,
    'TFJS_PATH': TFJS_PATH,
    'TFLITE_PATH': TFLITE_PATH,
    'PROTOC_PATH': PROTOC_PATH
}

files = {
'PIPELINE_CONFIG': os.path.join(CHECKPOINT_PATH, 'pipeline.config'),
'TF_RECORD_SCRIPT': os.path.join(SCRIPTS_PATH, TF_RECORD_SCRIPT_NAME),
'LABELMAP': os.path.join(ANNOTATION_PATH, LABEL_MAP_NAME)
}

for path in paths.values():
    create_directory(path)

base_model_file = os.path.join(BASE_MODEL_PATH, BASE_MODEL_NAME + '.tar.gz')
download_file(BASE_MODEL_URL, base_model_file)

protobuf_archive = os.path.join(paths['PROTOC_PATH'], 'protoc-22.5-win64.zip')
download_file("https://github.com/protocolbuffers/protobuf/releases/download/v22.5/protoc-22.5-win64.zip", protobuf_archive)
extract_protobuf_archive(paths['PROTOC_PATH'])

clone_repository("https://github.com/tensorflow/models", paths['APIMODEL_PATH'])
os.system(f"cd {os.path.join(paths['APIMODEL_PATH'], 'research')} && protoc object_detection/protos/*.proto --python_out=.")

labels = [
{'name': 'ThumbsUp', 'id': 1},
{'name': 'ThumbsDown', 'id': 2},
{'name': 'ThankYou', 'id': 3},
{'name': 'LiveLong', 'id': 4}
]
generate_label_map(labels, files['LABELMAP'])

if not os.path.exists(files['TF_RECORD_SCRIPT']):
    clone_repository("https://github.com/nicknochnack/GenerateTFRecord", paths['SCRIPTS_PATH'])
    train_record_file = os.path.join(ANNOTATION_PATH, 'train.record')
    test_record_file = os.path.join(ANNOTATION_PATH, 'test.record')
    generate_tfrecord_script(os.path.join(IMAGE_PATH, 'train'), files['LABELMAP'], train_record_file)
    generate_tfrecord_script(os.path.join(IMAGE_PATH, 'test'), files['LABELMAP'], test_record_file)

pipeline_config = load_pipeline_config(files['PIPELINE_CONFIG'])
update_pipeline_config(pipeline_config, labels, BASE_MODEL_PATH, files['LABELMAP'], train_record_file, test_record_file)
save_pipeline_config(pipeline_config, files['PIPELINE_CONFIG'])

TRAINING_SCRIPT = os.path.join(APIMODEL_PATH, 'research', 'object_detection', 'model_main_tf2.py')
command = f"python {TRAINING_SCRIPT} --model_dir={CHECKPOINT_PATH} --pipeline_config_path={files['PIPELINE_CONFIG']} --num_train_steps=2000"
print(command)
os.system(command)

command = f"python {TRAINING_SCRIPT} --model_dir={CHECKPOINT_PATH} --pipeline_config_path={files['PIPELINE_CONFIG']} --checkpoint_dir={CHECKPOINT_PATH}"
print(command)
os.system(command)

configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = load_detection_model(configs)

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
image_path = os.path.join(IMAGE_PATH, 'test', 'livelong.02533422-940e-11eb-9dbd-5cf3709bbcc6.jpg')
img = cv2.imread(image_path)
image_np = np.array(img)
detections = detect_objects(image_np, detection_model, category_index)
image_np_with_detections = visualize_detections(image_np, detections, category_index)

plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.show()
