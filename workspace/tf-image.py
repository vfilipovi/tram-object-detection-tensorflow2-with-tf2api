import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import matplotlib.pyplot as plt
import warnings
matplotlib.use('Qt5Agg')
warnings.filterwarnings('ignore')

PATH_TO_SAVED_MODEL = "exported-models/my_model_efficientdet-d0-v1/saved_model"

print('Loading model...', end='')

# loading the model
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

# loading label_map
category_index_path = "exported-models/my_model_efficientdet-d0-v1/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(category_index_path, use_display_name=True)

# Array of images
img = ['inference/img3.jpg',
       'inference/img10.jpg'
       ]

for image_path in img:
    print('Running prediction for {}... '.format(image_path), end='')
    image_np = np.array(Image.open(image_path))

    # Input needs to be tensor
    input_tensor = tf.convert_to_tensor(image_np)

    # Model expects batch of images
    input_tensor = input_tensor[tf.newaxis, ...]

    # Running inference
    detections = detect_fn(input_tensor)

    # Output is tensor with extra dimension
    # Convert to numpy array and remove extra dimension with index[0]
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}

    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Copy image to numpy array for processing
    image_np_with_detections = image_np.copy()

    # Visualize results
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=6,
        min_score_thresh=.5,
        agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    print('Done')
    plt.show()
