import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import warnings
matplotlib.use('Qt5Agg')
warnings.filterwarnings('ignore')

PATH_TO_SAVED_MODEL = "exported-models/my_model_efficientdet-d0-v1/saved_model"

# loading the model
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

# loading label_map
category_index_path = "exported-models/my_model_efficientdet-d0-v1/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(category_index_path, use_display_name=True)


# Running inference for single image
def run_inference_for_single_image(model, image):
    # Convert to numpy array
    image = np.asarray(image)

    # Input needs to be tensor
    input_tensor = tf.convert_to_tensor(image)

    # Model expects batch of images
    input_tensor = input_tensor[tf.newaxis, ...]

    # Running inference
    output_dict = model(input_tensor)

    # Output is tensor with extra dimension
    # Convert to numpy array and remove extra dimension with index[0]
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}

    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict


cap = cv2.VideoCapture(0)  # 0 is webcam number 1


def run_inference(model, cap):
    while cap.isOpened():
        ret, image_np = cap.read()

        # Running detection
        output_dict = run_inference_for_single_image(model, image_np)

        # Visualize results
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)

        cv2.imshow('object_detection', image_np)

        # Close the windows by pressing "Q"
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


run_inference(detect_fn, cap)
