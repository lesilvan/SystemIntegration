from styx_msgs.msg import TrafficLight
import rospy

import cv2
import tensorflow as tf
import numpy as np
#import time
import os

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.optimizers import adam
from keras.utils.data_utils import Sequence
from keras.utils import plot_model, to_categorical
from keras import backend as K

class TLClassifier(object):
    def __init__(self):
        self.tl_classes_str = ['Red', 'Yellow','Green','--', 'Unknown']
        self.tl_classes = [TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN, '--', TrafficLight.UNKNOWN]

        # Detection model
        SSDLITE_GRAPH_FILE = 'light_classification/models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
        self.detection_graph = self.load_graph(SSDLITE_GRAPH_FILE)
        # Get placeholders for session
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Classification model
        CLASSIFIER_GRAPH_FILE = 'light_classification/models/tl_classifier_tf_model.pb'
        self.classifier_graph = self.load_graph(CLASSIFIER_GRAPH_FILE)
        # Get placeholders for session
        self.output_tensor = self.classifier_graph.get_tensor_by_name('activation_5_2/Softmax:0')
        self.input_tensor = self.classifier_graph.get_tensor_by_name('conv2d_1_input_2:0')

        # configuration for possible GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Start sessions for detection and classification
        self.sess = tf.Session(graph=self.detection_graph)
        self.sess1 = tf.Session(graph=self.classifier_graph)

    def __del__(self):
        self.sess.close()
        self.sess1.close()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Try to detect position of tl in image
        #t0 = time.time()
        coords = self.detect_tl_in_image(image, self.sess)
        #t1 = time.time()
        #rospy.loginfo("Detection timing "+str((t1-t0)*1000)+" ms")
        #rospy.loginfo("Coords "+str(coords))
        if coords.size > 0:
            # Crop image around detected traffic light and resize to (16x32 px)
            #t2 = time.time()
            img = cv2.resize(image[coords[0]:coords[2], coords[1]:coords[3]], (16,32))
            image_np = np.expand_dims(np.asarray(img, dtype=np.uint8), 0)

            pred_label = 4
            # Predict label (red, yellow, green)
            #t3 = time.time()
            predictions = self.sess1.run(self.output_tensor, {self.input_tensor: image_np})
            pred_label = np.argmax(predictions[0])
            #t4 = time.time()
            #rospy.loginfo("Classification timing: "+str((t4-t3)*1000)+" ms")
            color_string = self.tl_classes_str[pred_label]
            #rospy.loginfo("[INFO] TL_Classifier classified TL as "+ color_string)
            return self.tl_classes[pred_label]
        else:
            rospy.loginfo("[WARN] TL_Detector could not find a tl in image ")
            return TrafficLight.UNKNOWN

    def detect_tl_in_image(self, image, sess):
        """Uses session and pretrained model to detect traffic light in image and
        returns the coordinates in the image"""

        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                            feed_dict={self.image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.45
        search_class = 10 #ID for traffic light
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(confidence_cutoff, search_class, boxes, scores, classes)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        width, height = image.shape[1], image.shape[0]
        box_coords = self.to_image_coords(boxes, height, width)

        # Crop image to detected traffic light
        coords = np.squeeze(box_coords).astype(int)
        return coords

    def filter_boxes(self, min_score, search_class, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score and classes[i] == search_class:
                idxs.append(i)
                break

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        return box_coords

    def load_graph(self, graph_file):
        """Loads a frozen inference graph (for detection model)"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph
