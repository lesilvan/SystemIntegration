#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import yaml
from scipy.spatial import KDTree
import time

STATE_COUNT_THRESHOLD = 4

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')
        self.light_states = ["red", "yellow", "green", "-", "unknown"]
        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        """Callback function which stores the waypoints received over the /base_waypoints topic and
        initializes the waypoint_tree"""
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        t0 = time.time()
        self.image_counter = 0
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()
        t1 = time.time()
        rospy.loginfo("[INFO] Image callback needed "+str(1000*(t1-t0))+" ms")

        '''
        Publish upcoming red lights at camera frequency.
		Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
		of times till we start using it. Otherwise the previous stable state is
		used.
		'''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
        with a KDTree query
        Args:
            x: x position to match a waypoint to
            y: y position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x,y],1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #For testing just return the light state
        #return light.state

        if(not self.has_image):
            self.prev_light_loc = None
            return False

        #t0 = time.time()
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image , "rgb8")
        #t1 = time.time()
        #rospy.loginfo("Conversion time"+str(1000*(t1-t0))+" ms")
        status = self.light_classifier.get_classification(cv_image)
        #t2 = time.time()
        #rospy.loginfo("Classification time"+str(1000*(t2-t1))+" ms")

        c_state = self.light_states[status]
        #c_state2 = self.light_states[light.state]
        #rospy.loginfo("Predicted " + c_state +" for true " +c_state2)
        return status

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if self.pose:
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
					#rospy.loginfo("light: {}, car_wp_index: {}, d: {}".format(
					#                i, car_wp_idx, temp_wp_idx, d))
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx
        if closest_light and diff < 80: #only detect when less than 80 wpts ahead
            t0 = time.time()
            state = self.get_light_state(closest_light)
            t1 = time.time()
            rospy.loginfo("[INFO] Get light state needed "+str(1000*(t1-t0))+" ms")
            c_string = self.light_states[state]
            rospy.loginfo("[INFO] Detected traffic light with state "+c_string+" is "+str(diff)+"  WPTS ahead.")
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
