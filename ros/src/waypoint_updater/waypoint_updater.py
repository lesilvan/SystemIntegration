#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 30 # Number of waypoints we will publish. You can change this number
MAX_DECEL = .5 #

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /obstacle_waypoint

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        #Init further necessary member variables
        self.stopline_wp_idx = -1
        self.pose = None
        self.base_lane = None
        self.waypoints_2d = None
        self.waypoint_tree = None

        # Run node
        self.loop()

    def loop(self):
        """Runs the main loop of this node"""
        rate = rospy.Rate(50)

        while not rospy.is_shutdown():
            if self.pose and self.base_lane:
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        """Checks for the ID of the closest waypoint ahead of the car"""
        # Find closest waypoint (could be ahead or behind)
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        closest_idx = self.waypoint_tree.query([x,y], 1)[1]

        # Make sure to choose the one ahead of the car
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Usage of an hyperplane through closest_coord
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])
        # Use dot product: if both vectors are with the same orientation
        # the product is positive -> closest point is behind the car -> ID+1
        # otherwise: the found closest point is ahead -> done
        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
        if val >0:
            closest_idx = (closest_idx+1) % len(self.waypoints_2d)
        return closest_idx

    #def publish_waypoints(self, closest_idx):
    def publish_waypoints(self):
        """Publishes the next N waypoints the car should drive through"""
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()

        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            rospy.loginfo("Now at "+str(closest_idx)+", trying to stop at "+ str(self.stopline_wp_idx))
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            stop_idx = max(self.stopline_wp_idx - closest_idx -2, 0) #Two points back from line so car stops before line
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2*dist*MAX_DECEL)
            if vel < 1:
                vel = 0.0
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp

    def pose_cb(self, msg):
        """Callback function which stores the message received over the /current_pose topic"""
        self.pose = msg

    def waypoints_cb(self, waypoints):
        """Callback function which stores the waypoints received over the /base_waypoints topic and
        initializes the waypoint_tree"""
        self.base_lane = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        """Callback function which stores the stopline_wp_idx received over the /traffic_waypoint topic"""
        self.stopline_wp_idx = msg.data
        rospy.loginfo("Stopline wp_index received:  " +str(msg.data))

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
