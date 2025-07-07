#!/usr/bin/env python3

import rospy
import tf2_ros
import math
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion
import numpy as np

class Navigator:
    def __init__(self):
        rospy.init_node('navigator')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.mux_pub = rospy.Publisher('/cmd_vel_mux/select', String, queue_size=1)

        rospy.Subscriber('/planned_path', Path, self.path_callback)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)

        self.path = []
        self.current_path_segment_index = 0
        self.goal_reached = False
        self.last_selected = ""
        self.obstacle_nearby = False
        self.goal = None
        self.robot_pose = None

        self.k_alpha = rospy.get_param("~k_alpha", 1.0)
        self.epsilon = rospy.get_param("~epsilon", 0.1)
        self.max_linear_vel = rospy.get_param("~max_linear_vel", 0.22)
        self.max_angular_vel = rospy.get_param("~max_angular_vel", 2.0)
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 1.0)
        self.angle_threshold = rospy.get_param("~angle_threshold", math.pi / 6)
        self.obstacle_threshold = rospy.get_param("~obstacle_threshold", 0.5)

        self.rate = rospy.Rate(10)
        rospy.loginfo("✅ Navigator initialized.")
        self.run()

    def path_callback(self, msg):
        self.path = msg.poses
        self.current_path_segment_index = 0
        self.goal_reached = False
        rospy.loginfo(f"✅ Received path with {len(self.path)} waypoints")

    def scan_callback(self, msg):
        front = np.array(msg.ranges[len(msg.ranges)//2 - 20 : len(msg.ranges)//2 + 20])
        front = front[np.isfinite(front)]
        if len(front) > 0:
            min_dist = np.percentile(front, 20)
            self.obstacle_nearby = min_dist < self.obstacle_threshold
            rospy.loginfo_throttle(1.0, f"📏 Closest obstacle (20th percentile): {min_dist:.2f} m")
        else:
            self.obstacle_nearby = False

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0), rospy.Duration(2.0))
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
            return x, y, theta
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"⚠️ TF Lookup Failed: {e}")
            return None, None, None

    def find_lookahead_point(self, x, y):
        for i in range(self.current_path_segment_index, len(self.path)):
            wp = self.path[i].pose.position
            if math.hypot(wp.x - x, wp.y - y) >= self.lookahead_distance:
                self.current_path_segment_index = i
                return wp
        return self.path[-1].pose.position if self.path else None

    def compute_control(self, x, y, theta, goal):
        dx = goal.x - x
        dy = goal.y - y
        alpha = math.atan2(dy, dx) - theta
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi

        v = self.max_linear_vel
        w = self.k_alpha * alpha

        if abs(alpha) > self.angle_threshold:
            v = 0.05
            w = self.max_angular_vel if alpha > 0 else -self.max_angular_vel

        return v, max(min(w, self.max_angular_vel), -self.max_angular_vel)

    def switch_mux_input(self, input_name):
        if self.last_selected != input_name:
            self.mux_pub.publish(String(data=input_name))
            rospy.loginfo(f"🔁 Switching mux to {input_name}")
            self.last_selected = input_name

    def goal_callback(self, msg):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])
        rospy.loginfo(f"📍 New goal received: {self.goal}")

    def run(self):
        while not rospy.is_shutdown():
            if not self.path:
                rospy.loginfo_throttle(2.0, "⏳ Waiting for path...")
                self.cmd_vel_pub.publish(Twist())
                self.rate.sleep()
                continue

            x, y, theta = self.get_robot_pose()
            if x is None:
                self.cmd_vel_pub.publish(Twist())
                self.rate.sleep()
                continue

            final = self.path[-1].pose.position
            if math.hypot(final.x - x, final.y - y) < 0.05:
                if not self.goal_reached:
                    rospy.loginfo("🎯 Goal reached!")
                    self.goal_reached = True
                self.cmd_vel_pub.publish(Twist())
                self.rate.sleep()
                continue

            lookahead = self.find_lookahead_point(x, y)
            if lookahead is None:
                rospy.logwarn_throttle(2.0, "⚠️ No valid lookahead point. Stopping.")
                self.cmd_vel_pub.publish(Twist())
                self.rate.sleep()
                continue

            if self.obstacle_nearby:
                self.cmd_vel_pub.publish(Twist())
                rospy.logwarn_throttle(2.0, "🛑 Obstacle too close. Stopping robot.")
                self.rate.sleep()
                continue

            self.switch_mux_input("/cmd_vel")

            v, w = self.compute_control(x, y, theta, lookahead)
            cmd = Twist()
            cmd.linear.x = max(min(v, self.max_linear_vel), 0.0)
            cmd.angular.z = w
            self.cmd_vel_pub.publish(cmd)

            rospy.loginfo_throttle(1.0, f"🚙 cmd_vel: linear={cmd.linear.x:.2f}, angular={cmd.angular.z:.2f}")
            self.rate.sleep()

if __name__ == "__main__":
    try:
        Navigator()
    except rospy.ROSInterruptException:
        pass

