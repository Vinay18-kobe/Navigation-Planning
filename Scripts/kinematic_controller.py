#!/usr/bin/env python3

import rospy
import tf2_ros
import math
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from tf.transformations import euler_from_quaternion
from std_msgs.msg import String

class KinematicController:
    def __init__(self):
        rospy.init_node("kinematic_controller")

        # Initialize transform listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.mux_pub = rospy.Publisher('/mux_cmd_vel/select', String, queue_size=1)

        # Subscribers
        self.path_sub = rospy.Subscriber("/planned_path", Path, self.path_callback)

        # Internal variables
        self.path = []
        self.current_path_segment_index = 0
        self.goal_reached = False

        # Parameters
        self.k_alpha = rospy.get_param("~k_alpha", 1.0)
        self.epsilon = rospy.get_param("~epsilon", 0.1)
        self.max_linear_vel = rospy.get_param("~max_linear_vel", 0.22)
        self.max_angular_vel = rospy.get_param("~max_angular_vel", 2.0)
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 0.5)
        self.angle_threshold = rospy.get_param("~angle_threshold", math.pi / 4)
        self.obstacle_threshold = rospy.get_param("~obstacle_threshold", 0.5)  # Obstacle proximity threshold
        self.backward_distance = rospy.get_param("~backward_distance", 0.5)  # Distance to move backward when obstacle detected
        self.backward_duration = 1  # Duration to move backward in seconds
        self.turn_duration = 1  # Duration to turn away from obstacle in seconds

        self.rate = rospy.Rate(10)

        rospy.loginfo("✅ Kinematic controller initialized.")
        self.run()

    def path_callback(self, msg):
        self.path = msg.poses
        self.current_path_segment_index = 0
        self.goal_reached = False
        rospy.loginfo(f"✅ Received path with {len(self.path)} waypoints")

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
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi  # Normalize angle

        v = self.max_linear_vel
        w = self.k_alpha * alpha

        # Increase angular velocity if near obstacles or when sharp turns are needed
        if abs(alpha) > self.angle_threshold:
            v = 0.0
            w = self.max_angular_vel if alpha > 0 else -self.max_angular_vel

        # Adjust the angular velocity when obstacles are nearby
        if self.is_obstacle_nearby(x, y):
            rospy.loginfo("⚠️ Obstacle detected nearby. Increasing turning speed.")
            w *= 1.5  # Increase the angular velocity to make sharper turns

        # Ensure angular velocity stays within limits
        w = max(min(w, self.max_angular_vel), -self.max_angular_vel)

        return v, w

    def is_obstacle_nearby(self, x, y):
        # Placeholder for obstacle detection logic
        return math.hypot(x, y) < self.obstacle_threshold

    def move_backward(self):
        # Command to move backward
        rospy.loginfo("⚠️ Moving backward to avoid obstacle.")
        cmd = Twist()
        cmd.linear.x = -self.max_linear_vel  # Move backward
        self.cmd_vel_pub.publish(cmd)
        rospy.sleep(self.backward_duration)  # Move backward for 1 second (adjust as needed)

    def turn_away_from_obstacle(self):
        # Command to turn the robot away from the obstacle
        rospy.loginfo("⚠️ Turning away from the obstacle.")
        cmd = Twist()
        cmd.angular.z = self.max_angular_vel  # Turn in the opposite direction
        self.cmd_vel_pub.publish(cmd)
        rospy.sleep(self.turn_duration)  # Turn for 1 second (adjust as needed)

    def switch_to_potential_field_control(self):
        # Switch to potential field control mode
        rospy.loginfo("🔁 Switching mux to input2 (potential field control).")
        self.mux_pub.publish(String(data="input2"))

    def switch_to_kinematic_control(self):
        # Switch to kinematic control mode
        rospy.loginfo("🔁 Switching mux to input1 (kinematic control).")
        self.mux_pub.publish(String(data="input1"))

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

            # If obstacle detected, move backwards and turn
            if self.is_obstacle_nearby(x, y):
                # Switch to kinematic control when moving backwards
                self.switch_to_kinematic_control()
                # Move backward and turn away from the obstacle
                self.move_backward()  # Move backward
                self.turn_away_from_obstacle()  # Turn away from the obstacle

            else:
                # Switch to potential field control when no obstacle is near
                self.switch_to_potential_field_control()

            # Proceed with normal control after moving backward and turning
            v, w = self.compute_control(x, y, theta, lookahead)

            cmd = Twist()
            cmd.linear.x = max(min(v, self.max_linear_vel), 0)
            cmd.angular.z = w
            self.cmd_vel_pub.publish(cmd)

            # Extra confirmation for debugging
            rospy.loginfo_throttle(1.0, f"🚙 Publishing cmd_vel: linear={cmd.linear.x:.2f}, angular={cmd.angular.z:.2f}")
            self.rate.sleep()

if __name__ == "__main__":
    try:
        KinematicController()
    except rospy.ROSInterruptException:
        pass

