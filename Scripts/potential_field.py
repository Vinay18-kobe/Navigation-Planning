#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion

class PotentialField:
    def __init__(self):
        rospy.init_node('potential_field')

        # Publishers
        self.cmd_pub = rospy.Publisher('/mux_cmd_vel/input2', Twist, queue_size=10)
        self.mux_pub = rospy.Publisher('/mux_cmd_vel/select', String, queue_size=1)

        # Subscribers
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)

        # State variables
        self.goal = None
        self.robot_pose = None
        self.scan_data = None
        self.last_selected = ""

        # Parameters
        self.d0 = 0.8  # Influence distance for repulsion
        self.k_rep = 4.0  # Repulsive gain (adjusted)
        self.k_att = 1.2  # Attractive gain (adjusted)
        self.max_speed = 0.22
        self.max_angular_vel = 2.0  # Added max angular velocity
        self.goal_tolerance = 0.15
        self.obstacle_threshold = 0.5  # Adjusted to allow reaction to closer obstacles
        self.turn_boost_factor = 2.0  # Increased repulsion during turns
        self.repulsive_factor = 2.5  # Increase repulsion during turning towards path
        self.slowdown_factor = 0.2  # Slow down when obstacles are too close
        self.turning_factor = 2.0  # Factor to make turning faster during obstacle avoidance
        self.backward_duration = 1  # Duration to move backward in seconds
        self.turn_duration = 1  # Duration to turn away from obstacle in seconds
        self.rate = rospy.Rate(10)

        rospy.loginfo("✅ Potential Field controller initialized.")
        self.control_loop()

    def goal_callback(self, msg):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])
        rospy.loginfo(f"📍 New goal received: {self.goal}")

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.robot_pose = np.array([pos.x, pos.y, yaw])

    def scan_callback(self, scan):
        self.scan_data = scan

    def laser_to_cartesian(self, scan):
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges)
        mask = np.isfinite(ranges) & (ranges > 0.05)
        xs = ranges[mask] * np.cos(angles[mask])
        ys = ranges[mask] * np.sin(angles[mask])
        return np.stack((xs, ys), axis=-1)

    def compute_repulsive_force(self, points, theta):
        force = np.zeros(2)
        for p in points:
            d = np.linalg.norm(p)
            if d == 0 or d > self.d0:
                continue
            direction = -p / d
            magnitude = self.k_rep * np.exp(-d) * (1.0 / d - 1.0 / self.d0)
            if abs(theta) > np.pi / 6:  # Increase repulsion during turns
                magnitude *= self.turn_boost_factor
            magnitude = max(min(magnitude, 5.0), 0)  # Clip extreme values
            force += magnitude * direction

        # Add strong repulsive force when the robot is turning and near an obstacle
        if abs(theta) > np.pi / 6:
            force *= self.repulsive_factor

        return force

    def compute_attractive_force(self):
        if self.robot_pose is None or self.goal is None:
            return np.zeros(2)
        pos = self.robot_pose[:2]
        return self.k_att * (self.goal - pos)

    def force_to_cmd(self, force_vector):
        angle = np.arctan2(force_vector[1], force_vector[0])
        speed = min(np.linalg.norm(force_vector), self.max_speed)

        cmd = Twist()
        cmd.linear.x = speed * np.cos(angle)
        cmd.angular.z = 3.0 * angle  # More responsive turn
        return cmd

    def move_backward(self):
        # Command to move backward
        rospy.loginfo("⚠️ Moving backward to avoid obstacle.")
        cmd = Twist()
        cmd.linear.x = -self.max_speed  # Move backward
        self.cmd_pub.publish(cmd)
        rospy.sleep(self.backward_duration)  # Move backward for 1 second (adjust as needed)

    def turn_away_from_obstacle(self):
        # Command to turn the robot away from the obstacle
        rospy.loginfo("⚠️ Turning away from the obstacle.")
        cmd = Twist()
        cmd.angular.z = self.max_angular_vel  # Turn in the opposite direction
        self.cmd_pub.publish(cmd)
        rospy.sleep(self.turn_duration)  # Turn for 1 second (adjust as needed)

    def switch_to_potential_field_control(self):
        # Switch to potential field control mode
        rospy.loginfo("🔁 Switching mux to input2 (potential field control).")
        self.mux_pub.publish(String(data="input2"))

    def switch_to_kinematic_control(self):
        # Switch to kinematic control mode
        rospy.loginfo("🔁 Switching mux to input1 (kinematic control).")
        self.mux_pub.publish(String(data="input1"))

    def control_loop(self):
        while not rospy.is_shutdown():
            if self.scan_data is None or self.robot_pose is None or self.goal is None:
                self.rate.sleep()
                continue

            current_pos = self.robot_pose[:2]
            distance_to_goal = np.linalg.norm(self.goal - current_pos)

            if distance_to_goal < self.goal_tolerance:
                self.cmd_pub.publish(Twist())
                rospy.loginfo("🎯 Goal reached!")
                self.goal = None
                continue

            # If obstacle is nearby, activate potential field and turn towards the path
            min_scan_distance = np.nanmin(self.scan_data.ranges)
            if min_scan_distance < self.obstacle_threshold:
                # Switch to kinematic control when moving backward
                self.switch_to_kinematic_control()
                # Move backward and turn away from the obstacle
                self.move_backward()  # Move backward
                self.turn_away_from_obstacle()  # Turn away from the obstacle

            else:
                # Switch to potential field control when no obstacle is near
                self.switch_to_potential_field_control()

            # Proceed with normal control after moving backward and turning
            cartesian_points = self.laser_to_cartesian(self.scan_data)
            rep_force = self.compute_repulsive_force(cartesian_points, self.robot_pose[2])  # Pass robot's orientation (theta)
            att_force = self.compute_attractive_force()
            total_force = att_force + rep_force

            cmd = self.force_to_cmd(total_force)

            # Slow down the robot if too close to an obstacle and increase repulsion force
            cmd.linear.x = max(min(cmd.linear.x, self.max_speed * self.slowdown_factor), 0.0)  # Slow down if near obstacle
            cmd.angular.z = min(cmd.angular.z, self.max_angular_vel)

            self.cmd_pub.publish(cmd)
            rospy.loginfo("🚙 Avoiding obstacle and steering towards the path.")

            self.rate.sleep()

if __name__ == '__main__':
    try:
        PotentialField()
    except rospy.ROSInterruptException:
        pass

