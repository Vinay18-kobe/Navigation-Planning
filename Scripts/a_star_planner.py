#!/usr/bin/env python

import rospy
import numpy as np
import tf
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import heapq
import math
from collections import deque

class GlobalPlanner:
    def __init__(self):
        rospy.init_node('global_planner')

        self.map_data = None
        self.map_metadata = None
        self.start_pose = None
        self.goal_pose = None

        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)

        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=1, latch=True)

        self.occupancy_threshold = 50
        self.inflation_radius = 3
        self.connection_radius = 1.5
        self.smoothing_window = 5

        rospy.loginfo("✅ Global Planner Ready")
        rospy.spin()

    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_metadata = msg.info
        rospy.loginfo("🗺️ New map received ({}x{})".format(msg.info.width, msg.info.height))

    def amcl_callback(self, msg):
        self.start_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        rospy.loginfo("📍 Current robot position (AMCL): ({:.2f}, {:.2f})".format(*self.start_pose))
        self.try_plan()

    def goal_callback(self, msg):
        self.goal_pose = (msg.pose.position.x, msg.pose.position.y)
        self.goal_orientation = msg.pose.orientation
        rospy.loginfo("🎯 Goal position set to: ({:.2f}, {:.2f})".format(*self.goal_pose))
        self.try_plan()

    def try_plan(self):
        if self.map_data is None:
            rospy.logwarn("⚠️ No map data available")
            return

        if self.start_pose is None:
            rospy.logwarn("⚠️ No start position available")
            return

        if self.goal_pose is None:
            rospy.logwarn("⚠️ No goal position set")
            return

        rospy.loginfo("🧠 Planning path from ({:.2f}, {:.2f}) to ({:.2f}, {:.2f})".format(
            *self.start_pose, *self.goal_pose))

        inflated_map = self.inflate_obstacles()
        path = self.a_star(inflated_map, self.start_pose, self.goal_pose)

        if path:
            smoothed_path = self.smooth_path(path)
            self.publish_path(smoothed_path)
        else:
            rospy.logwarn("❌ No valid path found")

    def inflate_obstacles(self):
        from scipy.ndimage import binary_dilation, generate_binary_structure
        obstacle_map = (self.map_data > self.occupancy_threshold) | (self.map_data < 0)
        struct = generate_binary_structure(2, 1)
        inflated = binary_dilation(obstacle_map, structure=struct, iterations=self.inflation_radius)
        return inflated

    def to_grid_coords(self, x, y):
        origin = self.map_metadata.origin.position
        res = self.map_metadata.resolution
        gx = int((x - origin.x) / res)
        gy = int((y - origin.y) / res)
        return gx, gy

    def to_world_coords(self, gx, gy):
        origin = self.map_metadata.origin.position
        res = self.map_metadata.resolution
        x = gx * res + origin.x + res / 2.0
        y = gy * res + origin.y + res / 2.0
        return x, y

    def a_star(self, obstacle_map, start_world, goal_world):
        width = self.map_metadata.width
        height = self.map_metadata.height
        start = self.to_grid_coords(*start_world)
        goal = self.to_grid_coords(*goal_world)

        def heuristic(a, b):
            return math.hypot(b[0] - a[0], b[1] - a[1])

        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx in range(-int(self.connection_radius), int(self.connection_radius)+1):
                for dy in range(-int(self.connection_radius), int(self.connection_radius)+1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if self.has_line_of_sight(pos, (nx, ny), obstacle_map):
                            dist = math.hypot(dx, dy)
                            neighbors.append(((nx, ny), dist))
            return neighbors

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break

            for next_pos, step_cost in get_neighbors(current):
                new_cost = cost_so_far[current] + step_cost
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        if goal not in came_from:
            rospy.logwarn("⚠️ Path not found!")
            return []

        path = []
        current = goal
        while current:
            path.append(current)
            current = came_from[current]
        path.reverse()

        return [self.to_world_coords(*p) for p in path]

    def has_line_of_sight(self, pos1, pos2, obstacle_map):
        x0, y0 = pos1
        x1, y1 = pos2
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        n = dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n + 1):
            if obstacle_map[y, x]:
                return False
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return True

    def smooth_path(self, path):
        if len(path) < self.smoothing_window:
            return path

        smoothed = []
        window = deque(maxlen=self.smoothing_window)

        for point in path:
            window.append(point)
            avg_x = sum(p[0] for p in window) / len(window)
            avg_y = sum(p[1] for p in window) / len(window)
            smoothed.append((avg_x, avg_y))

        return smoothed

    def publish_path(self, path):
        ros_path = Path()
        ros_path.header.frame_id = "map"
        ros_path.header.stamp = rospy.Time.now()

        for i, (x, y) in enumerate(path):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = x
            pose.pose.position.y = y
            if i == len(path) - 1 and hasattr(self, 'goal_orientation'):
                pose.pose.orientation = self.goal_orientation
            else:
                pose.pose.orientation.w = 1.0
            ros_path.poses.append(pose)

        self.path_pub.publish(ros_path)
        rospy.loginfo("✅ Published path with {} points".format(len(path)))

if __name__ == '__main__':
    try:
        GlobalPlanner()
    except rospy.ROSInterruptException:
        pass

