from enum import Enum
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from nav_msgs.msg import Odometry

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class FSM_STATES(Enum):
    AT_START = 'At Start',
    HEADING_TO_TASK = 'Heading to Task',
    RETURNING_FROM_TASK = 'Returning from Task',
    TASK_DONE = 'Task Done',
    MOWING_LAWN = 'Mowing Lawn',
    SWITCHING_ROW = 'Switching Row'

class FSM(Node):

    def __init__(self):
        super().__init__('FSM')
        self.get_logger().info(f'{self.get_name()} created')

        self._subscriber = self.create_subscription(Odometry, "/odom", self._listener_callback, 1)
        self._publisher = self.create_publisher(Twist, "/cmd_vel", 1)

        # the blackboard
        self._cur_x = 0.0
        self._cur_y = 0.0
        self._cur_theta = 0.0
        self._cur_state = FSM_STATES.AT_START
        self._start_time = self.get_clock().now().nanoseconds * 1e-9

        # lawn stats
        self.row_length = 3
        self.row_offset = 1
        self.row_count  = 0
        self.row_max    = 3
        self.lock       = False
        self.goal_x     = 0
        self.goal_y     = 0
        self.goal_theta = 0
        

    def _drive_to_goal(self, goal_x, goal_y, goal_theta):
        self.get_logger().info(f'{self.get_name()} drive to goal')
        twist = Twist()

        x_diff = goal_x - self._cur_x
        y_diff = goal_y - self._cur_y
        dist = x_diff * x_diff + y_diff * y_diff
        self.get_logger().info(f'{self.get_name()} {x_diff} {y_diff}')

        # turn to the goal
        heading = math.atan2(y_diff, x_diff)
        if abs(self._cur_theta - heading) > math.pi/20: 
            if heading > self._cur_theta:
                twist.angular.z = 0.2
            else:
               twist.angular.z = -0.2
            # self.get_logger().info(f'{self.get_name()} turning towards goal')
            self._publisher.publish(twist)
            return False

        # pointing the right direction, so go there
        if dist > 0.1*0.1:
            twist.linear.x = 0.3
            self._publisher.publish(twist)
            # self.get_logger().info(f'{self.get_name()} driving to goal')
            return False

        # we are there, set the correct angle
        if abs(goal_theta - self._cur_theta) > math.pi/20: 
            if goal_theta > self._cur_theta:
                twist.angular.z = 0.005
            else:
               twist.angular.z = -0.005
            # self.get_logger().info(f'{self.get_name()} turning to goal direction')
            self._publisher.publish(twist)
        self.get_logger().info(f'{self.get_name()} at goal pose')
        return True


    def _do_state_at_start(self):
        self.get_logger().info(f'{self.get_name()} in start state')
        now = self.get_clock().now().nanoseconds * 1e-9
        if now > (self._start_time + 2):
            self._cur_state = FSM_STATES.HEADING_TO_TASK

    def _do_state_heading_to_task(self):
        self.get_logger().info(f'{self.get_name()} heading to task {self._cur_x} {self._cur_y} {self._cur_theta}')
        if self._drive_to_goal(1, 1, math.pi/2):
            self._cur_state = FSM_STATES.MOWING_LAWN

    def _do_state_returning_from_task(self):
        self.get_logger().info(f'{self.get_name()} returning from task ')
        if self._drive_to_goal(0, 0, 0):
            self._cur_state = FSM_STATES.TASK_DONE

    def _do_state_task_done(self):
        self.get_logger().info(f'{self.get_name()} task done')
    
    def _do_state_mowing_row(self):
        # save location data
        if not self.lock:
            self.goal_x = round(self._cur_x)
            self.goal_y = round(self._cur_y)+self.row_length*self.dirSin(self._cur_theta)
            self.goal_theta = (math.pi/2)*self.dirSin(self._cur_theta)
            self.lock = True

        self.get_logger().info(f'{self.get_name()} mowing row {self.goal_x} {self.goal_y} {self.goal_theta}')

        if self._drive_to_goal(self.goal_x, self.goal_y, self.goal_theta):
            self.lock = False
            self.row_count+=1
            if self.row_count>=self.row_max:
                self._cur_state = FSM_STATES.RETURNING_FROM_TASK
                self.get_logger().info(f'{self.get_name()} row count {self.row_count}')
            else:
                self._cur_state = FSM_STATES.SWITCHING_ROW

    def _do_state_switching_row(self):
        # save location data
        if not self.lock:
            self.goal_x = round(self._cur_x)+self.row_offset*self.dirSin(self._cur_theta)
            self.goal_y = round(self._cur_y)
            self.goal_theta = -(math.pi/2)*self.dirSin(self._cur_theta)
            self.lock = True

        self.get_logger().info(f'{self.get_name()} switching row {self.goal_x} {self.goal_y} {self.goal_theta}')

        if self._drive_to_goal(self.goal_x, self.goal_y, self.goal_theta):
            self.lock = False
            self.row_offset*=-1
            self._cur_state = FSM_STATES.MOWING_LAWN

    def _state_machine(self):
        if self._cur_state == FSM_STATES.AT_START:
            self._do_state_at_start()
        elif self._cur_state == FSM_STATES.HEADING_TO_TASK:
            self._do_state_heading_to_task()
        elif self._cur_state == FSM_STATES.RETURNING_FROM_TASK:
            self._do_state_returning_from_task()
        elif self._cur_state == FSM_STATES.TASK_DONE:
            self._do_state_task_done()
        elif self._cur_state == FSM_STATES.MOWING_LAWN:
            self._do_state_mowing_row()
        elif self._cur_state == FSM_STATES.SWITCHING_ROW:
            self._do_state_switching_row()
        else:
            self.get_logger().info(f'{self.get_name()} bad state {state_cur_state}')

    def _listener_callback(self, msg):
        pose = msg.pose.pose

        roll, pitch, yaw = euler_from_quaternion(pose.orientation)
        self._cur_x = pose.position.x
        self._cur_y = pose.position.y
        self._cur_theta = yaw
        self._state_machine()

    def dirSin(self, angle):
        if math.sin(angle)<0:
            return -1
        return 1
    
    def dirCos(self, angle):
        if math.cos(angle)<0:
            return -1
        return 1



def main(args=None):
    rclpy.init(args=args)
    node = FSM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()

