#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point,PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry,Path
from geometry_msgs.msg import PoseStamped, Twist
import math as m
from sklearn.cluster import DBSCAN,KMeans
import numpy as np
import time
from scipy.spatial.distance import cdist


sensor_range=2.5
min_range=0.05
robot_r=0.28
safe_d=robot_r*1.5
v_max=0.4
w_max=0.75
theta_traj=0.
left_arr,right_arr=[],[]
pose=[0.,0.,0.]
goalP=[0.,0.]

def create_marker(m_type,frame,id_m,rm,gm,bm,sx,sy,sz):
    mark = Marker()
    mark.header.frame_id=frame
    mark.header.stamp = rospy.Time.now()
    mark.ns = "markers"
    mark.id=id_m
    mark.type=m_type
    mark.action = Marker.ADD
    mark.color.r=rm
    mark.color.g=gm
    mark.color.b=bm
    mark.color.a=1.
    mark.scale.x=sx
    mark.scale.y = sy
    mark.scale.z = sz
    return mark


def get_laser(msg):
    global theta_traj,v_limit,left_arr,right_arr,pts1
    max_range=min(m.hypot(goalP[1]-pose[1],goalP[0]-pose[0]),sensor_range)
    range_filter = [np.nan if (i > max_range or i < min_range) else i for i in msg.ranges]
    angles = np.linspace(-m.pi, m.pi, 360).reshape(-1, 1) + pose[2]
    angles = np.arctan2(np.sin(angles), np.cos(angles)).reshape(-1, 1)
    ranges = np.array(range_filter).reshape(-1, 1)





def get_odom(msg):
    global pose
    pose[0]=msg.pose.pose.position.x
    pose[1] = msg.pose.pose.position.y
    quat = msg.pose.pose.orientation
    roll, pitch, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
    pose[2]=yaw


def get_goal(msg):
    global goalP
    print "Goal came"
    goalP = [msg.pose.position.x, msg.pose.position.y]


if __name__ == '__main__':
    rospy.init_node('snd_test2', anonymous=True)
    pub1=rospy.Publisher('/visualization_marker',Marker,queue_size=1)
    pub2 = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    sub1=rospy.Subscriber('/laser_scan',LaserScan,get_laser)
    rospy.Subscriber('/odom', Odometry, get_odom)
    rospy.Subscriber("/move_base_simple/goal", PoseStamped, get_goal)

    pts1 = create_marker(Marker.LINE_LIST, "/odom", 1, 1., 0., 0., 0.05, 0.05, 0.05)
    pts2 = create_marker(Marker.ARROW,"/odom", 2, 0., 0., 1., 0.05, 0.2, 0.05)
    pts3 = create_marker(Marker.ARROW, "/odom", 3, 0., 1., 0., 0.05, 0.2, 0.05)
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():

        # pub1.publish(pts1)
        #
        #
        # pts3.points = []
        # p = Point()
        # p.x = pose[0]
        # p.y = pose[1]
        # p.z = 0.2
        # pts3.points.append(p)
        # p = Point()
        # p.x = 1 * m.cos(theta_traj) + pose[0]
        # p.y = 1 * m.sin(theta_traj) + pose[1]
        # p.z = 0.2
        # pts3.points.append(p)
        # pub1.publish(pts3)
        #
        # if m.hypot(goalP[0]-pose[0],goalP[1]-pose[1])>0.2:
        #     angle_diff=m.atan2(m.sin(theta_traj-pose[2]),m.cos(theta_traj-pose[2]))
        #     if angle_diff/(m.pi/2.)<-w_max:
        #         w=-1
        #     elif angle_diff/(m.pi/2.)>w_max:
        #         w=1
        #     else:
        #         w=w_max*angle_diff/(m.pi/2.)
        #     if (m.pi/4.-abs(angle_diff))<0.:
        #         v=0.
        #     else:
        #         v=(m.pi/4.-abs(angle_diff))/(m.pi/4.)
        #     cmd=Twist()
        #     cmd.linear.x=v
        #     cmd.angular.z=w
        #     pub2.publish(cmd)
        # else:
        #     pub2.publish(Twist())

        rate.sleep()