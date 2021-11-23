#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point,PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped, Twist
import math as m
from sklearn.cluster import DBSCAN,KMeans
import numpy as np


min_gap_size=0.6
min_angle_diff=0.15
min_d=0.3
sensor_range=2.5
min_range=0.05
pose=[0,0,0]
obstacles=[]
mid_points=[]

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
    global obstacles,mid_points
    new_ranges=msg.ranges[-90:]+msg.ranges[:90]
    range_filter = [np.nan if (i > sensor_range or i < min_range) else i for i in new_ranges]
    angles = np.linspace(m.pi/2.0 , -m.pi/2.0 , 180).reshape(-1, 1) + pose[2]
    angles = np.arctan2(np.sin(angles), np.cos(angles)).reshape(-1, 1)
    ranges = np.array(range_filter).reshape(-1, 1)

    dx = (ranges * np.cos(angles)) + (pose[0] + m.cos(pose[2]) * 0.05)
    dy = (ranges * np.sin(angles)) + (pose[1] + m.sin(pose[2]) * 0.05)
    coords = np.hstack((dx, dy))
    obstacles = coords[~np.isnan(coords)].reshape(-1, 2)

    dbscan = DBSCAN(eps=0.3, min_samples=3, n_jobs=-1).fit(obstacles)
    clusters = obstacles[dbscan.labels_ != -1]
    cluster_labels = dbscan.labels_[dbscan.labels_ != -1]

    gaps = []
    for c_num in range(cluster_labels.max()):
        current_arr = clusters[cluster_labels == c_num]
        target_arr = clusters[cluster_labels > c_num]
        diffx = current_arr[:, 0].reshape(-1, 1) - target_arr[:, 0].reshape(-1, 1).T
        diffy = current_arr[:, 1].reshape(-1, 1) - target_arr[:, 1].reshape(-1, 1).T
        sqr_dist = np.sqrt(diffx ** 2 + diffy ** 2)
        curr_p = current_arr[np.argmin(sqr_dist) / target_arr.shape[0]]
        tar_p = target_arr[np.argmin(sqr_dist) % target_arr.shape[0]]
        angle1 = m.atan2(curr_p[1] - pose[1], curr_p[0] - pose[0])
        angle2 = m.atan2(tar_p[1] - pose[1], tar_p[0] - pose[0])
        if np.min(sqr_dist) > min_gap_size and abs( m.atan2(m.sin(angle1 - angle2), m.cos(angle1 - angle2))) > min_angle_diff:
            gaps.append([list(curr_p), list(tar_p)])

    mid_points = [[(pair[0][0] + pair[1][0]) / 2., (pair[0][1] + pair[1][1]) / 2.] for pair in gaps]

if __name__ == '__main__':
    rospy.init_node('laser_test', anonymous=True)
    pub1=rospy.Publisher('/visualization_marker',Marker,queue_size=1)
    sub1=rospy.Subscriber('/scan',LaserScan,get_laser)

    pts2 = create_marker(Marker.POINTS, "/base_link", 2, 0., 0., 1., 0.05, 0.05, 0.05)
    pts3 = create_marker(Marker.POINTS, "/base_link", 3, 0., 1., 0., 0.1, 0.1, 0.1)
    while not rospy.is_shutdown():
        pts2.points = []
        if not obstacles==[]:
            for i,j in obstacles:
                p = Point()
                p.x = i
                p.y = j
                pts2.points.append(p)
        pub1.publish(pts2)
        pts3.points = []
        if not mid_points == []:
            for i, j in mid_points:
                p = Point()
                p.x = i
                p.y = j
                p.z = 0.5
                pts3.points.append(p)
        pub1.publish(pts3)
