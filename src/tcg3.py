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

pose=[0.0,0.0,0.0]
goalP=[0.0,0.0]
sensor_range=2.0
min_range=0.05
robot_r=0.20
safe_d=robot_r*3.0
safe_v=robot_r*4.0
gaps=[]
next_corner=[]
mid_theta=0.0
scs_theta=0.0
last_angle=0.0
max_v=0.5
max_w=0.75
control_signal=[0,0]

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

def create_ranges(start, stop, N, endpoint=True):
    if endpoint==1:
        divisor = N-1
    else:
        divisor = N
    steps = (1.0/divisor) * (stop - start)
    return steps[:,None]*np.arange(N) + start[:,None]

def control(dist,alpha,kb,min_d):
    if dist<0.2:
        return 0.0,0.0
    else:
        v_break=max((0.5-min_d)/0.5,0.0)
        v_limit=m.sqrt(1.0-v_break)*max_v
        km=(max_w-0.5*kb*v_limit)/(m.pi/4.0)
        v=kb*v_limit*m.cos(alpha)
        w_raw=(km*alpha)+v*m.sin(alpha)/dist
        if w_raw>max_w:
            w=max_w
        elif w_raw<-max_w:
            w=-max_w
        else:
            w=w_raw
        return v,w

def get_laser(msg):
    global gaps,next_corner,mid_theta,scs_theta,last_angle,control_signal
    a=time.time()
    goal_dist = m.hypot(goalP[1] - pose[1], goalP[0] - pose[0])
    goal_angle = m.atan2(goalP[1]-pose[1],goalP[0]-pose[0])
    lookhead_dist=min(goal_dist,sensor_range)

    range_filter = [np.nan if (i > lookhead_dist or i < min_range) else i for i in msg.ranges[90:-90]]
    angles = np.linspace(-m.pi/2.0 , m.pi/2.0 , 180).reshape(-1, 1) + pose[2]
    angles = np.arctan2(np.sin(angles), np.cos(angles)).reshape(-1, 1)
    ranges = np.array(range_filter).reshape(-1, 1)

    dx = (ranges * np.cos(angles)) + (pose[0] + m.cos(pose[2]) * 0.04)
    dy = (ranges * np.sin(angles)) + (pose[1] + m.sin(pose[2]) * 0.04)
    coords = np.hstack((dx, dy))
    coords = coords[~np.isnan(coords)].reshape(-1, 2)

    path_to_goal = 'high'
    safety='high'
    if coords.shape[0]!=0:
        next_p=[pose[0]+lookhead_dist*m.cos(goal_angle),pose[1]+lookhead_dist*m.sin(goal_angle)]
        traj=create_ranges(np.array([pose[0], pose[1]]), np.array([next_p[0], next_p[1]]), 10).T
        sqr_dist = cdist(traj, coords, 'euclidean')
        min_reading=min([range_ for range_ in range_filter if not m.isnan(range_)])
        if np.min(sqr_dist) > (robot_r*1.5):
            path_to_goal='high'
            last_angle=goal_angle
            alfa = m.atan2(m.sin(goal_angle - pose[2]), m.cos(goal_angle - pose[2]))
            p=m.hypot(goalP[0]-pose[0],goalP[1]-pose[1])
            v,w=control(p,alfa,m.tanh(p),min_reading)
            control_signal = [v, w]
        else:
            path_to_goal='low'
            angle1r = m.atan2(coords[0, 1] - pose[1], coords[0, 0] - pose[0])
            angle1 = m.atan2(m.sin(angle1r - pose[2]), m.cos(angle1r - pose[2]))
            angle2r = m.atan2(coords[-1, 1] - pose[1], coords[-1, 0] - pose[0])
            angle2 = m.atan2(m.sin(angle2r - pose[2]), m.cos(angle2r - pose[2]))
            if (m.pi / 2.0 + angle1) > 0.2:
                distance = m.hypot(coords[0, 1] - pose[1], coords[0, 0] - pose[0])
                new_p = [pose[0] + distance * m.cos(pose[2] - m.pi / 2.0),
                         pose[1] + distance * m.sin(pose[2] - m.pi / 2.0)]
                coords = np.vstack((np.array([new_p]).reshape(1, -1), coords))
            if (m.pi / 2.0 - angle2) > 0.2:
                distance = m.hypot(coords[-1, 1] - pose[1], coords[-1, 0] - pose[0])
                new_p = [pose[0] + distance * m.cos(pose[2] + m.pi / 2.0),
                         pose[1] + distance * m.sin(pose[2] + m.pi / 2.0)]
                coords = np.vstack((coords, np.array([new_p]).reshape(1, -1)))
            dbscan = DBSCAN(eps=0.3, min_samples=1).fit(coords)
            clusters = coords[dbscan.labels_ != -1]
            cluster_labels = dbscan.labels_[dbscan.labels_ != -1]
            gaps = []
            gap_sizes = []
            angles = []
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
                gap_size = m.hypot(tar_p[0] - curr_p[0], tar_p[1] - curr_p[1])
                if np.min(sqr_dist) > 2 * robot_r:
                    gaps.append([list(curr_p), list(tar_p)])
                    gap_sizes.append(gap_size)
                    angles.append([angle1, angle2])

            min_angle = m.pi
            min_ind = []
            min_c_angle = m.pi
            corner = 'left'
            alpha_array = []
            for ind, [ang1, ang2] in enumerate(angles):
                a_diffs = [abs(m.atan2(m.sin(goal_angle - ang1), m.cos(goal_angle - ang1))),
                           abs(m.atan2(m.sin(goal_angle - ang2), m.cos(goal_angle - ang2)))]
                a12_diff = m.atan2(m.sin(ang1 - ang2), m.cos(ang1 - ang2))
                if min(a_diffs) < min_angle:
                    min_ind = [ind, a_diffs.index(min(a_diffs))]
                    min_angle = min(a_diffs)
                    if min_ind[1] == 1 and a12_diff > 0.0:
                        corner = 'right'
                        mid_theta = m.atan2(m.sin(ang2 + a12_diff / 2.0), m.cos(ang2 + a12_diff / 2.0))
                        close_cor = gaps[min_ind[0]][min_ind[1]]
                        close_dist = m.hypot(close_cor[0] - pose[0], close_cor[1] - pose[1])
                        # print close_dist
                        push_theta = ang2 + m.asin(min((robot_r + safe_d) / close_dist,1.0))
                        scs_theta = m.atan2(m.sin(push_theta), m.cos(push_theta))
                        min_c_angle = ang2
                    elif min_ind[1] == 1 and a12_diff < 0.0:
                        corner = 'left'
                        mid_theta = m.atan2(m.sin(ang2 + a12_diff / 2.0), m.cos(ang2 + a12_diff / 2.0))
                        close_cor = gaps[min_ind[0]][min_ind[1]]
                        close_dist = m.hypot(close_cor[0] - pose[0], close_cor[1] - pose[1])
                        # print close_dist
                        push_theta = ang2 - m.asin(min((robot_r + safe_d) / close_dist,1.0))
                        scs_theta = m.atan2(m.sin(push_theta), m.cos(push_theta))
                        min_c_angle = ang2
                    elif min_ind[1] == 0 and a12_diff > 0.0:
                        corner = 'left'
                        mid_theta = m.atan2(m.sin(ang1 - a12_diff / 2.0), m.cos(ang1 - a12_diff / 2.0))
                        close_cor = gaps[min_ind[0]][min_ind[1]]
                        close_dist = m.hypot(close_cor[0] - pose[0], close_cor[1] - pose[1])
                        # print close_dist
                        push_theta = ang1 + m.asin(min((robot_r + safe_d) / close_dist,1.0))
                        scs_theta = m.atan2(m.sin(push_theta), m.cos(push_theta))
                        min_c_angle = ang1
                    elif min_ind[1] == 0 and a12_diff < 0.0:
                        corner = 'right'
                        mid_theta = m.atan2(m.sin(ang1 - a12_diff / 2.0), m.cos(ang1 - a12_diff / 2.0))
                        close_cor = gaps[min_ind[0]][min_ind[1]]
                        close_dist = m.hypot(close_cor[0] - pose[0], close_cor[1] - pose[1])
                        # print close_dist
                        push_theta = ang1 + m.asin(min((robot_r + safe_d) / close_dist,1.0))
                        scs_theta = m.atan2(m.sin(push_theta), m.cos(push_theta))
                        min_c_angle = ang1
            mid_diff = m.atan2(m.sin(mid_theta - min_c_angle), m.cos(mid_theta - min_c_angle))
            scs_diff = m.atan2(m.sin(scs_theta - min_c_angle), m.cos(scs_theta - min_c_angle))
            if abs(mid_diff) < abs(scs_diff):
                last_angle = mid_theta
            else:
                last_angle = scs_theta

            if min_reading > safe_v:
                safety = 'high'
                p=lookhead_dist
                min_dist=safe_v
            else:
                safety = 'low'
                diffs = coords - np.array([pose[0], pose[1]]).reshape(1, -1)
                nearest_pt = coords[np.argmin(np.hypot(diffs[:, 0], diffs[:, 1])), :]
                goal_pt=[pose[0]+lookhead_dist*m.cos(last_angle),pose[1]+lookhead_dist*m.sin(last_angle)]
                last_angle,p,min_dist= calc_vg(pose,goal_pt,nearest_pt)
            alfa=m.atan2(m.sin(last_angle-pose[2]),m.cos(last_angle-pose[2]))
            p = m.hypot(goalP[0] - pose[0], goalP[1] - pose[1])
            v, w = control(p, alfa, m.tanh(p), 0.5)
            control_signal = [v, w]
    else:
        path_to_goal='high'
        safety='high'
        alfa = m.atan2(m.sin(goal_angle - pose[2]), m.cos(goal_angle - pose[2]))
        p = m.hypot(goalP[0] - pose[0], goalP[1] - pose[1])
        v, w = control(p, alfa, m.tanh(p), 0.5)
        control_signal = [v, w]
    print "Time elapsed : "+str(time.time()-a)

def calc_vg(p0,p1,p2):
    ang1_=m.atan2(p1[1]-p0[1],p1[0]-p0[0])
    ang1=m.atan2(m.sin(ang1_-p0[2]),m.cos(ang1_-p0[2]))
    sign1=ang1/abs(ang1)
    ang2=m.atan2(p2[1]-p0[1],p2[0]-p0[0])
    ang2=m.atan2(m.sin(ang2-p0[2]),m.cos(ang2-p0[2]))
    sign2=ang2/abs(ang2)
    diff=(ang1-ang2)
    if sign1!=sign2 and abs(diff)>=m.pi:
        vg=-sign2*(3*m.pi/2.0)-diff
    elif sign1!=sign2 and abs(diff)<m.pi:
        vg=-sign2*(m.pi/2.0)-diff
    elif sign1==sign2 and abs(ang2)>=abs(ang1):
        vg=-sign2*(m.pi/2.0)-diff
    elif sign1==sign2 and abs(ang2)<abs(ang1):
        vg=sign2*(m.pi/2.0)-diff
    dist=m.hypot(p1[0]-p0[0],p1[1]-p0[1])
    dist2=m.hypot(p2[0]-p0[0],p2[1]-p0[1])
    return ang1_+vg,dist,dist2

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
    rospy.init_node('tcg_test', anonymous=True)
    pub1 = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
    pub2 = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    sub1 = rospy.Subscriber('/laser_scan', LaserScan, get_laser)
    rospy.Subscriber('/odom', Odometry, get_odom)
    rospy.Subscriber("/move_base_simple/goal", PoseStamped, get_goal)

    pts1 = create_marker(Marker.LINE_LIST, "/odom", 1, 1., 0., 0., 0.05, 0.05, 0.05)
    pts2 = create_marker(Marker.POINTS, "/odom", 2, 0., 0., 1., 0.05, 0.2, 0.05)
    pts3 = create_marker(Marker.ARROW, "/odom", 3, 0., 1., 0., 0.05, 0.2, 0.05)
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        pts1.points=[]
        for p1,p2 in gaps:
            a=Point()
            a.x=p1[0]
            a.y=p1[1]
            a.z=0.1
            b=Point()
            b.x=p2[0]
            b.y=p2[1]
            b.z=0.1
            pts1.points.append(a)
            pts1.points.append(b)
        pub1.publish(pts1)
        pts2.points = []
        if next_corner!=[]:
            # print next_corner
            a = Point()
            a.x = next_corner[0]
            a.y = next_corner[1]
            a.z = 0.1
            pts2.points.append(a)
        pub1.publish(pts2)
        pts3.points=[]
        a=Point()
        a.x=pose[0]
        a.y=pose[1]
        b=Point()
        b.x=pose[0]+m.cos(last_angle)
        b.y=pose[1]+m.sin(last_angle)
        pts3.points.append(a)
        pts3.points.append(b)
        pub1.publish(pts3)
        if m.hypot(goalP[1]-pose[1],goalP[0]-pose[0])>0.2:
            con=Twist()
            con.linear.x=control_signal[0]
            con.angular.z=control_signal[1]
            pub2.publish(con)
        else:
            pub2.publish(Twist())
        rate.sleep()

