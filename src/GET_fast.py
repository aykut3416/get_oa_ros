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
import time
from scipy.spatial.distance import cdist
from threading import Thread
import pyximport
pyximport.install(setup_args={"script_args":["--compiler=unix"], "include_dirs":np.get_include()}, reload_support=True)
import new_lib
pose=[0,0,0]
goalP=[0.,0.]
subgoal=[]
min_gap_size=0.6
min_angle_diff=0.15
min_d=0.3
sensor_range=2.5
min_range=0.05
bubble_maxr=0.6
bubble_step=0.2

obstacles=[]
bubble_list=[]
control_ind=[]
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

def point_to_point(p1,p2,p3):
    dist = m.hypot(p1[1] - p2[1], p1[0] - p2[0])
    n_p = max(int(dist / bubble_step), 2)
    dist2 = m.hypot(p3[1] - p2[1], p3[0] - p2[0])
    p_g = max(int(dist2 / bubble_step), 2)

    my_arr = np.zeros((n_p + p_g-1, 2))
    line1=new_lib.create_ranges(p1,p2,n_p)[:-1]
    my_arr[0:n_p-1, :] = np.array(line1).reshape(-1,2)
    line2=new_lib.create_ranges(p2,p3,p_g)
    my_arr[n_p-1:,:] = np.array(line2).reshape(-1,2)
    return my_arr,n_p-1

def RRT_path(pts):
    V=[]
    cost=[0]
    V.append([pose[0],pose[1]])
    parents={0:-1}
    poses=[pose]
    traj=[]
    candidates=[]
    controls=[]
    for pt_x in pts:
        nbors = new_lib.nearest_neighbours(pt_x,np.array(V).reshape(-1, 2), min(len(V),3))
        min_c=1000.
        best_pxy=[]
        for ind in nbors:
            control, points = new_lib.robot_inverse_model(poses[ind],[pt_x[0],pt_x[1]])
            if control['T'] == 0:
                continue
            min_dist = new_lib.array_to_array_dist(np.array(points)[:, :2], obstacles)
            if min_dist>min_d and (cost[ind] + control['T'])<min_c:
                min_c = cost[ind] + control['T']
                min_parent = ind
                sample_x = [points[-1][0], points[-1][1], control['last_angle']]
                best_pxy = np.array(points).reshape(-1,3)[:, :2]
                best_vw=[control['V'],control['w']]
        if min_c != 1000.:
            V.append(sample_x[:2])
            poses.append(sample_x)
            cost.append(min_c)
            parents[len(V)-1]=min_parent
            traj.append(best_pxy)
            candidates.append(len(V)-1)
            controls.append(best_vw)
    if m.hypot(V[-1][0]-pts[-1,0],V[-1][1]- pts[-1, 1]) < 0.1:
        path = get_path(len(V) - 1, parents)
        best_ctrl=controls[path[1]-1]
        return np.concatenate([traj[ind - 1] for ind in path[1:]], axis=0), best_ctrl, cost[-1]
    else:
        return [], [] ,0

def get_path(child_id,parents):
    path=[child_id]
    root=parents[child_id]
    while root!=-1:
        path.append(root)
        root=parents[root]
    path.reverse()
    return path

def calculate_paths(midp,subgoal,results,ind):

    arr,control_ind=point_to_point(pose[:2],midp,subgoal)
    if m.hypot(arr[-1,0]-arr[-2,0],arr[-1,1]-arr[-2,1])<0.01:
        arr=arr[:-1]
    new_arr=new_lib.eband_algorithm(arr,obstacles,min_d,3,control_ind)

    path, control, cost = RRT_path(pts=new_arr)

    if path != []:
        results[ind]=(path, cost,midp,control)
    else:
        results[ind] =([], [],midp,control)

def get_laser(msg):
    global obstacles,mid_points

    range_filter = [np.nan if (i > sensor_range or i < min_range) else i for i in msg.ranges[90:-90]]
    angles = np.linspace(-m.pi/2.0 , m.pi/2.0 , 180).reshape(-1, 1) + pose[2]
    angles = np.arctan2(np.sin(angles), np.cos(angles)).reshape(-1, 1)
    ranges = np.array(range_filter).reshape(-1, 1)

    dx = (ranges * np.cos(angles)) + (pose[0] + m.cos(pose[2]) * 0.04)
    dy = (ranges * np.sin(angles)) + (pose[1] + m.sin(pose[2]) * 0.04)
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

    dist = m.hypot(pose[0] - goalP[0], pose[1] - goalP[1])
    if dist < 1.0:
        g_angle = m.atan2(goalP[1] - pose[1], goalP[0] - pose[0])
        mid_points.append([pose[0] + m.cos(g_angle) * (dist / 2.), pose[1] + m.sin(g_angle) * (dist / 2.)])


    #
    # a = time.time()
    # bubble_list,control_ind = midp_to_goal(mid_points)
    # b = time.time()
    # print b - a
    # #------------------------OK------------------

def get_odom(msg):
    global pose
    pose[0]=msg.pose.pose.position.x
    pose[1] = msg.pose.pose.position.y
    quat = msg.pose.pose.orientation
    roll, pitch, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
    pose[2]=yaw

def get_goal(msg):
    global goalP
    goalP = [msg.pose.position.x, msg.pose.position.y]


if __name__ == '__main__':
    rospy.init_node('ebands_test', anonymous=True)
    pub1=rospy.Publisher('/visualization_marker',Marker,queue_size=1)
    pub2 = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    sub1=rospy.Subscriber('/scan',LaserScan,get_laser)
    rospy.Subscriber('/odom2', Odometry, get_odom)
    rospy.Subscriber("/move_base_simple/goal", PoseStamped, get_goal)

    pts1 = create_marker(Marker.LINE_LIST, "/odom", 1, 1., 0., 0., 0.1, 0.1, 0.1)
    pts2 = create_marker(Marker.POINTS,"/odom", 2, 0., 0., 1., 0.05, 0.05, 0.05)
    pts3 = create_marker(Marker.POINTS, "/odom", 3, 0., 1., 0., 0.1, 0.1, 0.1)
    rate = rospy.Rate(10)
    best_path=[]
    best_midp=[]
    my_paths=[]
    old_path=[]
    new_cost=0.
    old_cost=0.
    while pose[0]==0 and pose[1]==0:
        pass
    goalP=[pose[0],pose[1]]
    print "Goalp is determined as "+str(goalP)
    while not rospy.is_shutdown():

        a = time.time()
        try:
            if m.hypot(goalP[0]-pose[0],goalP[1]-pose[1])>0.2:
                dist = m.hypot(goalP[1] - pose[1], goalP[0] - pose[0])
                if dist < 2.5:
                    subgoal = goalP
                else:
                    lookhead_dist = min(dist, 3.0)
                    goal_angle = m.atan2(goalP[1] - pose[1], goalP[0] - pose[0])
                    subgoal = [pose[0] + lookhead_dist * m.cos(goal_angle), pose[1] + lookhead_dist * m.sin(goal_angle)]

                if best_path!=[]:
                    mid_points.append(best_midp)

                results = [None] * len(mid_points)
                thread_list = [None] * len(mid_points)
                copied_mids = mid_points[:]
                for t_ind in range(len(thread_list)):
                    thread_list[t_ind] = Thread(target=calculate_paths, args=(copied_mids[t_ind], subgoal,results,t_ind))
                    thread_list[t_ind].start()

                for t in thread_list:
                    if t is not None:
                        t.join()

                my_paths = []
                costs = []
                controls =[]
                mids=[]
                for res_ind in range(len(results)):
                    (path, cost,midp,control) = results[res_ind]
                    if path != []:
                        my_paths.append(path)
                        costs.append(cost)
                        mids.append(midp)
                        controls.append(control)
                if my_paths != []:
                    best_ind = costs.index(min(costs))
                    best_path = my_paths[best_ind]
                    best_midp = mids[best_ind]
                    best_control = controls[best_ind]
        except IndexError:
            print "Here2"
            pass

        copy_path=np.copy(best_path)
        if m.hypot(goalP[0]-pose[0],goalP[1]-pose[1])>0.2 and copy_path!=[]:
            msg=Twist()
            msg.linear.x=best_control[0]
            msg.angular.z=best_control[1]
            pub2.publish(msg)
        elif  m.hypot(goalP[0]-pose[0],goalP[1]-pose[1])>0.2:
            msg=Twist()
            msg.linear.x=0
            target_angle= m.atan2(goalP[1]-pose[1],goalP[0]-pose[0])
            msg.angular.z = m.atan2(m.sin(target_angle-pose[2]),m.cos(target_angle-pose[2]))
            pub2.publish(msg) 
        else:
            pub2.publish(Twist())
            best_path=[]


        pts1.points = []
        if copy_path != []:
            for ind in range(0,copy_path.shape[0] - 1):
                p = Point()
                p.x = copy_path[ind, 0]
                p.y = copy_path[ind, 1]
                p.z = 0.2
                pts1.points.append(p)
                p = Point()
                p.x = copy_path[ind + 1, 0]
                p.y = copy_path[ind + 1, 1]
                p.z = 0.2
                pts1.points.append(p)
            pub1.publish(pts1)

        #pts2.points = []
        #if not coords==[]:
        #     for i,j in coords:
        #         p = Point()
        #         p.x = i
        #         p.y = j
        #         pts2.points.append(p)
        #pub1.publish(pts2)


        pts3.points = []
        if not mid_points == []:
            for i, j in mid_points:
                p = Point()
                p.x = i
                p.y = j
                p.z = 0.5
                pts3.points.append(p)
        pub1.publish(pts3)

        e = time.time()
        if (e - a) > 0.01:
            print "Loop takes------ : " + str(e - a)
            # print len(costs)
        else:
            pass
            # print len(costs)
        rate.sleep()