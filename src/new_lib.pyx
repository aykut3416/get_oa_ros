cimport numpy as np
import numpy as npy
cdef extern from "math.h":
    cpdef double sin(double x)
    cpdef double cos(double x)
    cpdef double atan2(double y, double x)
    cpdef double sqrt(double x)
    cpdef double pow(double x,int y);
  
cdef struct out_x:
    float V
    float w
    float T
    float last_angle
    
def robot_forward_model(list p1, float v, float w, float T, int n_step):
    cdef float angles[50]
    cdef float x_p[50]
    cdef float y_p[50]
    cdef float deltat=T/float(n_step)
    cdef list traj=[]
    cdef int i
    x_p[0]=p1[0]
    y_p[0]=p1[1]
    angles[0]=p1[2]
    for i in range(1,n_step+1):
        angles[i]=angles[i-1]+deltat*w
        x_p[i]=x_p[i-1]+cos((angles[i]+angles[i-1])/2.0)*v*deltat
        y_p[i]=y_p[i-1]+sin((angles[i]+angles[i-1])/2.0)*v*deltat
        traj.append([x_p[i],y_p[i],angles[i]])
    return traj

def robot_inverse_model(list p1, list p2, float V_max=0.5, float w_max=0.75,float delta_t=0.2):
    
    cdef float diff_theta_points = atan2(p2[1]-p1[1],p2[0]-p1[0])
    cdef float diff_theta = atan2(sin(diff_theta_points-p1[2]),cos(diff_theta_points-p1[2]))
    cdef float arc_len, dx, dy,dist, radi
    cdef int s_p,i
    cdef out_x mystruct
    cdef list traj
    traj=[]
    
    if abs(diff_theta)<0.01:
        arc_len=sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
        mystruct.T=max(arc_len/V_max,abs(diff_theta)/w_max)
        mystruct.V=arc_len/mystruct.T
        mystruct.w=0
        mystruct.last_angle=p1[2]
        s_p=int(mystruct.T/delta_t)
        if s_p<2:
            s_p=2
        dx=(p2[0]-p1[0])/float(s_p)
        dy=(p2[1]-p1[1])/float(s_p)
        for i in range(1,s_p+1):
            traj.append([p1[0]+dx*i,p1[1]+dy*i,p1[2]])
        return mystruct,traj
    elif abs(diff_theta)>1.57:
        mystruct.T=0
        mystruct.V=0
        mystruct.w=0
        return mystruct,traj
    else:
        dist = sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
        radi = dist/ (2*sin(diff_theta))
        arc_len = radi*diff_theta*2
        mystruct.T=max(abs(arc_len)/V_max,abs(diff_theta*2)/w_max)
        mystruct.V=arc_len/mystruct.T
        mystruct.w=(diff_theta*2)/mystruct.T
        mystruct.last_angle=p1[2]+diff_theta*2
        s_p=int(mystruct.T/delta_t)
        if s_p<2:
            s_p=2
        traj=robot_forward_model(p1,mystruct.V,mystruct.w,mystruct.T,s_p)
        return mystruct,traj

def point_to_array_dist(double px,double py, np.ndarray[np.float64_t,ndim=2] arr):
    
    cdef double dpt,dist_j0
    cdef int nt,i
    cdef np.ndarray[np.float64_t,ndim=1] xt,yt

    nt=arr.shape[0]
    xt=arr[:,0]
    yt=arr[:,1]
    dist_j0=9e100
    for i in range(nt):
        dpt=sqrt((xt[i]-px)**2+(yt[i]-py)**2)
        if dpt<dist_j0:
            dist_j0=dpt
    return dist_j0


def array_to_array_dist(np.ndarray[np.float64_t,ndim=2] a1, np.ndarray[np.float64_t,ndim=2] a2):
    
    cdef double dpt,dist_j0
    cdef int n1,n2,i,j
    n1=a1.shape[0]
    n2=a2.shape[0]
    dist_j0=9e100
    for i in range(n1):
        for j in range(n2):
            dpt=sqrt((a1[i,0]-a2[j,0])**2+(a1[i,1]-a2[j,1])**2)
            if dpt<dist_j0:
                dist_j0=dpt
    return dist_j0

def nearest_neighbours(np.ndarray[np.float64_t,ndim=1] p1, np.ndarray[np.float64_t,ndim=2] targeta,int k):
    cdef np.ndarray[np.float64_t,ndim=1] cx,cy,dists
    cx=targeta[:,0]-p1[0]
    cy=targeta[:,1]-p1[1]
    dists=npy.hypot(cx,cy)
    return dists.argsort()[:k]

def create_ranges(list p1,list p2,int N):
    cdef float dx,dy,dz
    cdef list result
    cdef int dim
    result=[]
    dim=len(p1)
    if dim==1:
        dx=(p2[0]-p1[0])/float(N)
        for i in range(1,N+1):
            result.append(p1[0]+i*dx)
        return result
    elif dim==2:
        dx=(p2[0]-p1[0])/float(N)
        dy=(p2[1]-p1[1])/float(N)
        for i in range(1,N+1):
            result.append((p1[0]+i*dx,p1[1]+i*dy))
    elif dim==3:
        dx=(p2[0]-p1[0])/float(N)
        dy=(p2[1]-p1[1])/float(N)
        dz=(p2[2]-p1[2])/float(N)
        for i in range(1,N+1):
            result.append((p1[0]+i*dx,p1[1]+i*dy,p1[2]+i*dz))
    return result

def collision_check(np.ndarray[np.float64_t,ndim=2] a1, np.ndarray[np.float64_t,ndim=2] a2,float coll_radi):
    cdef float min_dist
    min_dist=array_to_array_dist(a1,a2)
    return min_dist>coll_radi

def radius_neighbors(list a1,np.ndarray[np.float64_t,ndim=2] targeta,float search_radi):
    cdef np.ndarray[np.float64_t,ndim=1] cx,cy,dists
    cx=targeta[:,0]-a1[0]
    cy=targeta[:,1]-a1[1]
    dists=npy.hypot(cx,cy)
    return dists.argsort()[:dists[dists<search_radi].shape[0]]

def circular_dist(list p1,list p2):
    cdef float diff_theta_points = atan2(p2[1]-p1[1],p2[0]-p1[0])
    cdef float diff_theta = atan2(sin(diff_theta_points-p1[2]),cos(diff_theta_points-p1[2]))
    cdef float arc_len,dist, radi
    if abs(diff_theta)<0.01:
        arc_len=sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
        return arc_len+abs(diff_theta)/3.14
    elif abs(diff_theta)>1.57:
        return 1000
    else:
        dist = sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
        radi = dist/ (2*sin(diff_theta))
        arc_len = radi*diff_theta*2+abs(diff_theta)/3.14
        return arc_len

def filter_free(np.ndarray[np.float64_t,ndim=2] points,np.ndarray[np.float64_t,ndim=2] obstacles,float r_n):
    cdef list free=[]
    cdef float dist
    cdef int in_size=points.shape[0]
    cdef int i
    for i in range(in_size):
        dist=point_to_array_dist(points[i,0],points[i,1],obstacles)
        if dist>r_n:
            free.append([points[i,0],points[i,1]])
    return free

def first_neighbours(np.ndarray[np.float64_t,ndim=2] points, np.ndarray[np.float64_t,ndim=2] obstacles):
    cdef double dpt,dist_j0
    cdef int i,j,n1,n2,ind_j0
    cdef list dists,neighs
    n1=points.shape[0]
    n2=obstacles.shape[0]
    dists,neighs=[],[]
    for i in range(n1):
        dist_j0=9e100
        for j in range(n2):
            dpt=sqrt((points[i,0]-obstacles[j,0])**2+(points[i,1]-obstacles[j,1])**2)
            if dpt<dist_j0:
                dist_j0=dpt
                ind_j0=j
        dists.append(dist_j0)
        neighs.append(ind_j0)
    return dists,neighs

def calc_forces(list dists, list neighs, np.ndarray[np.float64_t,ndim=2] points, np.ndarray[np.float64_t,ndim=2] obstacles,float radius,int control_ind):
    cdef double dpt,dist_j0,power
    cdef np.ndarray[np.float64_t,ndim=2] ext_f,intern_f,total_f
    cdef int i,j,n1,n2
    n1=len(dists)
    ext_f=npy.zeros((n1,2))
    intern_f=npy.zeros((n1,2))
    total_f=npy.zeros((n1,2))
    for i in range(1,n1-1):
        if (dists[i]<radius):
            power=(radius-dists[i])
            ext_f[i,0]=power*(points[i,0]-obstacles[neighs[i],0])/dists[i]
            ext_f[i,1]=power*(points[i,1]-obstacles[neighs[i],1])/dists[i]
        intern_f[i,0]=((points[i-1,0]-points[i,0])+(points[i+1,0]-points[i,0]))/sqrt((points[i-1,0]-points[i,0])**2+(points[i+1,0]-points[i,0])**2)
        intern_f[i,1]=((points[i-1,1]-points[i,1])+(points[i+1,1]-points[i,1]))/sqrt((points[i-1,1]-points[i,1])**2+(points[i+1,1]-points[i,1])**2)
    total_f=ext_f*0.3+intern_f*0.2
    total_f[control_ind,:]=0
    return total_f


def eband_algorithm(np.ndarray[np.float64_t,ndim=2] points,np.ndarray[np.float64_t,ndim=2] obstacles,float radius,int step,int control_ind):
    cdef int i,j
    cdef list dists,neighs
    cdef np.ndarray[np.float64_t,ndim=2] forces
    cdef int k = points.shape[0]
    if k>=3:
        for i in range(step):
            dists,neighs=first_neighbours(points,obstacles)
            forces = calc_forces(dists,neighs,points,obstacles,radius,control_ind)
            points=points+forces
    return points


