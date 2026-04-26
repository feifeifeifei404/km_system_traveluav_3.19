#!/usr/bin/env python3
import json, math, time
from pathlib import Path
import airsim, numpy as np, rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

T=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]],dtype=np.float64)

def q2r(q):
    x,y,z,w=q;xx,yy,zz=x*x,y*y,z*z;xy,xz,yz=x*y,x*z,y*z;wx,wy,wz=w*x,w*y,w*z
    return np.array([[1-2*(yy+zz),2*(xy-wz),2*(xz+wy)],[2*(xy+wz),1-2*(xx+zz),2*(yz-wx)],[2*(xz-wy),2*(yz+wx),1-2*(xx+yy)]],dtype=np.float64)

def r2q(r):
    t=float(np.trace(r))
    if t>0:s=math.sqrt(t+1)*2;w=.25*s;x=(r[2,1]-r[1,2])/s;y=(r[0,2]-r[2,0])/s;z=(r[1,0]-r[0,1])/s
    elif r[0,0]>r[1,1] and r[0,0]>r[2,2]:s=math.sqrt(1+r[0,0]-r[1,1]-r[2,2])*2;w=(r[2,1]-r[1,2])/s;x=.25*s;y=(r[0,1]+r[1,0])/s;z=(r[0,2]+r[2,0])/s
    elif r[1,1]>r[2,2]:s=math.sqrt(1+r[1,1]-r[0,0]-r[2,2])*2;w=(r[0,2]-r[2,0])/s;x=(r[0,1]+r[1,0])/s;y=.25*s;z=(r[1,2]+r[2,1])/s
    else:s=math.sqrt(1+r[2,2]-r[0,0]-r[1,1])*2;w=(r[1,0]-r[0,1])/s;x=(r[0,2]+r[2,0])/s;y=(r[1,2]+r[2,1])/s;z=.25*s
    q=np.array([x,y,z,w],dtype=np.float64);return q/np.linalg.norm(q)

def rot_angle_deg(r):
    c=np.clip((np.trace(r)-1.)*0.5,-1.,1.)
    return float(np.degrees(np.arccos(c)))

def ned2enu(p_ned,q_ned):
    p=T@p_ned; r=T@q2r(q_ned)@T.T; return p,r2q(r),r

def rot_to_rpy_deg(r):
    sy=math.sqrt(r[0,0]*r[0,0]+r[1,0]*r[1,0])
    singular=sy<1e-6
    if not singular:
        roll=math.atan2(r[2,1],r[2,2]); pitch=math.atan2(-r[2,0],sy); yaw=math.atan2(r[1,0],r[0,0])
    else:
        roll=math.atan2(-r[1,2],r[1,1]); pitch=math.atan2(-r[2,0],sy); yaw=0.0
    return [float(np.degrees(roll)),float(np.degrees(pitch)),float(np.degrees(yaw))]

def mat_to_list(r): return [[float(v) for v in row] for row in r]

def pose_dict(p,q): return {'position':[float(v) for v in p],'orientation_xyzw':[float(v) for v in q]}

def best_fit_rotation(rot_pairs):
    M=np.zeros((3,3),dtype=np.float64)
    for Ra,Ro in rot_pairs: M += Ra @ Ro.T
    U,_,Vt=np.linalg.svd(M)
    R=U@Vt
    if np.linalg.det(R)<0: U[:,-1]*=-1; R=U@Vt
    return R

class RigorousFrameVerifier(Node):
    def __init__(self):
        super().__init__('rigorous_frame_verifier')
        for k,v in [('airsim_ip','127.0.0.1'),('airsim_port',25001),('vehicle_name','Drone_1'),('odom_topic','/lidar_slam/odom'),('output_dir','/mnt/data/frame_verifier_output'),('target_samples',6),('min_sample_gap_sec',1.0),('min_yaw_change_deg',20.0),('min_pos_change_m',0.3)]: self.declare_parameter(k,v)
        self.ip=self.get_parameter('airsim_ip').value; self.port=int(self.get_parameter('airsim_port').value); self.vehicle=self.get_parameter('vehicle_name').value; self.odom_topic=self.get_parameter('odom_topic').value; self.out=Path(self.get_parameter('output_dir').value)
        self.target_samples=int(self.get_parameter('target_samples').value); self.min_gap=float(self.get_parameter('min_sample_gap_sec').value); self.min_yaw=float(self.get_parameter('min_yaw_change_deg').value); self.min_pos=float(self.get_parameter('min_pos_change_m').value)
        self.out.mkdir(parents=True,exist_ok=True)
        self.client=airsim.MultirotorClient(ip=self.ip,port=self.port,timeout_value=3.0); self.client.confirmConnection()
        qos=QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST,depth=10)
        self.create_subscription(Odometry,self.odom_topic,self.odom_cb,qos)
        self.odom_p=None; self.odom_q=None; self.odom_r=None; self.samples=[]; self.last_sample_time=0.0; self.timer=self.create_timer(0.2,self.tick)
        self.get_logger().info(f'rotation calibration ready, target_samples={self.target_samples}')
    def odom_cb(self,msg):
        self.odom_p=np.array([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z],dtype=np.float64)
        self.odom_q=np.array([msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w],dtype=np.float64)
        self.odom_r=q2r(self.odom_q)
    def airsim_pose(self):
        s=self.client.getMultirotorState(vehicle_name=self.vehicle); p=s.kinematics_estimated.position; q=s.kinematics_estimated.orientation
        return ned2enu(np.array([p.x_val,p.y_val,p.z_val],dtype=np.float64),np.array([q.x_val,q.y_val,q.z_val,q.w_val],dtype=np.float64))
    def should_take_sample(self,ap,ar):
        if len(self.samples)==0: return True,'first'
        last=self.samples[-1]
        dt=time.monotonic()-self.last_sample_time
        if dt<self.min_gap: return False,f'wait_gap<{self.min_gap:.1f}s'
        dpos=float(np.linalg.norm(ap-last['airsim_pos']))
        drot=rot_angle_deg(ar@last['airsim_rot'].T)
        if dpos<self.min_pos and drot<self.min_yaw: return False,f'need move/turn more (pos={dpos:.2f}m rot={drot:.1f}deg)'
        return True,f'accepted pos={dpos:.2f}m rot={drot:.1f}deg'
    def add_sample(self,ap,aq,ar):
        ok,reason=self.should_take_sample(ap,ar)
        if not ok:
            self.get_logger().info(reason,throttle_duration_sec=1.0)
            return
        s={'idx':len(self.samples),'airsim_pos':ap.copy(),'airsim_quat':aq.copy(),'airsim_rot':ar.copy(),'odom_pos':self.odom_p.copy(),'odom_quat':self.odom_q.copy(),'odom_rot':self.odom_r.copy(),'reason':reason}
        self.samples.append(s); self.last_sample_time=time.monotonic()
        self.get_logger().info(f'sample {len(self.samples)}/{self.target_samples} saved: {reason}')
    def finalize(self):
        R=best_fit_rotation([(s['airsim_rot'],s['odom_rot']) for s in self.samples])
        rpy=rot_to_rpy_deg(R)
        residuals=[]
        for s in self.samples:
            Rest=R@s['odom_rot']
            err=rot_angle_deg(s['airsim_rot']@Rest.T)
            residuals.append({'idx':s['idx'],'angle_residual_deg':err,'airsim_pose_enu':pose_dict(s['airsim_pos'],s['airsim_quat']),'odom_pose':pose_dict(s['odom_pos'],s['odom_quat'])})
        mean_err=float(np.mean([x['angle_residual_deg'] for x in residuals]))
        max_err=float(np.max([x['angle_residual_deg'] for x in residuals]))
        result={'target_samples':self.target_samples,'used_samples':len(self.samples),'rotation_matrix_airsim_from_odom':mat_to_list(R),'rotation_quat_xyzw':pose_dict(np.zeros(3),r2q(R))['orientation_xyzw'],'rotation_rpy_deg':rpy,'mean_angle_residual_deg':mean_err,'max_angle_residual_deg':max_err,'samples':residuals}
        j=self.out/'rotation_calibration.json'; t=self.out/'rotation_calibration_summary.txt'
        j.write_text(json.dumps(result,indent=2,ensure_ascii=False))
        lines=[f'Final RPY deg: {rpy}',f'Mean residual deg: {mean_err:.3f}',f'Max residual deg: {max_err:.3f}','Per-sample residuals:']
        for x in residuals: lines.append(f'  sample {x["idx"]}: {x["angle_residual_deg"]:.3f} deg')
        t.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        self.get_logger().info(f'final rotation rpy_deg = {np.round(rpy,3).tolist()}')
        self.get_logger().info(f'mean residual = {mean_err:.3f} deg, max residual = {max_err:.3f} deg')
        self.get_logger().info(f'saved {j}')
        self.get_logger().info(f'saved {t}')
        raise SystemExit
    def tick(self):
        if self.odom_r is None:
            self.get_logger().info('waiting /lidar_slam/odom ...',throttle_duration_sec=2.0)
            return
        try: ap,aq,ar=self.airsim_pose()
        except Exception as e:
            self.get_logger().error(f'airsim read failed: {e}')
            return
        self.add_sample(ap,aq,ar)
        if len(self.samples)>=self.target_samples: self.finalize()

def main():
    rclpy.init(); node=RigorousFrameVerifier()
    try:rclpy.spin(node)
    except (KeyboardInterrupt,SystemExit): pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__=='__main__': main()
