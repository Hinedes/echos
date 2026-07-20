"""Continue through the step-1560 yaw-target toggle to find the crash."""
import genesis as gs
import numpy as np

gs.init(backend=gs.amdgpu)
scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=True))

body_w,body_d,body_h=0.165,0.165,0.0545;mass=0.225;g_=9.81;arm=0.0511
Kp_att=0.06;Kd_att=0.018;Kp_pos=3.5;Kd_pos=3.6;MAX_TILT=20.0
MAX_HACC=g_*np.tan(np.radians(MAX_TILT))
rotors=[np.array([arm,-arm,0]),np.array([arm,arm,0]),np.array([-arm,-arm,0]),np.array([-arm,arm,0])]
body=scene.add_entity(gs.morphs.Box(size=(body_w,body_d,body_h),pos=(0,0,1),fixed=False))
scene.add_entity(gs.morphs.Box(size=(7.0,0.01,2.0),pos=(2.75,-1.5,1),fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.01,6.0,2.0),pos=(-0.5,1,1),fixed=True))
scene.add_entity(gs.morphs.Box(size=(3.5,0.01,2.0),pos=(1.25,1.5,1),fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.01,3.0,2.0),pos=(3.0,2.75,1),fixed=True))
scene.add_entity(gs.morphs.Box(size=(3.5,0.01,2.0),pos=(4.75,4.0,1),fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.5,0.8,1.5),pos=(1.5,1.0,0.75),fixed=True))
scene.add_entity(gs.morphs.Plane())
scene.build();body.set_mass(mass);rs=scene.sim.rigid_solver;li=0

def qconj(q):w,x,y,z=q;return np.array([w,-x,-y,-z])
def qm(q1,q2):w1,x1,y1,z1=q1;w2,x2,y2,z2=q2;return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
def rtb(q):w,x,y,z=q;return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def qeuler(q):
 w,x,y,z=q;roll=np.degrees(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)));pitch=np.degrees(np.arcsin(np.clip(2*(w*y-z*x),-1,1)));yaw=np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
 return roll,pitch,yaw
def rtq(R):
 t=np.trace(R)
 if t>0:s=0.5/np.sqrt(t+1);return np.array([0.25/s,(R[2,1]-R[1,2])*s,(R[0,2]-R[2,0])*s,(R[1,0]-R[0,1])*s])
 if R[0,0]>R[1,1]and R[0,0]>R[2,2]:s=2*np.sqrt(max(0,1+R[0,0]-R[1,1]-R[2,2]));return np.array([(R[2,1]-R[1,2])/s,0.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s])
 if R[1,1]>R[2,2]:s=2*np.sqrt(max(0,1+R[1,1]-R[0,0]-R[2,2]));return np.array([(R[0,2]-R[2,0])/s,(R[0,1]+R[1,0])/s,0.25*s,(R[1,2]+R[2,1])/s])
 s=2*np.sqrt(max(0,1+R[2,2]-R[0,0]-R[1,1]));return np.array([(R[1,0]-R[0,1])/s,(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,0.25*s])
def qfzy(zd,ps):
 xc=np.array([np.cos(ps),np.sin(ps),0]);zn=zd/np.linalg.norm(zd);yd=np.cross(zn,xc)
 if np.linalg.norm(yd)<1e-10:yd=np.array([0,1,0])
 else:yd/=np.linalg.norm(yd)
 return rtq(np.column_stack([np.cross(yd,zn),yd,zn]))
def ppd(pd,pc,vc,ps=0):
 e=pd-pc;a=Kp_pos*e-Kd_pos*vc
 a[0]=np.clip(a[0],-MAX_HACC,MAX_HACC);a[1]=np.clip(a[1],-MAX_HACC,MAX_HACC)
 Tv=mass*(a+np.array([0,0,g_]));Tt=np.linalg.norm(Tv)
 if Tt<1e-6:return mass*g_,np.array([1,0,0,0])
 return Tt,qfzy(Tv/Tt,ps)
def apd(qd,qc,ow):
 qe=qm(qconj(qd),qc);s=1 if qe[0]>=0 else -1
 tau=-Kp_att*2*s*qe[1:4]-Kd_att*rtb(qc)@ow
 return tau,qe
def mxr(T,tx,ty,tz=0):
 i=1/(4*arm)
 ts=np.array([T/4-tx*i-ty*i+tz*i,T/4+tx*i-ty*i-tz*i,T/4-tx*i+ty*i-tz*i,T/4-tx*i+ty*i+tz*i])
 clamped=np.clip(ts,0,None)
 return clamped,~(ts==clamped),ts

def apply_forces(solver, lin_idx, rotors_def, arm_len, thrusts):
    ft=np.zeros(3);tt=np.zeros(3);ys=[1,-1,-1,1]
    for i,(r,f)in enumerate(zip(rotors_def,thrusts)):
        F=np.array([0,0,f]);tt+=np.cross(r,F);tt[2]+=arm_len*ys[i]*f;ft+=F
    solver.apply_links_external_force(force=ft.reshape(1,3),links_idx=[lin_idx],ref="link_origin",local=True)
    solver.apply_links_external_torque(torque=tt.reshape(1,3),links_idx=[lin_idx],ref="link_origin",local=True)

OBS_YAW=-np.radians(-84.9)

# Run to step 1500
for st in range(1501):
    rs.clear_external_force()
    qc=body.get_quat().cpu().numpy();p=body.get_pos().cpu().numpy()
    v=body.get_vel().cpu().numpy();om=body.get_ang().cpu().numpy()
    if st<300:tgt=np.array([p[0]+2,0,1]);Tt,qd=ppd(tgt,p,v)
    elif st<1431:tgt=np.array([3.24,1.56,1]);Tt,qd=ppd(tgt,p,v)
    elif st<1560:tgt=np.array([p[0],p[1],1]);Tt,qd=ppd(tgt,p,v,OBS_YAW)
    else:tgt=np.array([p[0],p[1],1]);Tt,qd=ppd(tgt,p,v)
    tau,qe=apd(qd,qc,om);ts,sat,ts_raw=mxr(Tt,*tau)
    apply_forces(rs,li,rotors,arm,ts.clip(0,None))
    scene.step()

# Continue from step 1500 through yaw-toggle at 1560, looking for crash
MAX_POST=600  # from step 1500 to step 2100 (60 steps over 1560 + 540 more)
use_obs_yaw=True  # start with OBS_YAW
crash_log=[]

for step in range(1500,1500+MAX_POST):
    rs.clear_external_force()
    qc=body.get_quat().cpu().numpy();p=body.get_pos().cpu().numpy()
    v=body.get_vel().cpu().numpy();om=body.get_ang().cpu().numpy()
    
    # Toggle yaw target at step 1560
    if step==1560:
        use_obs_yaw=False
    
    desired_yaw = OBS_YAW if use_obs_yaw else 0.0
    tgt=np.array([p[0],p[1],1])
    Tt,qd=ppd(tgt,p,v,desired_yaw)
    tau,qe=apd(qd,qc,om);ts,sat,ts_raw=mxr(Tt,*tau)
    apply_forces(rs,li,rotors,arm,ts.clip(0,None))
    scene.step()
    
    r,pitch,yaw=qeuler(qc)
    qe_angle=2*np.arccos(np.clip(qe[0],-1,1))
    
    crash_log.append((step-1500,r,pitch,yaw,p[2],Tt,om[0],om[1],om[2],tau[0],tau[1],tau[2],qe_angle,desired_yaw,step>=1560,ts.copy(),sat.copy()))
    
    if abs(r)>45 or abs(pitch)>45 or p[2]<0.1:
        break

# Print full timeline
print("=== Full timeline: HOLD+OBSERVE through yaw toggle ===")
print("rel  abs  z r  p  y  yr qe_a  tau_x tau_y tau_z  Tt  rotors     sat  dr")
print("-"*80)
for e in crash_log:
    rel,r,pitch,yaw,z,Tt,ox,oy,oz,tx,ty,tz,qea,des_yaw,toggled,ts,sat=e
    toggle_mark="T" if toggled else "."
    des_yaw_label="%4.0f"%(np.degrees(des_yaw) if isinstance(des_yaw,(float,int)) else des_yaw)
    sat_str="S" if np.any(sat) else "."
    print("rel=%3d  z=%.2f r=%+.1f p=%+.1f y=%.1f qe=%.4f Tx=%+.4f Ty=%+.4f Tz=%+.4f Tt=%.4f rot=[%.3f,%.3f,%.3f,%.3f] %s %s" %
          (rel,z,r,pitch,yaw,qea,tx,ty,tz,Tt,ts[0],ts[1],ts[2],ts[3],sat_str,toggle_mark))
    if abs(r)>45 or abs(pitch)>45 or z<0.1:
        div="\n*** DIVERGED: "
        if abs(r)>45:div+="roll=%.1f "%r
        if abs(pitch)>45:div+="pitch=%.1f "%pitch
        if z<0.1:div+="z=%.3f "%z
        print(div)
        break
gs.destroy()
