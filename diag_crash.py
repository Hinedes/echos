"""Reproduce HOLD_AND_OBSERVE crash from step 1500 state."""
import genesis as gs
import numpy as np

gs.init(backend=gs.amdgpu)
scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=True))

body_w,body_d,body_h=0.165,0.165,0.0545;mass=0.225;g_=9.81;arm=0.0511
Kp_att=0.06;Kd_att=0.018;Kp_pos=3.5;Kd_pos=3.6;MAX_TILT=20.0
MAX_HACC=g_*np.tan(np.radians(MAX_TILT))
rotors=[np.array([arm,-arm,0.0]),np.array([arm,arm,0.0]),np.array([-arm,-arm,0.0]),np.array([-arm,arm,0.0])]

body=scene.add_entity(gs.morphs.Box(size=(body_w,body_d,body_h),pos=(0,0,1),fixed=False))
scene.add_entity(gs.morphs.Box(size=(7.0,0.01,2.0),pos=(2.75,-1.5,1.0),fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.01,6.0,2.0),pos=(-0.5,1.0,1.0),fixed=True))
scene.add_entity(gs.morphs.Box(size=(3.5,0.01,2.0),pos=(1.25,1.5,1.0),fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.01,3.0,2.0),pos=(3.0,2.75,1.0),fixed=True))
scene.add_entity(gs.morphs.Box(size=(3.5,0.01,2.0),pos=(4.75,4.0,1.0),fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.5,0.8,1.5),pos=(1.5,1.0,0.75),fixed=True))
scene.add_entity(gs.morphs.Plane())
scene.build();body.set_mass(mass);rs=scene.sim.rigid_solver;li=0

def qconj(q):w,x,y,z=q;return np.array([w,-x,-y,-z])
def qm(q1,q2):w1,x1,y1,z1=q1;w2,x2,y2,z2=q2;return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
def rtb(q):w,x,y,z=q;return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def qeuler(q):
    w,x,y,z=q
    roll=np.degrees(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)))
    pitch=np.degrees(np.arcsin(np.clip(2*(w*y-z*x),-1,1)))
    yaw=np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
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
 sat=~(ts==clamped)
 return clamped,sat,ts

def apply_forces(solver, lin_idx, rotors_def, arm_len, thrusts):
    ft=np.zeros(3);tt=np.zeros(3);ys=[1,-1,-1,1]
    for i,(r,f)in enumerate(zip(rotors_def,thrusts)):
        F=np.array([0,0,f]);tt+=np.cross(r,F);tt[2]+=arm_len*ys[i]*f;ft+=F
    solver.apply_links_external_force(force=ft.reshape(1,3),links_idx=[lin_idx],ref="link_origin",local=True)
    solver.apply_links_external_torque(torque=tt.reshape(1,3),links_idx=[lin_idx],ref="link_origin",local=True)

OBS_YAW = -np.radians(-84.9)

for st in range(1501):
    rs.clear_external_force()
    qc=body.get_quat().cpu().numpy()
    p=body.get_pos().cpu().numpy()
    v=body.get_vel().cpu().numpy()
    om=body.get_ang().cpu().numpy()

    if st < 300:
        tgt=np.array([p[0]+2,0,1]);Tt,qd=ppd(tgt,p,v)
    elif st < 1431:
        tgt=np.array([3.24,1.56,1]);Tt,qd=ppd(tgt,p,v)
    elif st < 1560:
        tgt=np.array([p[0],p[1],1]);Tt,qd=ppd(tgt,p,v,OBS_YAW)
    else:
        tgt=np.array([p[0],p[1],1]);Tt,qd=ppd(tgt,p,v)
    tau,qe=apd(qd,qc,om)
    ts,sat,ts_raw=mxr(Tt,*tau)
    apply_forces(rs, li, rotors, arm, ts.clip(0,None))
    scene.step()

p_1500=p.copy(); qc_1500=qc.copy(); v_1500=v.copy(); om_1500=om.copy(); tgt_1500=tgt.copy()
r1500, p1500, y1500 = qeuler(qc_1500)

# ===== HOLD_AND_OBSERVE continuation with logging =====
MAX_STEPS=300
log=[]

def check_divergence(step_rel, e, prev_qe_angle, qe_inc_count):
    r,pitch,yaw=qeuler(np.array([e['qw'],e['qx'],e['qy'],e['qz']]))
    qerr_norm=abs(1-e['qnorm'])
    if abs(r)>45 or abs(pitch)>45:
        return 'roll/pitch >45deg', step_rel
    if e['z']<0.5:
        return 'altitude <0.5m', step_rel
    if qerr_norm>1e-4:
        return 'quaternion norm error >1e-4', step_rel
    if not np.all(np.isfinite([e['rotor_0'],e['rotor_1'],e['rotor_2'],e['rotor_3']])):
        return 'non-finite rotor thrust', step_rel
    if e['sat_0'] or e['sat_1'] or e['sat_2'] or e['sat_3']:
        return 'rotor saturation', step_rel
    qe_angle=2*np.arccos(np.clip(e['qe_w'],-1,1))
    if prev_qe_angle is not None and qe_angle>prev_qe_angle:
        qe_inc_count+=1
        if qe_inc_count>=10:
            return 'attitude error increasing 10+ steps', step_rel
    else:
        qe_inc_count=0
    return None, qe_inc_count

prev_qe_angle=None; qe_inc_count=0; trigger=None; trigger_step=None

for step in range(1500,1500+MAX_STEPS):
    rs.clear_external_force()
    qc=body.get_quat().cpu().numpy()
    p=body.get_pos().cpu().numpy()
    v=body.get_vel().cpu().numpy()
    om=body.get_ang().cpu().numpy()
    tgt=np.array([p[0],p[1],1])
    Tt,qd=ppd(tgt,p,v,OBS_YAW)
    tau,qe=apd(qd,qc,om)
    ts,sat,ts_raw=mxr(Tt,*tau)
    apply_forces(rs, li, rotors, arm, ts.clip(0,None))
    scene.step()
    e={'step':step-1500,'x':p[0],'y':p[1],'z':p[2],
       'qx':qc[1],'qy':qc[2],'qz':qc[3],'qw':qc[0],'qnorm':np.linalg.norm(qc),
       'vx':v[0],'vy':v[1],'vz':v[2],'wx':om[0],'wy':om[1],'wz':om[2],
       'tgt_x':tgt[0],'tgt_y':tgt[1],'tgt_z':tgt[2],
       'des_yaw':np.degrees(OBS_YAW),
       'qd_w':qd[0],'qd_x':qd[1],'qd_y':qd[2],'qd_z':qd[3],
       'qe_w':qe[0],'qe_x':qe[1],'qe_y':qe[2],'qe_z':qe[3],
       'tau_x':tau[0],'tau_y':tau[1],'tau_z':tau[2],'Tt':Tt,
       'rotor_0':ts[0],'rotor_1':ts[1],'rotor_2':ts[2],'rotor_3':ts[3],
       'sat_0':bool(sat[0]),'sat_1':bool(sat[1]),'sat_2':bool(sat[2]),'sat_3':bool(sat[3]),
       'raw_0':ts_raw[0],'raw_1':ts_raw[1],'raw_2':ts_raw[2],'raw_3':ts_raw[3]}
    log.append(e)
    if trigger is None:
        t,val=check_divergence(step-1500,e,prev_qe_angle,qe_inc_count)
        if t is not None:
            trigger=t; trigger_step=step-1500
        qe_angle=2*np.arccos(np.clip(qe[0],-1,1))
        if prev_qe_angle is not None and qe_angle>prev_qe_angle:
            qe_inc_count+=1
        else:
            qe_inc_count=0
        prev_qe_angle=qe_angle

print("=== HOLD_AND_OBSERVE crash reproduction ===")
print("State at step 1500:")
print("  pos=(%.3f, %.3f, %.3f)  vel=(%.3f, %.3f, %.3f)" % (p_1500[0],p_1500[1],p_1500[2],v_1500[0],v_1500[1],v_1500[2]))
print("  roll=%.2f  pitch=%.2f  yaw=%.2f" % (r1500,p1500,y1500))
print("  quat=(%.6f, %.6f, %.6f, %.6f)" % tuple(qc_1500))
print("  omega=(%.4f, %.4f, %.4f) rad/s" % (om_1500[0], om_1500[1], om_1500[2]))
print("Steps: %d" % len(log))
if trigger:
    print("DIVERGED at step %d (rel=%d): %s" % (1500+trigger_step,trigger_step,trigger))
else:
    print("STABLE for all %d steps" % len(log))

if trigger:
    ti=trigger_step
    print("\n--- Timeline [-5:+5] around divergence ---")
    for i in range(max(0,ti-5),min(len(log),ti+5)):
        e=log[i]
        r,pitch,yaw=qeuler(np.array([e['qw'],e['qx'],e['qy'],e['qz']]))
        qe_angle=2*np.arccos(np.clip(e['qe_w'],-1,1))
        sat_str="S" if (e['sat_0'] or e['sat_1'] or e['sat_2'] or e['sat_3']) else "."
        print("rel=%3d: z=%.3f r=%.1f p=%.1f y=%.1f qe_a=%.5f tau=(%+.4f,%+.4f,%+.4f) Tt=%.4f rot=[%.4f,%.4f,%.4f,%.4f] %s" %
              (e['step'],e['z'],r,pitch,yaw,qe_angle,
               e['tau_x'],e['tau_y'],e['tau_z'],e['Tt'],
               e['rotor_0'],e['rotor_1'],e['rotor_2'],e['rotor_3'],sat_str))
    # full state at trigger
    e=log[ti]
    print("\n--- Full trigger state ---")
    for k,v in e.items():
        print("  %s: %s" % (k,v))

print("\n--- Quaternion error angle (first 40 steps) ---")
for i in range(min(40,len(log))):
    e=log[i]
    qe_angle=2*np.arccos(np.clip(e['qe_w'],-1,1))
    print("rel=%3d: qe_a=%.6f qe=(%+.4f,%+.4f,%+.4f,%+.4f)" % (e['step'],qe_angle,e['qe_w'],e['qe_x'],e['qe_y'],e['qe_z']))

gs.destroy()

# ===== Control test 1: position hold at step 1500 state, no yaw =====
print("\n\n===== CONTROL TEST 1: position hold (no yaw) at step 1500 state =====")
gs.init(backend=gs.amdgpu)
s2=gs.Scene(show_viewer=False,rigid_options=gs.options.RigidOptions(enable_collision=True))
b2=s2.add_entity(gs.morphs.Box(size=(body_w,body_d,body_h),pos=p_1500,fixed=False))
s2.add_entity(gs.morphs.Plane())
s2.build();b2.set_mass(mass);rs2=s2.sim.rigid_solver
b2.set_quat(qc_1500)  # after build

for step in range(100):
    rs2.clear_external_force()
    qc=b2.get_quat().cpu().numpy();p=b2.get_pos().cpu().numpy()
    v=b2.get_vel().cpu().numpy();om=b2.get_ang().cpu().numpy()
    tgt=np.array([p_1500[0],p_1500[1],1])
    Tt,qd=ppd(tgt,p,v)
    tau,qe=apd(qd,qc,om)
    ts,sat,ts_raw=mxr(Tt,*tau)
    apply_forces(rs2,0,rotors,arm,ts.clip(0,None))
    s2.step()
    r,pitch,yaw=qeuler(qc)
    print("step=%d: z=%.3f r=%.1f p=%.1f y=%.1f v=(%.3f,%.3f,%.3f) om=(%.4f,%.4f,%.4f)" %
          (step,p[2],r,pitch,yaw,v[0],v[1],v[2],om[0],om[1],om[2]))
gs.destroy()

# ===== Control test 2: yaw change at origin =====
print("\n\n===== CONTROL TEST 2: yaw command at origin =====")
gs.init(backend=gs.amdgpu)
s3=gs.Scene(show_viewer=False,rigid_options=gs.options.RigidOptions(enable_collision=True))
b3=s3.add_entity(gs.morphs.Box(size=(body_w,body_d,body_h),pos=(0,0,1),fixed=False))
s3.add_entity(gs.morphs.Plane())
s3.build();b3.set_mass(mass);rs3=s3.sim.rigid_solver
b3.set_quat(np.array([1,0,0,0]))

for step in range(200):
    rs3.clear_external_force()
    qc=b3.get_quat().cpu().numpy();p=b3.get_pos().cpu().numpy()
    v=b3.get_vel().cpu().numpy();om=b3.get_ang().cpu().numpy()
    tgt=np.array([0,0,1])
    Tt,qd=ppd(tgt,p,v,OBS_YAW)
    tau,qe=apd(qd,qc,om)
    ts,sat,ts_raw=mxr(Tt,*tau)
    apply_forces(rs3,0,rotors,arm,ts.clip(0,None))
    s3.step()
    r,pitch,yaw=qeuler(qc)
    if step<20 or (step%20==0) or (abs(r)>45 or abs(pitch)>45 or p[2]<0.5):
        div="!!!" if (abs(r)>45 or abs(pitch)>45 or p[2]<0.5) else ""
        print("step=%d: z=%.3f r=%.1f p=%.1f y=%.1f om=(%.4f,%.4f,%.4f) Tt=%.4f %s" %
              (step,p[2],r,pitch,yaw,om[0],om[1],om[2],Tt,div))
        if div:
            break
gs.destroy()
