"""Check: does full corridor with static body (no dynamics) detect wall correctly?"""
import genesis as gs
import numpy as np

gs.init(backend=gs.amdgpu)

# Full corridor + dynamic body, reproduce step 1652 conditions
scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=True))
body=scene.add_entity(gs.morphs.Box(size=(0.165,0.165,0.0545),pos=(0,0,1),fixed=False))
scene.add_entity(gs.morphs.Box(size=(7.0,0.01,2.0),pos=(2.75,-1.5,1.0),fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.01,6.0,2.0),pos=(-0.5,1.0,1.0),fixed=True))
scene.add_entity(gs.morphs.Box(size=(3.5,0.01,2.0),pos=(1.25,1.5,1.0),fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.01,3.0,2.0),pos=(3.0,2.75,1.0),fixed=True))
scene.add_entity(gs.morphs.Box(size=(3.5,0.01,2.0),pos=(4.75,4.0,1.0),fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.5,0.8,1.5),pos=(1.5,1.0,0.75),fixed=True))
scene.add_entity(gs.morphs.Plane())
emit_off=(0.0865,0.0,0.0)
af=scene.add_sensor(gs.sensors.Raycaster(
    pattern=gs.sensors.SphericalPattern(fov=(60.0,0.0),n_points=(7,1)),
    entity_idx=body.idx,pos_offset=emit_off,max_range=5.0,no_hit_value=-1.0))
scene.build();body.set_mass(0.225)

# Place body at step 1652 position with PURE yaw (no roll/pitch)
yaw=np.radians(80)  # approximate yaw from the Euler data
R=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])

# Body pos such that sensor + forward offset = (3.077, 2.794, 1.0)
body_pos = np.array([3.077,2.794,1.0]) - R @ np.array([0.0865,0,0])
body.set_pos(body_pos)
# Pure yaw quaternion (no roll/pitch)
body.set_quat(np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)]))
for _ in range(5): scene.step()
d=af.read(); rf=d.distances.cpu().numpy().flatten()
az_d=np.linspace(-30,30,7)
ew=body.get_pos().cpu().numpy() + R @ np.array([0.0865,0,0])
print("=== Step 1652 repro STATIC (pure yaw, no roll/pitch) ===")
print("Body pos: (%.3f, %.3f, %.3f)" % tuple(body.get_pos().cpu().numpy()))
print("Sensor:   (%.3f, %.3f, %.3f)" % tuple(ew))
for i in range(7):
    db=np.array([np.cos(np.radians(az_d[i])),np.sin(np.radians(az_d[i])),0]); dw=R@db
    hit=rf[i]>=0
    if hit:
        print("  az=%d (%+.0f°): HIT  r=%.4f   dir=(%.3f,%.3f,z=%.3f)" % (i,az_d[i],rf[i],dw[0],dw[1],dw[2]))
    else:
        if dw[0] < 0:  # heading toward wall
            wall_t = (3.005-ew[0])/dw[0]
            in_range = 0 < wall_t <= 5
            print("  az=%d (%+.0f°): MISS dir=(%.3f,%.3f,z=%.3f) wall_t=%.3f %s" % (i,az_d[i],dw[0],dw[1],dw[2],wall_t,"IN_RANGE" if in_range else "BEYOND_RANGE"))
        else:
            print("  az=%d (%+.0f°): MISS dir=(%.3f,%.3f,z=%.3f) (away from wall)" % (i,az_d[i],dw[0],dw[1],dw[2]))

# Now check: would ANY beam z-component cause the ray to miss the wall?
print("\n--- Wall z-extent: [0, 2]. Sensor z: %.3f ---" % ew[2])
for i in range(7):
    db=np.array([np.cos(np.radians(az_d[i])),np.sin(np.radians(az_d[i])),0]); dw=R@db
    if dw[0] < 0:
        wall_t = (3.005-ew[0])/dw[0]
        if 0 < wall_t <= 5:
            z_at_wall = ew[2] + wall_t * dw[2]
            print("  az=%d: hit_z=%.3f %s wall range" % (i, z_at_wall, "IN" if 0<=z_at_wall<=2 else "OUTSIDE"))
gs.destroy()

# Now test with the dynamic simulation to see actual roll/pitch
gs.init(backend=gs.amdgpu)
scene=gs.Scene(show_viewer=False,rigid_options=gs.options.RigidOptions(enable_collision=True))
body=scene.add_entity(gs.morphs.Box(size=(0.165,0.165,0.0545),pos=(0,0,1),fixed=False))
scene.add_entity(gs.morphs.Box(size=(7.0,0.01,2.0),pos=(2.75,-1.5,1.0),fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.01,6.0,2.0),pos=(-0.5,1.0,1.0),fixed=True))
scene.add_entity(gs.morphs.Box(size=(3.5,0.01,2.0),pos=(1.25,1.5,1.0),fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.01,3.0,2.0),pos=(3.0,2.75,1.0),fixed=True))
scene.add_entity(gs.morphs.Box(size=(3.5,0.01,2.0),pos=(4.75,4.0,1.0),fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.5,0.8,1.5),pos=(1.5,1.0,0.75),fixed=True))
scene.add_entity(gs.morphs.Plane())
emit_off=(0.0865,0.0,0.0)
af=scene.add_sensor(gs.sensors.Raycaster(
    pattern=gs.sensors.SphericalPattern(fov=(60.0,0.0),n_points=(7,1)),
    entity_idx=body.idx,pos_offset=emit_off,max_range=5.0,no_hit_value=-1.0))
scene.build();body.set_mass(0.225);rs=scene.sim.rigid_solver;li=0

mass=0.225;g_=9.81;arm=0.0511;Kp_att=0.06;Kd_att=0.018;Kp_pos=3.5;Kd_pos=3.6
MAX_TILT=20.0;MAX_HACC=g_*np.tan(np.radians(MAX_TILT))
rotors=[np.array([arm,-arm,0]),np.array([arm,arm,0]),np.array([-arm,-arm,0]),np.array([-arm,arm,0])]
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
def apd(qd,qc,ow):qe=qm(qconj(qd),qc);s=1 if qe[0]>=0 else -1;return -Kp_att*2*s*qe[1:4]-Kd_att*rtb(qc)@ow
def mxr(T,tx,ty,tz=0):i=1/(4*arm);return np.array([T/4-tx*i-ty*i+tz*i,T/4+tx*i-ty*i-tz*i,T/4-tx*i+ty*i-tz*i,T/4-tx*i+ty*i+tz*i])
def arf(ts):
 ft=np.zeros(3);tt=np.zeros(3);ys=[1,-1,-1,1]
 for i,(r,f)in enumerate(zip(rotors,ts)):F=np.array([0,0,f]);tt+=np.cross(r,F);tt[2]+=arm*ys[i]*f;ft+=F
 rs.apply_links_external_force(force=ft.reshape(1,3),links_idx=[li],ref="link_origin",local=True)
 rs.apply_links_external_torque(torque=tt.reshape(1,3),links_idx=[li],ref="link_origin",local=True)

# Run to step 1652 and log
for st in range(1653):
    rs.clear_external_force()
    qc=body.get_quat().cpu().numpy();p=body.get_pos().cpu().numpy();v=body.get_vel().cpu().numpy();om=body.get_ang().cpu().numpy()
    df=af.read();rf=df.distances.cpu().numpy().flatten()
    Rb=rtb(qc).T;ew=p+Rb@np.array([0.0865,0,0])
    if st==1652:
        az_d=np.linspace(-30,30,7)
        roll,pitch,yaw=qeuler(qc)
        print("\n=== Step 1652 DYNAMIC (with roll/pitch) ===")
        print("Body pos: (%.3f, %.3f, %.3f) Euler: roll=%.1f pitch=%.1f yaw=%.1f" % (p[0],p[1],p[2],roll,pitch,yaw))
        print("Sensor:   (%.3f, %.3f, %.3f)" % (ew[0],ew[1],ew[2]))
        for i in range(7):
            db=np.array([np.cos(np.radians(az_d[i])),np.sin(np.radians(az_d[i])),0]); dw=Rb@db
            hit=rf[i]>=0
            if hit:
                print("  az=%d (%+.0f°): HIT  r=%.4f   dir=(%.3f,%.3f,z=%.3f)" % (i,az_d[i],rf[i],dw[0],dw[1],dw[2]))
            else:
                if dw[0] < 0:
                    wall_t=(3.005-ew[0])/dw[0]
                    z_at_wall=ew[2]+wall_t*dw[2]
                    in_range=0<wall_t<=5
                    print("  az=%d (%+.0f°): MISS dir=(%.3f,%.3f,z=%.3f) wall_t=%.3f z_wall=%.3f %s" % (i,az_d[i],dw[0],dw[1],dw[2],wall_t,z_at_wall,"IN_RANGE" if in_range else "BEYOND_R z_%s"%("IN" if 0<=z_at_wall<=2 else "OUT")))
                else:
                    print("  az=%d (%+.0f°): MISS dir=(%.3f,%.3f,z=%.3f) (away)" % (i,az_d[i],dw[0],dw[1],dw[2]))
    if st<300:tgt=np.array([p[0]+2,0,1]);Tt,qd=ppd(tgt,p,v);arf(np.clip(mxr(Tt,*apd(qd,qc,om)),0,None))
    elif st<1431:tgt=np.array([3.24,1.56,1]);Tt,qd=ppd(tgt,p,v);arf(np.clip(mxr(Tt,*apd(qd,qc,om)),0,None))
    elif st<1560:theta=-np.radians(-84.9);Tt,qd=ppd(np.array([p[0],p[1],1]),p,v,theta);arf(np.clip(mxr(Tt,*apd(qd,qc,om)),0,None))
    else:Tt,qd=ppd(np.array([p[0],p[1],1]),p,v);arf(np.clip(mxr(Tt,*apd(qd,qc,om)),0,None))
    scene.step()
gs.destroy()
