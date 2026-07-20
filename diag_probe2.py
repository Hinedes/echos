"""Verify the Box primitive wall at x=3 is detected by ARGUS FLOOD from both sides.
Then verify no cells beyond the wall become free with corrected mapper."""
import genesis as gs
import numpy as np

gs.init(backend=gs.amdgpu)
scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=True))

body_w,body_d,body_h=0.165,0.165,0.0545;mass=0.225;g_=9.81;arm=0.0511
Kp_att,Kd_att,Kp_pos,Kd_pos=0.06,0.018,3.5,3.6;MAX_TILT=20.0
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
emit_off=(body_w/2+0.004,0.0,0.0)
af=scene.add_sensor(gs.sensors.Raycaster(
    pattern=gs.sensors.SphericalPattern(fov=(60.0,0.0),n_points=(7,1)),
    entity_idx=body.idx,pos_offset=emit_off,max_range=5.0,no_hit_value=-1.0))
scene.build();body.set_mass(mass);rs=scene.sim.rigid_solver;li=0

def qconj(q):w,x,y,z=q;return np.array([w,-x,-y,-z])
def qm(q1,q2):w1,x1,y1,z1=q1;w2,x2,y2,z2=q2;return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
def rtb(q):w,x,y,z=q;return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
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
def mxr(T,tx,ty,tz=0):i=1/(4*arm);return np.array([T/4-tx*i-ty*i+tz*i,T/4+tx*i-ty*i-tz*i,T/4-tx*i+ty*i-tz*i,T/4+tx*i+ty*i+tz*i])
def arf(ts):
 ft=np.zeros(3);tt=np.zeros(3);ys=[1,-1,-1,1]
 for i,(r,f)in enumerate(zip(rotors,ts)):F=np.array([0,0,f]);tt+=np.cross(r,F);tt[2]+=arm*ys[i]*f;ft+=F
 rs.apply_links_external_force(force=ft.reshape(1,3),links_idx=[li],ref="link_origin",local=True)
 rs.apply_links_external_torque(torque=tt.reshape(1,3),links_idx=[li],ref="link_origin",local=True)

class M:
 def __init__(s,b,res=0.08):
  s.res=res;s.xm,s.xM,s.ym,s.yM=b;s.w=int((s.xM-s.xm)/res)+1;s.h=int((s.yM-s.ym)/res)+1
  s.hits=np.zeros((s.h,s.w),dtype=np.int32);s.views=np.zeros((s.h,s.w),dtype=np.int32)
  s.wall_xings=[]  # tracks rays where raycaster returned HIT but endpoint crosses x>3.0
  s.wall_penetrations=[]  # tracks rays where raycaster returned MISS but would hit wall at x=3
 def w2g(s,x,y):return int((x-s.xm)/s.res),int((y-s.ym)/s.res)
 def g2w(s,ix,iy):return s.xm+(ix+0.5)*s.res,s.ym+(iy+0.5)*s.res
 def ib(s,ix,iy):return 0<=ix<s.w and 0<=iy<s.h
 def ur(s,ew,dr,rng,mx,hit,eps=0.004,step=0,mode='F',az_i=0,raw_range=-1):
  ex,ey=ew[0],ew[1]
  if hit:ex2=ex+dr[0]*(rng+eps);ey2=ey+dr[1]*(rng+eps)
  else:ex2=ex+dr[0]*mx;ey2=ey+dr[1]*mx
  ix0,iy0=s.w2g(ex,ey);ix1,iy1=s.w2g(ex2,ey2)
  n=max(abs(ix1-ix0)+1,abs(iy1-iy0)+1)
  for i in range(n+1):
   t=i/max(n,1);cx=int(round(ix0+t*(ix1-ix0)));cy=int(round(iy0+t*(iy1-iy0)))
   if not s.ib(cx,cy):continue
   s.views[cy,cx]+=1
   if i>=n-1 and hit:s.hits[cy,cx]+=1
  # Check if this ray actually CROSSES x=3 (transitions from x>3 to x<=3 or vice versa)
  # within the wall's y,z range. Rays entirely on one side are NOT penetrations.
  crossed_wall=False
  prev_rx=ex
  for i in range(1,n+1):
   t=i/max(n,1)
   rx=ex+dr[0]*t*(mx if not hit else (rng+eps))
   ry=ey+dr[1]*t*(mx if not hit else (rng+eps))
   # Crossing x=3: prev on one side, current on other side
   if (prev_rx>3.0 and rx<=3.0) or (prev_rx<3.0 and rx>=3.0):
    if 1.5<=ry<=4.0:
     crossed_wall=True
     break
   prev_rx=rx
  if crossed_wall:
    if hit:
     s.wall_xings.append((step,mode,az_i,raw_range,ex,ey,dr[0],dr[1],rng))
    else:
     s.wall_penetrations.append((step,mode,az_i,raw_range,ex,ey,dr[0],dr[1],rng))

m=M((-2,7,-2,5),0.08);eps=0.004;az_d=np.linspace(-30,30,7);az_c=np.cos(np.radians(az_d))

# Simulate the observation scan: fly to (3.24, 1.56), yaw for scan, then position hold
for st in range(1700):
 rs.clear_external_force()
 qc=body.get_quat().cpu().numpy();p=body.get_pos().cpu().numpy();v=body.get_vel().cpu().numpy();om=body.get_ang().cpu().numpy()
 df=af.read();rf=df.distances.cpu().numpy().flatten()
 Rb=rtb(qc).T;ew=p+Rb@np.array([emit_off[0],0,0])
 for i in range(7):
  az=np.radians(az_d[i]);db=np.array([np.cos(az),np.sin(az),0]);dw=Rb@db;h=rf[i]>=0;rr=rf[i]if h else 0
  m.ur(ew,dw,rr,5,h,eps,st,'F',i,rf[i])
 if st<300:tgt=np.array([p[0]+2,0,1]);Tt,qd=ppd(tgt,p,v);arf(np.clip(mxr(Tt,*apd(qd,qc,om)),0,None))
 elif st<1431:tgt=np.array([3.24,1.56,1]);Tt,qd=ppd(tgt,p,v);arf(np.clip(mxr(Tt,*apd(qd,qc,om)),0,None))
 elif st<1560:
  theta=-np.radians(-84.9);Tt,qd=ppd(np.array([p[0],p[1],1]),p,v,theta)
  arf(np.clip(mxr(Tt,*apd(qd,qc,om)),0,None))
 else:Tt,qd=ppd(np.array([p[0],p[1],1]),p,v);arf(np.clip(mxr(Tt,*apd(qd,qc,om)),0,None))
 scene.step()

print("=== Wall-crossing analysis ===")
print(f"Total rays where raycaster HIT but endpoint crosses x>3: {len(m.wall_xings)}")
print(f"  (these are FALSE POSITIVES — wall correctly detected, but diag check flags any end_x>3.0)")
print(f"Total rays where raycaster MISSED and would have hit wall at x=3: {len(m.wall_penetrations)}")
print(f"  (these are TRUE wall-penetrations — wall invisible to sensor)")

print("\n--- Sample of wall_xings (raycaster hit but endpoint crosses x>3.0 — false positives) ---")
for i, rl in enumerate(m.wall_xings[:5]):
    step, mode, az_i, raw_range, ex, ey, dx, dy, rng = rl
    hit_x = ex + dx * rng
    print(f"  step={step} az={az_i} raw={raw_range:.4f} rng={rng:.4f} hit_x={hit_x:.4f}")

# Count true penetrations where wall is within range
true_pen = 0
for rl in m.wall_penetrations:
    step, mode, az_i, raw, ex, ey, dx, dy, rng = rl
    # Check if wall is within 5m range in the direction's x-component
    if dx < 0:  # heading -X (toward wall from +X side)
        wall_t = (3.005 - ex) / dx
        if 0 < wall_t <= 5.0:
            true_pen += 1
    elif dx > 0:  # heading +X (toward wall from -X side)
        wall_t = (2.995 - ex) / dx
        if 0 < wall_t <= 5.0:
            true_pen += 1

# Check step 1656 offending ray specifically
step_1656 = [rl for rl in m.wall_penetrations if rl[0] == 1656]
print(f"\nStep 1656 penetrations: {len(step_1656)}")
if step_1656:
    for rl in step_1656:
        step, mode, az_i, raw, ex, ey, dx, dy, rng = rl
        print(f"  az={az_i} raw={raw:.3f} emitter=({ex:.3f},{ey:.3f}) dir=({dx:.3f},{dy:.3f})")
        wall_dist = (3.005 - ex) / dx if dx != 0 else float('inf')
        print(f"    wall t={wall_dist:.3f} (need >0 and <5)")

print(f"\n--- Shows ALL {len(m.wall_penetrations)} true penetrations below ---")
for rl in m.wall_penetrations:
    step, mode, az_i, raw, ex, ey, dx, dy, rng = rl
    hit_str = "HIT" if raw>=0 else "MISS"
    print(f"  step={step} az={az_i} {hit_str} raw={raw:.3f} emitter=({ex:.3f},{ey:.3f}) dir=({dx:.3f},{dy:.3f})")

# Check the frontier-2 cell at grid (56, 81)
tx = 56
ty = 81
occ = np.zeros((m.h, m.w))
mv = m.views > 0
occ[mv & (m.hits >= 1)] = 1.0
occ[mv & (m.hits == 0)] = 0.5
print(f"\n=== Frontier #2 cell (2.52,4.52) ===")
print(f"  occ={occ[ty,tx]:.1f} views={m.views[ty,tx]} hits={m.hits[ty,tx]}")
print(f"  {'UNKNOWN (expected)' if occ[ty,tx]==0 else 'FREE (bad)'}")
print(f"  {'PASS' if occ[ty,tx]==0 else 'FAIL'}")
