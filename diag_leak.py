import genesis as gs, numpy as np
gs.init(backend=gs.amdgpu)
scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=True))
body_w,body_d,body_h=0.165,0.165,0.0545;mass=0.225;g_=9.81;arm=0.0511
Kp_att,Kd_att,Kp_pos,Kd_pos=0.06,0.018,3.5,3.6
MAX_TILT=20.0;MAX_HACC=g_*np.tan(np.radians(MAX_TILT))
rotors=[np.array([arm,-arm,0.0]),np.array([arm,arm,0.0]),np.array([-arm,-arm,0.0]),np.array([-arm,arm,0.0])]
body=scene.add_entity(gs.morphs.Box(size=(body_w,body_d,body_h),pos=(0,0,1),fixed=False))
scene.add_entity(gs.morphs.Box(size=(7,0.01,2),pos=(2.75,-1.5,1),fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.01,6,2),pos=(-0.5,1,1),fixed=True))
scene.add_entity(gs.morphs.Box(size=(3.5,0.01,2),pos=(1.25,1.5,1),fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.01,3,2),pos=(3,2.75,1),fixed=True))  # inner_right wall at x=3
scene.add_entity(gs.morphs.Box(size=(3.5,0.01,2),pos=(4.75,4,1),fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.5,0.8,1.5),pos=(1.5,1,0.75),fixed=True))
scene.add_entity(gs.morphs.Plane())
emit_off=(body_w/2+0.004,0,0)
af=scene.add_sensor(gs.sensors.Raycaster(pattern=gs.sensors.SphericalPattern(fov=(60,0),n_points=(7,1)),entity_idx=body.idx,pos_offset=emit_off,max_range=5,no_hit_value=-1.0))
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
def qfzy(zd,ps):xc=np.array([np.cos(ps),np.sin(ps),0]);zn=zd/np.linalg.norm(zd);yd=np.cross(zn,xc)
 if np.linalg.norm(yd)<1e-10:yd=np.array([0,1,0])
 else:yd/=np.linalg.norm(yd)
 return rtq(np.column_stack([np.cross(yd,zn),yd,zn]))
def ppd(pd,pc,vc,ps=0):e=pd-pc;a=Kp_pos*e-Kd_pos*vc;a[0]=np.clip(a[0],-MAX_HACC,MAX_HACC);a[1]=np.clip(a[1],-MAX_HACC,MAX_HACC);Tv=mass*(a+np.array([0,0,g_]));Tt=np.linalg.norm(Tv);return(Tt,qfzy(Tv/Tt,ps))if Tt>1e-6 else(mass*g_,np.array([1,0,0,0]))
def apd(qd,qc,ow):qe=qm(qconj(qd),qc);s=1 if qe[0]>=0 else -1;return -Kp_att*2*s*qe[1:4]-Kd_att*rtb(qc)@ow
def mxr(T,tx,ty,tz=0):i=1/(4*arm);return np.array([T/4-tx*i-ty*i+tz*i,T/4+tx*i-ty*i-tz*i,T/4-tx*i+ty*i-tz*i,T/4+tx*i+ty*i+tz*i])
def arf(ts):ft=np.zeros(3);tt=np.zeros(3);ys=[1,-1,-1,1]
 for i,(r,f)in enumerate(zip(rotors,ts)):F=np.array([0,0,f]);tt+=np.cross(r,F);tt[2]+=arm*ys[i]*f;ft+=F
 rs.apply_links_external_force(force=ft.reshape(1,3),links_idx=[li],ref="link_origin",local=True)
 rs.apply_links_external_torque(torque=tt.reshape(1,3),links_idx=[li],ref="link_origin",local=True)
def rb(p=(0,0,1),q=None):body.set_pos(p);body.set_quat(q if q is not None else np.array([1,0,0,0]));rs.clear_external_force()

class M:
 def __init__(s,b,res=0.08):
  s.res=res;s.xm,s.xM,s.ym,s.yM=b;s.w=int((s.xM-s.xm)/res)+1;s.h=int((s.yM-s.ym)/res)+1
  s.hits=np.zeros((s.h,s.w),dtype=np.int32);s.views=np.zeros((s.h,s.w),dtype=np.int32)
  s.cell_history={}  # (ix,iy) -> list of (step, mode, az, raw_range, hit, emitter_x, emitter_y, dir_x, dir_y)
 def w2g(s,x,y):return int((x-s.xm)/s.res),int((y-s.ym)/s.res)
 def g2w(s,ix,iy):return s.xm+(ix+0.5)*s.res,s.ym+(iy+0.5)*s.res
 def ib(s,ix,iy):return 0<=ix<s.w and 0<=iy<s.h
 def ur(s,ew,dr,rng,mx,hit,eps=0.004,step=0,mode='F',az_i=0):
  ex,ey=ew[0],ew[1]
  if hit:ex2=ex+dr[0]*(rng+eps);ey2=ey+dr[1]*(rng+eps)
  else:ex2=ex+dr[0]*mx;ey2=ey+dr[1]*mx
  ix0,iy0=s.w2g(ex,ey);ix1,iy1=s.w2g(ex2,ey2)
  n=max(abs(ix1-ix0)+1,abs(iy1-iy0)+1);cr=5
  cells_along=[]
  for i in range(n+1):
   t=i/max(n,1);cx=int(round(ix0+t*(ix1-ix0)));cy=int(round(iy0+t*(iy1-iy0)));trm=(i>=n-1 and hit)
   cells_along.append((cx,cy))
   for ddx in range(-cr,cr+1):
    for ddy in range(-cr,cr+1):
     nx,ny=cx+ddx,cy+ddy
     if not s.ib(nx,ny):continue
     if abs(ddx)+abs(ddy)>cr:continue
     s.views[ny,nx]+=1
     if trm and abs(ddx)+abs(ddy)<=1:s.hits[ny,nx]+=1
  # Log history for the frontier-2 cell (2.52, 4.52) => ix=56, iy=81
  target_cells=[(56,81),(55,81),(57,81),(56,80),(56,82),(55,80),(55,82),(57,80),(57,82),
                (54,81),(58,81),(56,79),(56,83)]
  for (ix,iy) in target_cells:
   if 0<=ix<s.w and 0<=iy<s.h:
    if (ix,iy) in cells_along:
     if (ix,iy) not in s.cell_history:s.cell_history[(ix,iy)]=[]
     s.cell_history[(ix,iy)].append((step,mode,az_i,rng,hit,ex,ey,dr[0],dr[1]))
 def gm(s):
  o=np.zeros((s.h,s.w));m=s.views>0;r=np.zeros_like(s.views,float)
  r[m]=s.hits[m].astype(float)/s.views[m];o[m&(s.hits>=1)]=1.0;o[m&(r<0.02)]=0.5;return o

m=M((-2,7,-2,5),0.08);eps=0.004;az_d=np.linspace(-30,30,7);az_c=np.cos(np.radians(az_d))
for st in range(1700):
 rs.clear_external_force()
 qc=body.get_quat().cpu().numpy();p=body.get_pos().cpu().numpy();v=body.get_vel().cpu().numpy();om=body.get_ang().cpu().numpy()
 df=af.read();rf=df.distances.cpu().numpy().flatten();rfs=rf+eps*az_c
 if st%5==0:dt=scene.add_sensor?  # skip THROW for speed, FLOOD is the issue
 Rb=rtb(qc).T;ew=p+Rb@np.array([emit_off[0],0,0])
 for i in range(7):
  az=np.radians(az_d[i]);db=np.array([np.cos(az),np.sin(az),0]);dw=Rb@db;h=rf[i]>=0;rr=rf[i]if h else 0
  m.ur(ew,dw,rr,5,h,eps,st,'F',i)
 if st<300:tgt=np.array([p[0]+2,0,1]);Tt,qd=ppd(tgt,p,v);arf(np.clip(mxr(Tt,*apd(qd,qc,om)),0,None))
 elif st<1431:tgt=np.array([3.24,1.56,1]);Tt,qd=ppd(tgt,p,v);arf(np.clip(mxr(Tt,*apd(qd,qc,om)),0,None))
 elif st<1560:  # yaw alignment
  theta=-np.radians(-84.9);Tt,qd=ppd(np.array([p[0],p[1],1]),p,v,theta)
  arf(np.clip(mxr(Tt,*apd(qd,qc,om)),0,None))
 else:  # scan
  Tt,qd=ppd(np.array([p[0],p[1],1]),p,v);arf(np.clip(mxr(Tt,*apd(qd,qc,om)),0,None))
 scene.step();p2=body.get_pos().cpu().numpy()

occ=m.gm();h,w=occ.shape

# Frontier #2 cell (2.52, 4.52) = ix=56, iy=81
target_ix, target_iy = 56, 81
print(f"Target cell (2.52, 4.52): ix={target_ix}, iy={target_iy}")
print(f"  occ={occ[target_iy,target_ix]:.1f} views={m.views[target_iy,target_ix]} hits={m.hits[target_iy,target_ix]}")
print(f"  History: {len(m.cell_history.get((target_ix,target_iy),[]))} ray events")

# Wall cells at x=3, y=1.5-4 (inner_right)
print(f"\nInner right wall (x=3) cell at x=2.92,y=2.0: occ={occ[50,61]:.1f} views={m.views[50,61]} hits={m.hits[50,61]}")
print(f"Inner right wall (x=3) cell at x=3.00,y=2.0: occ={occ[50,62]:.1f} views={m.views[50,62]} hits={m.hits[50,62]}")
print(f"Inner right wall (x=3) cell at x=3.08,y=2.0: occ={occ[50,63]:.1f} views={m.views[50,63]} hits={m.hits[50,63]}")

# Show 7x7 around frontier #2
print(f"\n7x7 neighborhood of frontier #2 (ix={target_ix}, iy={target_iy}):")
print(f"{'ix':>4s} {'iy':>4s} {'x':>5s} {'y':>5s} {'occ':>4s} {'views':>5s} {'hits':>4s}")
for dy in range(-3,4):
    for dx in range(-3,4):
        nx, ny = target_ix+dx, target_iy+dy
        if 0<=nx<w and 0<=ny<h:
            wx,wy=m.g2w(nx,ny)
            v=occ[ny,nx]
            os="U"if v==0 else("F"if v==0.5 else"O")
            print(f"{nx:4d} {ny:4d} {wx:5.2f} {wy:5.2f} {os:>4s} {m.views[ny,nx]:5d} {m.hits[ny,nx]:4d}")

# Find first ray that hit the target cell
hist=m.cell_history.get((target_ix,target_iy),[])
if hist:
    hist.sort(key=lambda x:x[0])
    first=hist[0]
    print(f"\nFirst ray event at step {first[0]}: mode={first[1]} az_idx={first[2]} range={first[3]:.3f} hit={first[4]}")
    print(f"  emitter: ({first[5]:.3f},{first[6]:.3f})")
    print(f"  direction: ({first[7]:.4f},{first[8]:.4f})")
    # Check if this ray should have hit the inner right wall
    ex,ey=first[5],first[6];dx,dy=first[7],first[8]
    if dx!=0:
        tw_x3=(3.0-ex)/dx  # t to reach x=3
        if tw_x3>0:
            wy_at_x3=ey+dy*tw_x3
            print(f"  t to x=3: {tw_x3:.3f}, y at x=3: {wy_at_x3:.3f}")
            if 1.5<=wy_at_x3<=4:
                print(f"  *** SHOULD HAVE HIT INNER RIGHT WALL at (3.0, {wy_at_x3:.3f}) ***")
                print(f"  *** Raycaster missed the wall! This is the known Y-wall bug. ***")
            else:
                print(f"  Beam misses wall at x=3 (y={wy_at_x3:.3f} outside [1.5,4])")
        else:
            print(f"  t={tw_x3:.3f} <= 0: beam goes away from wall at x=3")
else:
    print("No ray hit frontier #2 cell directly — cleared via clearance circle")

# Check all events that could have affected frontier #2
print(f"\nAll ray events within 2 cells of frontier #2:")
nearby=set()
for (ix,iy),hlist in m.cell_history.items():
    if abs(ix-target_ix)<=2 and abs(iy-target_iy)<=2:
        nearby.add((ix,iy))
        for ev in hlist:
            wx,wy=m.g2w(ix,iy)
            print(f"  cell ({wx:.2f},{wy:.2f}) step {ev[0]} mode={ev[1]} az={ev[2]} hit={ev[4]}")
