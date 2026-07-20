import sys
with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

replacement = r"""        if state=="EXPLORE" and step%replan_interval==0:
            occ=mapper.get_map()
            inflate_r=4
            inflated=np.ones_like(occ)
            for iy in range(h):
                for ix in range(w):
                    if occ[iy,ix]==0.5: inflated[iy,ix]=0
            for iy in range(h):
                for ix in range(w):
                    if occ[iy,ix]==1.0:
                        for dy in range(-inflate_r,inflate_r+1):
                            for dx in range(-inflate_r,inflate_r+1):
                                nx,ny=ix+dx,iy+dy
                                if 0<=nx<w and 0<=ny<h:
                                    if occ[ny,nx]==0.5: continue
                                    inflated[ny,nx]=1.0
            for tp in trajectory:
                tix,tiy=mapper.w2g(tp[0],tp[1])
                for ddx in range(-2,3):
                    for ddy in range(-2,3):
                        nx,ny=tix+ddx,tiy+ddy
                        if 0<=nx<w and 0<=ny<h: inflated[ny,nx]=0
            fg=np.zeros((h,w),dtype=np.int32)
            for iy in range(1,h-1):
                for ix in range(1,w-1):
                    if occ[iy,ix]==0.5 and inflated[iy,ix]==0:
                        if (occ[iy-1,ix]==0.0 or occ[iy+1,ix]==0.0 or occ[iy,ix-1]==0.0 or
                            occ[iy,ix+1]==0.0):
                            fg[iy,ix]=1
            label=np.zeros((h,w),dtype=np.int32)
            cur_l=1;equiv={}
            for iy in range(1,h-1):
                for ix in range(1,w-1):
                    if fg[iy,ix]==0: continue
                    up=label[iy-1,ix];left=label[iy,ix-1]
                    ul=label[iy-1,ix-1];ur=label[iy-1,ix+1]
                    neighbors=[l for l in [up,left,ul,ur] if l>0]
                    if not neighbors: label[iy,ix]=cur_l;cur_l+=1
                    else:
                        m=min(neighbors);label[iy,ix]=m
                        for n in neighbors:
                            if n!=m: equiv[n]=m
            for iy in range(1,h-1):
                for ix in range(1,w-1):
                    l=label[iy,ix]
                    if l>0:
                        while l in equiv: l=equiv[l]
                        label[iy,ix]=l
            clust_stats={}
            for iy in range(1,h-1):
                for ix in range(1,w-1):
                    l=label[iy,ix]
                    if l>0:
                        if l not in clust_stats: clust_stats[l]=[]
                        clust_stats[l].append((ix,iy))
            clustered=[]
            for l,cells in clust_stats.items():
                if len(cells)<5: continue
                xs=[p[0] for p in cells];ys=[p[1] for p in cells]
                cix=int(round(sum(xs)/len(xs)));ciy=int(round(sum(ys)/len(ys)))
                cix=max(1,min(w-2,cix));ciy=max(1,min(h-2,ciy))
                best_d=999;gx,gy=cix,ciy
                for dd in range(-3,4):
                    for ee in range(-3,4):
                        nx,ny=cix+dd,ciy+ee
                        if 0<=nx<w and 0<=ny<h and occ[ny,nx]==0.5 and inflated[ny,nx]==0:
                            d=abs(dd)+abs(ee)
                            if d<best_d: best_d=d;gx,gy=nx,ny
                cx,cy=mapper.g2w(gx,gy)
                clustered.append({"cx":cx,"cy":cy,"cix":gx,"ciy":gy,"n":len(cells)})
            clustered.sort(key=lambda c:-c['n'])
            clustered=clustered[:20]
            if step==0:
                print(f'  step 0: '+str(len(clustered))+f' clusters from '+str(cur_l-1)+f' components')
            sx,sy=mapper.w2g(pos[0],pos[1])
            best=None;best_score=-1;best_key=None
            for cl in clustered:
                if cl["cx"] < pos[0] + 0.3: continue
                if not (0<=cl["cix"]<w and 0<=cl["ciy"]<h): continue
                if not (inflated[cl["ciy"],cl["cix"]]==0): continue
                if not (occ[cl["ciy"],cl["cix"]]==0.5): continue
                path=astar_path(occ,inflated,(sx,sy),(cl["cix"],cl["ciy"]))
                if path is None: continue
                gain=cl["n"];cost=len(path)*mapper.res
                score=gain/(cost+0.01)
                key=(score,cl["cix"],cl["ciy"])
                if best is None or score>best_score or (abs(score-best_score)<1e-6 and key<best_key):
                    best=cl;best_score=score;best_key=key;current_path=path
            if best is None:
                decisions.append((step,"NO_FRONTIER",pos[0],pos[1]))
                state="RTL"
                print(f'  step {step}: no reachable frontier -> RTL')
            else:
                decisions.append((step,"FRONTIER",best["cx"],best["cy"]))
                frontier_log.append({"step":step,"cx":best["cx"],"cy":best["cy"],"n":best["n"],"score":best_score})
                state="FLY_TO_FRONTIER";wp_idx=0
                if len(frontier_log)<=10:
                    print(f'  step {step}: frontier #'+str(len(frontier_log))+f' ({best["cx"]:.2f},{best["cy"]:.2f}) '+str(best["n"])+f' cells score={best_score:.3f}')
    if state=="FLY_TO_FRONTIER":
        if wp_idx>=len(current_path):
            state="EXPLORE"
        else:
"""

# Find the first line starting with 'if state=="EXPLORE"'
start = None
for i, line in enumerate(lines):
    if 'if state=="EXPLORE"' in line:
        start = i
        break
if start is None:
    print("ERROR: could not find EXPLORE state line")
    sys.exit(1)

# Find the line after the EXPLORE block where 'if state=="FLY_TO_FRONTIER"' starts
end = None
for i in range(start + 1, len(lines)):
    if 'if state=="FLY_TO_FRONTIER"' in line:
        end = i
        break

if end is None:
    end = start + 120  # fallback

new_lines = lines[:start] + replacement.split('\n') + lines[end:]
with open(sys.argv[1], 'w') as f:
    f.write('\n'.join(new_lines))
print(f"Replaced lines {start+1} to {end+1}")
