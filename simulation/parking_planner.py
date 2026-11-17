"""Sensor-built cost map and footprint-checked car-kinematic parking search."""
import heapq
import math
import time
import cv2
import numpy as np

def wrap(a):return (a+math.pi)%(2*math.pi)-math.pi

class EvidenceMap:
    def __init__(self):
        self.origin=np.array([-33.,-30.]);self.res=.2
        self.free=np.zeros((250,280),bool)
        self.obstacle=np.zeros_like(self.free)
        self.semantic=np.zeros_like(self.free,dtype=np.float32)
    def cells(self,points):
        ij=np.floor((points-self.origin)/self.res).astype(int)
        good=(ij[:,0]>=0)&(ij[:,0]<self.free.shape[1])&(ij[:,1]>=0)&(ij[:,1]<self.free.shape[0])
        return ij[good]
    def update(self,capture,labels,confidence,reference,footprint,min_solid_height=.30):
        from lidar_evidence import update_evidence
        return update_evidence(self,capture,labels,confidence,reference,footprint,min_solid_height)
    def save(self,path):
        np.savez_compressed(path,free=self.free,obstacle=self.obstacle,semantic=self.semantic,origin=self.origin,res=self.res)


class Planner:
    def __init__(self,evidence,footprint,route_preferences=None):
        self.map=evidence;self.length,self.width=footprint
        self.route_preferences=route_preferences
        self._clearance_cache=None
        self._clearance_hits=0
        self.offsets=[]
        for i in range(72):
            yaw=i*2*math.pi/72
            yy,xx=np.mgrid[-18:19,-18:19]
            x=xx*evidence.res;y=yy*evidence.res
            longitudinal=x*math.cos(yaw)+y*math.sin(yaw)-1.35
            lateral=-x*math.sin(yaw)+y*math.cos(yaw)
            mask=(abs(longitudinal)<=self.length/2+.13)&(abs(lateral)<=self.width/2+.13)
            self.offsets.append(np.stack([xx[mask],yy[mask]],1))
    def clearance(self,pose):
        x,y,yaw=pose[:3]
        cell_x=math.floor((x-self.map.origin[0])/self.map.res)
        cell_y=math.floor((y-self.map.origin[1])/self.map.res)
        heading=int(round(yaw/(2*math.pi)*72))%72
        key=(cell_x,cell_y,heading)
        if self._clearance_cache is not None and key in self._clearance_cache:
            self._clearance_hits+=1
            return self._clearance_cache[key]
        result=self._cell_clearance(cell_x,cell_y,heading)
        if self._clearance_cache is not None:self._clearance_cache[key]=result
        return result
    def _cell_clearance(self,cell_x,cell_y,heading):
        points=self.offsets[heading]+(cell_x,cell_y)
        if (points<0).any() or (points[:,0]>=self.map.free.shape[1]).any() or (points[:,1]>=self.map.free.shape[0]).any():return False,0.
        yy,xx=points[:,1],points[:,0]
        if self.map.obstacle[yy,xx].any():return False,0.
        coverage=self.map.free[yy,xx].mean()
        if coverage<.94:return False,0.
        return True,float(self.map.semantic[yy,xx].mean())
    def clearance_diagnostics(self,pose):
        """Describe the same raster footprint used by clearance, without changing it."""
        x,y,yaw=pose[:3]
        base=np.floor((np.asarray([x,y])-self.map.origin)/self.map.res).astype(int)
        points=self.offsets[int(round(yaw/(2*math.pi)*72))%72]+base
        inside=(points[:,0]>=0)&(points[:,1]>=0)&(points[:,0]<self.map.free.shape[1])&(points[:,1]<self.map.free.shape[0])
        cells=points[inside];xx,yy=cells[:,0],cells[:,1]
        blocked=self.map.obstacle[yy,xx]
        unknown=~self.map.free[yy,xx]&~blocked
        def positions(selected):return (self.map.origin+(cells[selected]+.5)*self.map.res).tolist()
        return {'pose':list(map(float,pose[:3])),'footprint_m':[self.length+.26,self.width+.26],
                'obstacle_cells':int(blocked.sum()),'unobserved_cells':int(unknown.sum()),
                'out_of_bounds_cells':int((~inside).sum()),
                'free_fraction':float(self.map.free[yy,xx].sum()/len(points)),
                'blocked_xy':positions(blocked),'unobserved_xy':positions(unknown)}
    def plan(self,start,goal,timeout=20.,coarse_to_fine=False):
        # Evidence is immutable during a search. Never reuse this cache after
        # a search or across a refreshed sensor map.
        self._clearance_cache={}
        self._clearance_hits=0
        try:
            path,report=self._plan(start,goal,timeout,coarse_to_fine)
            report['clearance_cache_hits']=self._clearance_hits
            report['clearance_cache_entries']=len(self._clearance_cache)
            return path,report
        finally:
            self._clearance_cache=None
    def _plan(self,start,goal,timeout,coarse_to_fine):
        begin=time.perf_counter()
        start=tuple(start);goal=tuple(goal)
        if not self.clearance(start)[0]:return None,{'reason':'start footprint lacks measured clearance'}
        if not self.clearance(goal)[0]:return None,{'reason':'parking goal footprint is blocked or unobserved'}
        def key(p):return (round(p[0]/.4),round(p[1]/.4),round(wrap(p[2])/math.radians(5)),int(p[3]))
        # Weighted search prioritizes reaching a useful parking heading. It finds
        # feasible paths; it does not claim globally shortest/optimal routes.
        def heuristic(p):return 1.15*math.hypot(p[0]-goal[0],p[1]-goal[1])+2.5*abs(wrap(p[2]-goal[2]))
        states=[(*start,1)];parents=[None];segments=[None];costs=[0.]
        queue=[(heuristic(start),0)];best={key(states[0]):0.};expanded=0
        while queue and expanded<35000 and time.perf_counter()-begin<timeout:
            _,index=heapq.heappop(queue);p=states[index]
            # A cheaper arrival may have superseded this queued state.
            if costs[index]>best.get(key(p),float('inf')):continue
            expanded+=1
            if math.hypot(p[0]-goal[0],p[1]-goal[1])<.25 and abs(wrap(p[2]-goal[2]))<math.radians(7):
                path=[]
                while parents[index] is not None:
                    path.append(segments[index]);index=parents[index]
                path=np.concatenate(path[::-1]) if path else np.array([[*start,1.]])
                return path,{'expanded':expanded,'seconds':time.perf_counter()-begin,'semantic_cost_used':True,'reason':'planned','coarse_to_fine':coarse_to_fine}
            # Street search advances 1.8 m at a time, then returns to 0.6 m
            # primitives near the approach. Every intermediate 0.1 m pose
            # still receives the original full-footprint clearance check.
            primitive_steps=18 if coarse_to_fine and math.hypot(p[0]-goal[0],p[1]-goal[1])>6. else 6
            for direction in [1,-1]:
                for steer in [-.65,-.4,0.,.4,.65]:
                    cur=list(p[:3]);segment=[];penalty=0.;valid=True
                    for _ in range(primitive_steps):
                        ds=direction*.1;middle=cur[2]+ds*math.tan(steer)/2.7/2
                        cur[0]+=ds*math.cos(middle);cur[1]+=ds*math.sin(middle)
                        cur[2]=wrap(cur[2]+ds*math.tan(steer)/2.7)
                        ok,risk=self.clearance(cur)
                        if not ok:valid=False;break
                        penalty+=risk
                        segment.append([*cur,direction])
                        # Parking may finish between primitive endpoints; fixed
                        # 0.6 m endpoints otherwise skip a reachable narrow goal.
                        if math.hypot(cur[0]-goal[0],cur[1]-goal[1])<.2 and abs(wrap(cur[2]-goal[2]))<math.radians(7):break
                    if not valid:continue
                    new=(*cur,direction)
                    cost=costs[index]+len(segment)*.1*(1 if direction==1 else 1.35)+.1*abs(steer)+penalty*.7+(1.5 if direction!=p[3] else 0)
                    if self.route_preferences is not None:
                        cost+=self.route_preferences.transition_cost(p,new,len(segment)*.1)
                    k=key(new)
                    if cost>=best.get(k,float('inf')):continue
                    best[k]=cost;states.append(new);parents.append(index);segments.append(np.asarray(segment));costs.append(cost)
                    heapq.heappush(queue,(cost+heuristic(new),len(states)-1))
        termination='timeout' if time.perf_counter()-begin>=timeout else 'expansion_limit' if expanded>=35000 else 'search_exhausted'
        return None,{'reason':'no footprint-safe path through observed space','expanded':expanded,'seconds':time.perf_counter()-begin,'coarse_to_fine':coarse_to_fine,'termination':termination}
