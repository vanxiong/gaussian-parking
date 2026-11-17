"""Finite traffic costs, separate from hard footprint collision clearance."""
import math
import numpy as np


class LanePreferences:
    def __init__(self, evidence, settings):
        self.origin=evidence.origin;self.res=evidence.res
        self.dividers=settings.get('dividers',[])
        yy,xx=np.indices(evidence.free.shape)
        x=self.origin[0]+(xx+.5)*self.res;y=self.origin[1]+(yy+.5)*self.res
        # Compile continuous lane costs once; primitive expansion uses one lookup.
        self.cost=np.zeros((72,*evidence.free.shape),dtype=np.float32)
        for zone in settings.get('zones',[]):
            l,b,r,t=zone['bounds'];mask=(x>=l)&(x<=r)&(y>=b)&(y<=t)
            weight=np.ones_like(x)
            if zone.get('relax_bounds'):
                l,b,r,t=zone['relax_bounds']
                weight[(x>=l)&(x<=r)&(y>=b)&(y<=t)]=zone.get('relax_factor',.2)
            for i in range(72):
                yaw=i*2*math.pi/72;choices=[]
                for lane in zone['lanes']:
                    h=math.radians(lane['heading_deg']);cx,cy=lane['center']
                    lateral=-(x-cx)*math.sin(h)+(y-cy)*math.cos(h)
                    choices.append(zone['center_weight']*np.minimum(lateral*lateral,16)
                                   +zone['direction_weight']*(1-math.cos(yaw-h)))
                self.cost[i][mask]+=np.min(choices,axis=0)[mask]*weight[mask]

    @staticmethod
    def center(p):
        # Planner poses are at the rear axle; markings apply to the body center.
        return (p[0]+1.35*math.cos(p[2]),p[1]+1.35*math.sin(p[2]))

    def transition_cost(self, previous, current, distance):
        a=self.center(previous);b=self.center(current)
        x=int(math.floor((b[0]-self.origin[0])/self.res))
        y=int(math.floor((b[1]-self.origin[1])/self.res))
        travel=current[2]+(math.pi if current[3]<0 else 0)
        heading=int(round(travel/(2*math.pi)*72))%72
        total=float(self.cost[heading,y,x])*distance if 0<=y<self.cost.shape[1] and 0<=x<self.cost.shape[2] else 0.
        for divider in self.dividers:
            sx,sy=divider['start'];ex,ey=divider['end'];dx=ex-sx;dy=ey-sy
            length=math.hypot(dx,dy)
            if not length:continue
            side_a=((a[0]-sx)*dy-(a[1]-sy)*dx)/length
            side_b=((b[0]-sx)*dy-(b[1]-sy)*dx)/length
            along=((b[0]-sx)*dx+(b[1]-sy)*dy)/(length*length)
            if 0<=along<=1:
                total+=distance*divider.get('proximity_weight',0)*max(0,1-abs(side_b))
            # Charge each actual crossing, including repeated zigzag crossings.
            if (side_a<0<=side_b) or (side_b<0<=side_a):
                fraction=side_a/(side_a-side_b)
                ix=a[0]+fraction*(b[0]-a[0]);iy=a[1]+fraction*(b[1]-a[1])
                u=((ix-sx)*dx+(iy-sy)*dy)/(length*length)
                if 0<=u<=1:total+=divider['crossing_cost']
        return total
