"""Distinguish unseen space from sensed obstacles using the controller footprint."""
import math
import numpy as np

def destination_state(costmap,goal,footprint):
    if costmap is None:return 'unobserved'
    a=np.arange(-footprint[0]/2-.12,footprint[0]/2+.13,.15)+1.35
    b=np.arange(-footprint[1]/2-.12,footprint[1]/2+.13,.15)
    xx,yy=np.meshgrid(a,b);yaw=goal[2]
    xy=np.stack([goal[0]+xx*math.cos(yaw)-yy*math.sin(yaw),goal[1]+xx*math.sin(yaw)+yy*math.cos(yaw)],-1).reshape(-1,2)
    ij=np.floor((xy-costmap['origin'])/float(costmap['res'])).astype(int)
    if (ij<0).any() or (ij[:,0]>=costmap['free'].shape[1]).any() or (ij[:,1]>=costmap['free'].shape[0]).any():return 'outside_map'
    if costmap['obstacle'][ij[:,1],ij[:,0]].any():return 'blocked'
    return 'clear' if costmap['free'][ij[:,1],ij[:,0]].mean()>=.92 else 'unobserved'
