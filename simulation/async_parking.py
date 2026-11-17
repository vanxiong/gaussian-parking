"""Timestamp checks and delayed-path handover, independent of the simulator."""
import math
import numpy as np


def perception_stale(sim_time, wall_time, captured_sim, captured_wall, max_wall_age=8.):
    if captured_sim is None or captured_wall is None:
        return True
    return wall_time-captured_wall>max_wall_age or sim_time-captured_sim>4.


def align_path(path, current, max_distance=.65, max_heading=math.radians(20)):
    """Trim only the initial gear segment; never jump across a reverse cusp."""
    if path is None or len(path)==0:
        return None
    changes=np.flatnonzero(path[:,3]!=path[0,3])
    end=int(changes[0]) if len(changes) else len(path)
    nearest=int(np.argmin(np.linalg.norm(path[:end,:2]-current[:2],axis=1)))
    heading=abs((path[nearest,2]-current[2]+math.pi)%(2*math.pi)-math.pi)
    if np.linalg.norm(path[nearest,:2]-current[:2])>max_distance or heading>max_heading:
        return None
    # Validate this short connection against the latest measured footprint map.
    target=path[nearest]
    connection=np.tile(target,(6,1))
    connection[:,:2]=np.linspace(current[:2],target[:2],6)
    dyaw=(target[2]-current[2]+math.pi)%(2*math.pi)-math.pi
    connection[:,2]=current[2]+np.linspace(0,dyaw,6)
    return np.concatenate([connection,path[nearest+1:]])
