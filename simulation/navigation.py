"""Single-goal vehicle navigation on a pavement map with sensed obstacle costs.

Layout supplies route connectivity across unseen pavement. It never overrides the
controller's separate requirement for measured braking clearance.
"""
import json
import math
import time
from pathlib import Path
import numpy as np
from parking_planner import EvidenceMap,Planner
from lane_preferences import LanePreferences
from destination_evidence import destination_state


def navigation_map(evidence,layout_path):
    result=EvidenceMap()
    result.origin=evidence.origin.copy();result.res=evidence.res
    result.free=np.zeros_like(evidence.free)
    yy,xx=np.indices(result.free.shape)
    x=xx*result.res+result.origin[0]+result.res/2
    y=yy*result.res+result.origin[1]+result.res/2
    layout=json.loads(Path(layout_path).read_text())
    for left,bottom,right,top in layout['rectangles']:
        result.free|=(x>=left)&(x<=right)&(y>=bottom)&(y<=top)
    result.obstacle=evidence.obstacle.copy()
    result.semantic=evidence.semantic.copy()
    result.free[result.obstacle]=False
    return result


def plan_to_destination(evidence,footprint,start,goal,layout_path,timeout=40.,coarse_to_fine=True,terminal_direction=None):
    state=destination_state(vars(evidence),goal,footprint)
    path,report=_plan_to_destination(evidence,footprint,start,goal,layout_path,timeout,coarse_to_fine,terminal_direction)
    if path is not None and terminal_direction is not None and int(path[-1,3])!=terminal_direction:
        path=None;report['reason']='no path with requested parking entry direction'
    report['requested_terminal_direction']=terminal_direction
    report['destination_state']=state
    report['approach_to_observe']=path is not None and state=='unobserved'
    if path is None and report.get('failure_stage')=='goal':
        diag=report.get('clearance_diagnostics',{})
        report['goal_rejection']='obstacle' if diag.get('obstacle_cells') else 'outside_map' if diag.get('out_of_bounds_cells') else 'unmapped_pavement'
        report['reason']={'obstacle':'destination footprint overlaps obstacle evidence','outside_map':'destination footprint extends outside the map','unmapped_pavement':'destination footprint lacks mapped pavement'}[report['goal_rejection']]
    return path,report


def _plan_to_destination(evidence,footprint,start,goal,layout_path,timeout,coarse_to_fine,terminal_direction=None):
    # Reserve additional clearance for the search's discrete cells and yaw bins.
    search_footprint=[footprint[0]+.30,footprint[1]+.30]
    route_map=navigation_map(evidence,layout_path)
    layout=json.loads(Path(layout_path).read_text())
    preferences=LanePreferences(route_map,layout['lane_preferences']) if layout.get('lane_preferences') else None
    planner=Planner(route_map,search_footprint,route_preferences=preferences)
    begin=time.perf_counter();route=None;report={'reason':'no aligned parking approach on navigable pavement'}
    goal=np.asarray(goal)
    if not planner.clearance(goal)[0]:
        return None,{'reason':'parking goal footprint is blocked or unobserved','navigation':'single final pose; pavement layout + sensed obstacles','preset_intermediate_goals':0,
                     'clearance_diagnostics':planner.clearance_diagnostics(goal),'failure_stage':'goal'}
    # If already aligned in the approach corridor, use its remaining portion.
    start=np.asarray(start)
    axis=np.array([math.cos(goal[2]),math.sin(goal[2])])
    offset=goal[:2]-start[:2]
    along=float(offset@axis);lateral=abs(float(offset@np.array([-axis[1],axis[0]])))
    turn=(goal[2]-start[2]+math.pi)%(2*math.pi)-math.pi
    if abs(along)<=5. and lateral<.25 and abs(turn)<math.radians(7):
        direction=1 if along>=0 else -1
        count=max(2,math.ceil(np.linalg.norm(offset)/.1)+1)
        tail=np.tile(np.r_[goal,direction],(count,1))
        tail[:,:2]=np.linspace(start[:2],goal[:2],count)
        tail[:,2]=start[2]+np.linspace(0,turn,count)
        if (terminal_direction is None or direction==terminal_direction) and all(planner.clearance(p)[0] for p in tail):
            return tail,{'reason':'planned','seconds':time.perf_counter()-begin,'navigation':'single final pose; pavement layout + sensed obstacles',
                'preset_intermediate_goals':0,'generated_terminal_approach_m':abs(along),'terminal_direction':direction}
    # Prefer the stable long approach, but do not require a 5 m straight corridor.
    attempts=[];rejected=[]
    directions=(1,-1) if terminal_direction is None else (terminal_direction,)
    for length,direction in [(length,direction) for length in (5.,3.,1.5) for direction in directions]+[(0.,directions[0])]:
        approach=goal.copy();approach[:2]-=direction*length*axis
        tail=np.c_[np.linspace(approach,goal,max(2,math.ceil(length/.1)+1)),np.full(max(2,math.ceil(length/.1)+1),direction)]
        failure=next((p for p in tail if not planner.clearance(p)[0]),None)
        if failure is not None:
            rejected.append({'length_m':length,'direction':direction,'clearance':planner.clearance_diagnostics(failure)})
            continue
        remaining=timeout-(time.perf_counter()-begin)
        if remaining<=0:break
        curved=length==0.
        prefix,report=planner.plan(start,approach,timeout=min(5.,remaining*.25),
                                   coarse_to_fine=coarse_to_fine and not curved)
        # Longer primitives can miss a tight maneuver. Retry the original
        # fine search within the same overall time budget.
        if prefix is None and coarse_to_fine and not curved:
            coarse_report=report
            remaining=timeout-(time.perf_counter()-begin)
            if remaining>0:
                prefix,report=planner.plan(start,approach,timeout=min(4.,remaining*.25))
                report['coarse_attempt']=coarse_report
        attempts.append({'length_m':length,'direction':direction,'planning':dict(report)})
        if prefix is not None and (not curved or terminal_direction is None or int(prefix[-1,3])==terminal_direction):
            route=prefix if curved else np.concatenate([prefix,tail])
            report['generated_terminal_approach_m']=length
            report['terminal_direction']=int(route[-1,3])
            report['approach_type']='curved' if curved else 'straight'
            break
    # Traffic preferences are soft. A final direct curved search can relax
    # them while retaining the identical hard footprint and pavement checks.
    remaining=timeout-(time.perf_counter()-begin)
    if route is None and remaining>0 and preferences is not None:
        planner.route_preferences=None
        route,report=planner.plan(start,goal,timeout=remaining,coarse_to_fine=coarse_to_fine)
        report['lane_preferences_relaxed']=True
        if route is not None:
            report.update(generated_terminal_approach_m=0.,terminal_direction=int(route[-1,3]),approach_type='curved')
    report['approach_attempts']=attempts
    report['rejected_approaches']=rejected
    if route is None:
        report['failure_stage']='search' if attempts else 'approach'
        if rejected:report['clearance_diagnostics']=rejected[0]['clearance']
    report['navigation']='single final pose; pavement layout + sensed obstacles'
    report['preset_intermediate_goals']=0
    report['lane_preferences_used']=preferences is not None and not report.get('lane_preferences_relaxed',False)
    return route,report
