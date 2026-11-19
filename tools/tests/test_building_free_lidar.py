"""Analytical ray fixtures for the LiDAR-evidence diagnostic."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnose_building_free import free_ray_cells,hit_cells

SHAPE=(16,16,16)
ORIGIN=np.array([-4.,-4.,-4.])

def test_axis_ray_stops_half_meter_before_return():
    free=free_ray_cells(np.array([[3.,0,0]]),SHAPE,ORIGIN,.5)
    expected=np.array([[x,8,8] for x in range(9,13)])
    assert np.array_equal(np.argwhere(free),expected)
    assert hit_cells(np.array([[3.,0,0]]),SHAPE,ORIGIN,.5)[14,8,8]
    assert not free[14,8,8]

def test_negative_ray_does_not_include_zero_length_start_cell():
    free=free_ray_cells(np.array([[-3.,0,0]]),SHAPE,ORIGIN,.5)
    assert np.array_equal(np.argwhere(free),np.array([[x,8,8] for x in range(3,7)]))

def test_corner_crossings_do_not_mark_off_diagonal_cells():
    free=free_ray_cells(np.array([[3.,3.,3.]]),SHAPE,ORIGIN,.5)
    cells=np.argwhere(free)
    assert len(cells)>2
    assert np.all(cells[:,0]==cells[:,1]) and np.all(cells[:,1]==cells[:,2])

def test_ray_leaves_grid_without_wrapping():
    free=free_ray_cells(np.array([[10.,0,0]]),SHAPE,ORIGIN,.5)
    assert np.array_equal(np.argwhere(free),np.array([[x,8,8] for x in range(9,16)]))

def test_short_returns_leave_no_free_segment():
    assert not free_ray_cells(np.array([[.8,0,0]]),SHAPE,ORIGIN,.5).any()

def test_outside_returns_are_not_clamped_into_boundary_cells():
    hit=hit_cells(np.array([[10.,0,0],[0,0,0]]),SHAPE,ORIGIN,.5)
    assert hit.sum()==1 and hit[8,8,8]
