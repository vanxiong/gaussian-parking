"""Human-readable driving state, independent of the Isaac UI."""

def driving_status(mode, *, has_path, stale, planning_seconds=None, hold_reason='', destination='clear'):
    """Return title, explanation, and semantic color for the current controller state."""
    terminal = {
        'collision_audit_stop': ('COLLISION — STOPPED', 'Collision watchdog detected overlap; brakes applied.', 'red'),
        'parked': ('PARKED', 'Destination reached.', 'green'),
        'no_safe_path': ('DESTINATION BLOCKED', 'Obstacle evidence overlaps the destination footprint; brakes applied.', 'red') if destination=='blocked' else ('NO SAFE PATH', 'No valid route to the mapped destination; brakes applied.', 'red'),
        'placement_blocked': ('PLACEMENT BLOCKED', 'Choose a clear starting pose.', 'red'),
        'stopped': ('STOPPED', 'Driving is paused.', 'amber'),
    }
    if mode in terminal:
        return terminal[mode]
    search = None if planning_seconds is None else f'Path search running for {planning_seconds:.1f}s.'
    if search and not has_path:
        detail = search + ' Car is waiting with brakes applied.'
        if stale:
            detail += ' Waiting for fresh sensor evidence too.'
        return 'SEARCHING FOR PATH', detail, 'amber'
    if stale:
        return 'WAITING FOR PERCEPTION', 'Brakes applied: sensor evidence is missing or stale.' + (' ' + search if search else ''), 'amber'
    if not has_path:
        return 'WAITING FOR PATH', 'Waiting for fresh perception or the next path-search attempt.', 'amber'
    if hold_reason:
        return 'BRAKING / WAITING', hold_reason + (' ' + search if search else ''), 'amber'
    if search:
        return 'NAVIGATING / REPLANNING', 'Following the current path. ' + search, 'blue'
    if destination=='unobserved':
        return 'APPROACHING TO OBSERVE', 'Destination is unseen. Following mapped pavement at up to 1 m/s while checking observed braking clearance.', 'blue'
    return 'NAVIGATING', 'Following the path to the parking destination.', 'green'
