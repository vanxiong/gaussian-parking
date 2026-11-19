"""Default driving workspace, applied after Kit has created its dock windows."""
import asyncio


async def apply_driving_layout(sensor_window,controls_window,status_window):
    import omni.kit.app
    import omni.ui as ui
    app=omni.kit.app.get_app()
    for _ in range(3):
        await app.next_update_async()
    workspace=ui.Workspace
    viewport=workspace.get_window('Viewport')
    if viewport is None:
        viewport=next((w for w in workspace.get_windows() if w.title.startswith('Viewport')),None)
    if viewport is None:
        print('DRIVING_LAYOUT: viewport unavailable; leaving panels accessible',flush=True)
        return
    sensors=workspace.get_window(sensor_window.title)
    controls=workspace.get_window(controls_window.title)
    status=workspace.get_window(status_window.title)
    stage=workspace.get_window('Stage')
    properties=workspace.get_window('Property')
    content=workspace.get_window('Content')
    # Keep Stage above the right-hand status and controls, as in the reference.
    if properties is not None:
        controls.dock_in(properties,ui.DockPosition.SAME)
    elif stage is not None:
        controls.dock_in(stage,ui.DockPosition.BOTTOM,.65)
    else:
        controls.dock_in(viewport,ui.DockPosition.RIGHT,.30)
    await app.next_update_async()
    status.dock_in(controls,ui.DockPosition.TOP,.28)
    if content is not None:
        sensors.dock_in(content,ui.DockPosition.SAME)
    else:
        sensors.dock_in(viewport,ui.DockPosition.BOTTOM,.42)
    for _ in range(2):
        await app.next_update_async()
    workspace.set_dock_id_width(controls.dock_id,560)
    workspace.set_dock_id_height(sensors.dock_id,560)
    if stage is not None:
        workspace.set_dock_id_height(stage.dock_id,360)
    workspace.set_dock_id_height(status.dock_id,230)
    for window in (sensors,controls,status):
        window.focus()
    print('DRIVING_LAYOUT: sensors bottom; status middle right; controls bottom right',flush=True)


def schedule_driving_layout(sensor_window,controls_window,status_window):
    return asyncio.ensure_future(apply_driving_layout(sensor_window,controls_window,status_window))
