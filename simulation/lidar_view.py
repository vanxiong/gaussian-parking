"""Live car-relative LiDAR display; reads scans without changing sensor inputs."""
import numpy as np
from PIL import Image, ImageDraw

CAMERA_ORDER=('CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
              'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT')
CAMERA_VIEWS=('All six cameras','Front left','Front','Front right',
              'Back left','Back','Back right')


def render_scan(points, size=480, radius=30.):
    """Project ego-frame returns: forward is up, left is left; color by height."""
    image=Image.new('RGBA',(size,size),(12,18,28,255))
    draw=ImageDraw.Draw(image)
    center=size//2;scale=(size-20)/(2*radius)
    for distance in (5,10,20,30):
        if distance>radius:continue
        r=distance*scale
        draw.ellipse((center-r,center-r,center+r,center+r),outline=(48,62,78))
        draw.text((center+3,center-r+2),f'{distance} m',fill=(135,150,170))
    pixels=np.asarray(image).copy()
    points=np.asarray(points).reshape(-1,3)
    points=points[np.isfinite(points).all(axis=1)]
    uv=np.rint(np.c_[center-points[:,1]*scale,center-points[:,0]*scale]).astype(int)
    valid=(uv[:,0]>=1)&(uv[:,0]<size-1)&(uv[:,1]>=1)&(uv[:,1]<size-1)
    uv=uv[valid];height=points[valid,2]
    colors=np.tile([75,100,125,255],(len(uv),1))
    colors[height>.25]=[255,205,65,255]
    colors[height>2.]=[100,205,255,255]
    # Draw higher returns last so ground hits cannot cover obstacles.
    order=np.argsort(height);uv=uv[order];colors=colors[order]
    pixels[uv[:,1],uv[:,0]]=colors
    image=Image.fromarray(pixels);draw=ImageDraw.Draw(image)
    draw.rectangle((center-.95*scale,center-2.3*scale,center+.95*scale,center+2.3*scale),outline=(100,255,160),width=2)
    draw.polygon([(center,center-3.6*scale),(center-5,center-2.8*scale),(center+5,center-2.8*scale)],fill=(100,255,160))
    return image


class LidarView:
    def __init__(self,history,car):
        import omni.ui as ui
        self.history,self.car=history,car
        self.last_scan=None
        self.camera_time=None
        self.labels_enabled=True
        self.camera_view=0
        self.raw_capture=None
        self.predicted_capture=None
        self.predicted_single_capture=None
        self.camera_canvas=Image.new('RGBA',(672,480),(12,18,28,255))
        draw=ImageDraw.Draw(self.camera_canvas)
        draw.text((20,200),'Waiting for the first surround-camera capture',fill='white')
        self.provider=ui.ByteImageProvider()
        self.window=ui.Window('Sensor views | surround cameras + LiDAR',width=1175,height=585)
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label('Surround cameras (latest model input) | Live LiDAR: forward up, 30 m range',height=24)
                with ui.HStack(height=24):
                    ui.Label('Camera view',width=85)
                    self.view_picker=ui.ComboBox(0,*CAMERA_VIEWS,width=200)
                    self.view_picker.model.add_item_changed_fn(self.set_camera_view)
                    self.labels_model=ui.SimpleBoolModel(True)
                    ui.CheckBox(model=self.labels_model,width=24)
                    ui.Label('Show model labels')
                    self.labels_model.add_value_changed_fn(self.set_labels_enabled)
                ui.ImageWithProvider(self.provider,width=ui.Percent(100),height=ui.Fraction(1),
                                     fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                ui.Label('Gray: low returns | Yellow: obstacles | Blue: above 2 m',height=24)
                self.status=ui.Label('Waiting for LiDAR scan',height=24)
        self.upload(render_scan([]))

    def update_cameras(self,snapshots,timestamp,predicted=False,single_snapshots=None):
        """Use the existing synchronized capture, without creating render products."""
        if predicted:
            self.predicted_capture=(snapshots,timestamp)
            self.predicted_single_capture=(single_snapshots if single_snapshots is not None else snapshots,timestamp)
        else:
            self.raw_capture=(snapshots,timestamp)
        self.refresh_cameras()

    def set_labels_enabled(self,model):
        self.labels_enabled=model.get_value_as_bool()
        self.refresh_cameras()

    def set_camera_view(self,model,item=None):
        self.camera_view=model.get_item_value_model().as_int
        self.refresh_cameras()

    def clear_cameras(self):
        self.raw_capture=self.predicted_capture=None
        self.predicted_single_capture=None
        self.camera_time=None
        self.camera_canvas=Image.new('RGBA',(672,480),(12,18,28,255))
        ImageDraw.Draw(self.camera_canvas).text((20,200),'Waiting for a new camera capture',fill='white')
        self.last_scan=None

    def refresh_cameras(self):
        predicted=self.labels_enabled and self.predicted_capture is not None
        capture=self.predicted_capture if predicted else self.raw_capture
        if predicted and self.camera_view and self.predicted_single_capture is not None:
            capture=self.predicted_single_capture
        if capture is None:return
        snapshots,timestamp=capture
        cameras=dict(snapshots)
        self.camera_canvas=Image.new('RGBA',(672,480),(12,18,28,255))
        draw=ImageDraw.Draw(self.camera_canvas)
        draw.text((5,8),'MODEL LABELS | confidence >= 0.35' if predicted else 'Model input | labels off' if not self.labels_enabled else 'Model input | waiting for class predictions',fill='white')
        draw.text((5,26),'Latest completed prediction; image held until next result' if predicted else 'Latest synchronized camera capture',fill=(170,185,205))
        if self.camera_view:
            name=CAMERA_ORDER[self.camera_view-1]
            draw.text((8,58),CAMERA_VIEWS[self.camera_view],fill='white')
            if name in cameras:
                tile=Image.fromarray(cameras[name]).convert('RGBA')
                scale=min(660/tile.width,396/tile.height)
                tile=tile.resize((max(1,round(tile.width*scale)),max(1,round(tile.height*scale))),Image.Resampling.LANCZOS)
                self.camera_canvas.paste(tile,((672-tile.width)//2,78+(396-tile.height)//2))
            else:
                draw.text((8,90),'Waiting for this camera',fill=(170,185,205))
        for index,name in enumerate(CAMERA_ORDER if not self.camera_view else ()):
            x=(index%3)*224;y=60+(index//3)*195
            draw.text((x+5,y),name.removeprefix('CAM_').replace('_',' '),fill='white')
            if name not in cameras:continue
            tile=Image.fromarray(cameras[name]).convert('RGBA')
            tile.thumbnail((220,165),Image.Resampling.LANCZOS)
            self.camera_canvas.paste(tile,(x+(224-tile.width)//2,y+22+(165-tile.height)//2))
        self.camera_time=timestamp
        self.last_scan=None

    def upload(self,image):
        # The Kit binding rejects Python bytes; use its uint8 array entry point.
        composite=Image.new('RGBA',(1152,480))
        composite.paste(self.camera_canvas,(0,0));composite.paste(image,(672,0))
        self.pixels=np.array(composite,dtype=np.uint8).reshape(-1)
        self.provider.set_data_array(self.pixels,[composite.width,composite.height])

    def show(self):
        self.window.visible=True
        self.last_scan=None

    def update(self):
        if not self.window.visible:return
        if not self.history.scans:
            if self.last_scan is not None:
                self.upload(render_scan([]))
                self.last_scan=None
            self.status.text='Waiting for LiDAR scan'
            return
        timestamp,transform,xyz,_=self.history.scans[-1]
        age=max(0.,float(self.history.world.current_time)-timestamp)
        camera_age='waiting' if self.camera_time is None else f'{max(0.,float(self.history.world.current_time)-self.camera_time):.2f} s'
        self.status.text=f'LiDAR: {len(xyz):,} returns | scan age {age:.2f} s | camera capture age {camera_age} (simulation time)'
        if timestamp==self.last_scan:return
        from pxr import UsdGeom
        ego=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(self.car)).T
        relative=np.linalg.inv(ego)@transform
        points=xyz@relative[:3,:3].T+relative[:3,3]
        self.upload(render_scan(points))
        self.last_scan=timestamp
