"""Process/IO bridge; no USD or renderer calls on the driving thread."""
import json
from pathlib import Path
import queue
import subprocess
import threading
import numpy as np
from PIL import Image


class BackgroundCapture:
    def __init__(self,project,output,scene,cameras):
        self.mailbox=queue.Queue()
        self.ready=threading.Event()
        self.cameras=cameras
        calibration=output/'sensor_calibration.json'
        calibration.write_text(json.dumps(cameras))
        self.process=subprocess.Popen(
            [str(project/'runtime/python.bat'),str(project/'simulation/sensor_renderer.py'),
             '--scene',str(scene),'--calibration',str(calibration)],cwd=project,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,bufsize=1,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
        self.reader=threading.Thread(target=self._read,args=(output/'sensor_renderer.log',),daemon=True)
        self.reader.start()

    def _read(self,log):
        try:
            with log.open('w',encoding='utf-8') as stream:
                for line in self.process.stdout:
                    stream.write(line);stream.flush()
                    if line.startswith('SENSOR_READY'):self.ready.set()
                    if not line.startswith('SENSOR_RESULT '):continue
                    response=json.loads(line[len('SENSOR_RESULT '):])
                    if not response.get('error'):
                        directory=Path(response['capture'])
                        response['snapshots']=[(c['name'],np.asarray(Image.open(directory/(c['name']+'.png')).convert('RGB')))
                                               for c in self.cameras]
                    self.mailbox.put(response)
        except Exception as error:
            self.mailbox.put({'error':'Sensor reader: '+str(error)})

    def submit(self,request,metadata,lidar_points):
        def write():
            try:
                directory=Path(request['capture'])
                np.save(directory/'lidar.npy',lidar_points)
                (directory/'capture.json').write_text(json.dumps(metadata,indent=2))
                self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
            except Exception as error:
                self.mailbox.put({'id':request['id'],'error':'Sensor request: '+str(error)})
        threading.Thread(target=write,daemon=True).start()

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"shutdown":true}\n');self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (OSError,subprocess.TimeoutExpired):
                self.process.kill();self.process.wait(timeout=10)
