"""Resident frozen GaussianFormer3D worker, consuming JSON-lines from Windows."""
import json
from pathlib import Path
import re
import sys
import time
import numpy as np
import torch
import cv2
from run_gaussianformer3d import ROOT,build_model,build_queries,infer_capture
sys.path.insert(0,str(ROOT/'simulation'))
from parking_planner import EvidenceMap

torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
torch.manual_seed(42);np.random.seed(42)
network,config,tensors=build_model()
queries=build_queries(config)
evidence=EvidenceMap()
print('WORKER_READY',flush=True)
for line in sys.stdin:
    request={}
    try:
        request=json.loads(line)
        started=time.perf_counter()
        path=request['capture']
        match=re.match(r'^([A-Za-z]):[\\/](.*)$',path)
        if match:path='/mnt/'+match[1].lower()+'/'+match[2].replace('\\','/')
        capture=Path(path).resolve()
        capture.relative_to(ROOT)
        if request.get('test_delay'):time.sleep(float(request['test_delay']))
        if request.get('reset'):evidence=EvidenceMap()
        raw,confidence,reference,metadata,stats=infer_capture(network,config,capture,queries)
        labels=raw.copy();labels[labels==17]=0
        metrics=evidence.update(capture,labels,confidence,reference,request['footprint'],request.get('body_clearance_height',.30))
        np.save(capture/'raw_labels.npy',raw);np.save(capture/'labels.npy',labels);np.save(capture/'confidence.npy',confidence)
        evidence.save(capture/'costmap.npz')
        response={'id':request['id'],'phase':'perception','model':'GaussianFormer3D','frozen':True,
                  'strict_tensor_count':tensors,'metrics':metrics,'capture_pose':request['start'],
                  'world_from_reference':reference.tolist(),'seconds':time.perf_counter()-started,**stats,
                  'ground_truth_labels_used':False,'lidar_sweeps':metadata['lidar']['sweeps']}
        (capture/'driving_prediction.json').write_text(json.dumps(response,indent=2))
        print('WORKER_RESULT '+json.dumps(response),flush=True)
    except Exception as error:
        import traceback
        traceback.print_exc()
        print('WORKER_RESULT '+json.dumps({'id':request.get('id'),'error':str(error)}),flush=True)
