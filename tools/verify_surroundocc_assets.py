"""Check every selected occupancy/depth record without opening a GPU model."""
import json,time
import numpy as np
from install_surroundocc_inputs import ROOT,DATA,REPORTS

def main():
    started=time.perf_counter();selection=json.loads((DATA/'validation-files.json').read_text())
    histogram=np.zeros(17,np.int64);occupied=0;depth_points=0
    for name in selection['labels']:
        values=np.load(DATA/'surroundocc/samples'/name,allow_pickle=False)
        if values.ndim!=2 or values.shape[1]!=4 or values.dtype.kind not in 'iu':raise ValueError('Invalid label format: '+name)
        if (values<0).any() or (values[:,:3]>=np.array([200,200,16])).any() or (values[:,3]>16).any():raise ValueError('Invalid occupancy coordinate/class: '+name)
        histogram+=np.bincount(values[:,3],minlength=17);occupied+=len(values)
    for name in selection['depths']:
        values=np.fromfile(DATA/'nuscenes/depth_gt'/name,dtype='<f4').reshape(-1,3)
        if not np.isfinite(values).all() or (values[:,2]<=0).any():raise ValueError('Invalid depth triples: '+name)
        depth_points+=len(values)
    report={'passed':True,'label_files_checked':len(selection['labels']),'depth_files_checked':len(selection['depths']),
        'label_entries':occupied,'label_class_histogram':histogram.tolist(),'depth_triples':depth_points,
        'semantics':'Original sparse XYZ voxel indices and class IDs, not a reformatted occupancy grid',
        'elapsed_seconds':time.perf_counter()-started}
    (REPORTS/'asset-format-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
