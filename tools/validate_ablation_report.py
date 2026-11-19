"""Check report provenance, local links, exported tables and paired t intervals."""
import argparse,csv,hashlib,json,math,re
from pathlib import Path
from urllib.parse import unquote
from PIL import Image
from validate_car_appearance_comparison import local

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):
    if not math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-10):raise AssertionError(f'Numeric mismatch: {a} vs {b}')

def main():
    p=argparse.ArgumentParser();p.add_argument('report',type=Path);a=p.parse_args();out=a.report.parent
    evidence=json.loads((out/'study-evidence.json').read_text());body=a.report.read_text(encoding='utf-8')
    if sha(a.report)!=evidence['report_sha256']:raise AssertionError('Report checksum mismatch')
    count=0;intervals=0
    for batch,source in evidence['comparisons'].items():
        path=local(source['path'])
        if sha(path)!=source['sha256']:raise AssertionError('Source comparison checksum changed')
        validation=json.loads(local(source['validation_path']).read_text())
        if not validation['passed']:raise AssertionError('Invalid source comparison')
        d=json.loads(path.read_text());count+=len(d['runs'])
        if len(d['runs'])!=source['predictions']:raise AssertionError('Prediction count changed')
        if d['checkpoint_sha256']!=evidence['checkpoint_sha256']:raise AssertionError('Checkpoint differs')
        pairs=[]
        if batch=='appearance':pairs.append(('simple','realistic',None,d['paired_differences']))
        elif batch=='building':
            pairs.extend([('present','absent',None,d['paired_differences']),('present','absent','common_region',d['common_region_paired_differences'])])
        else:
            pairs.extend([('baseline',variant,None,comparison) for variant,comparison in d['comparisons'].items()])
        for first,second,scope,comparison in pairs:
            left=sorted([r for r in d['runs'] if r['variant']==first],key=lambda r:r['seed'])
            right=sorted([r for r in d['runs'] if r['variant']==second],key=lambda r:r['seed'])
            if len(left)!=20 or [r['seed'] for r in left]!=[r['seed'] for r in right]:raise AssertionError('Wrong pairing')
            for key,record in comparison.items():
                differences=[(b[scope][key]-a[scope][key]) if scope else b[key]-a[key] for a,b in zip(left,right)]
                mean=math.fsum(differences)/20;sd=math.sqrt(math.fsum((x-mean)**2 for x in differences)/19)
                # Independent numeric reference: two-sided 95% Student t critical, df=19.
                half=2.093024054408263*sd/math.sqrt(20)
                close(record['mean'],mean);close(record['sample_sd'],sd)
                close(record['mean_95pct_t_interval'][0],mean-half);close(record['mean_95pct_t_interval'][1],mean+half);intervals+=1
    if count!=140:raise AssertionError('Wrong initial population')
    extension=evidence.get('viewpoint_extension')
    if extension:
        source=local(extension['path']);receipt=json.loads(local(extension['validation_path']).read_text())
        if sha(source)!=extension['sha256'] or not receipt['passed'] or receipt['comparison_sha256']!=sha(source):raise AssertionError('Invalid extension provenance')
        data=json.loads(source.read_text())
        if len(data['runs'])!=400 or receipt['predictions_recomputed']!=400:raise AssertionError('Wrong extension population')
        if data['checkpoint_sha256']!=evidence['checkpoint_sha256']:raise AssertionError('Checkpoint differs')
        for name,digest in extension['copied_files'].items():
            if sha(out/name)!=digest:raise AssertionError('Copied extension file differs')
        count+=len(data['runs'])
    if count!=evidence['formal_prediction_count']:raise AssertionError('Wrong formal-study population')
    for name,key in [('ablation_effects.csv','effect_rows'),('condition_metrics.csv','condition_rows')]:
        with (out/name).open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
        expected=evidence[key]
        if len(rows)!=len(expected):raise AssertionError('Report CSV length mismatch')
        for r,e in zip(rows,expected):
            for k,v in e.items():
                if isinstance(v,(float,int)):close(r[k],v)
                elif r[k]!=v:raise AssertionError('Report CSV label mismatch')
    links=[];images=[]
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',body):
        if target.startswith(('https://','http://','#')):continue
        path=(out/unquote(target.split('#')[0])).resolve()
        if not path.is_file():raise AssertionError('Missing report link: '+str(path))
        links.append(str(path))
        if path.suffix=='.png':
            with Image.open(path) as im:
                im.verify()
            images.append(str(path))
    for name,item in evidence['figure_sources'].items():
        if sha(out/'figures'/(name+'.png'))!=item['sha256']:raise AssertionError('Copied figure differs from source')
    validation={'passed':True,'formal_predictions':count,'paired_intervals_independently_recomputed':intervals,
        'effect_table_rows':len(evidence['effect_rows']),'condition_table_rows':len(evidence['condition_rows']),
        'local_report_links_checked':len(links),'report_images_checked':len(images),
        'report_sha256':sha(a.report),'scope':'Source provenance, existing prediction-validation receipts, paired arithmetic, CSV agreement and local-link/image integrity; chart visual review performed separately.'}
    (out/'validation.json').write_text(json.dumps(validation,indent=2));print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
