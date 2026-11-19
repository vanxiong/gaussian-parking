"""Convert the pinned, untextured CC0 GLB sedan to a self-contained USD asset.

This intentionally supports only this asset's glTF subset and rejects unsupported
features. Node transforms and PBR material factors are retained; axes become
X forward, Y left, Z up. Run with the project's Windows Python (numpy only).
"""
import hashlib,json,struct
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'assets/parked_sedan'
SOURCE_SHA256='187c9935c40f952837fe2d98200b390415e695196d7b1280c9ec3ebb1bf4961b'

def load_glb(path):
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=SOURCE_SHA256:raise ValueError('Unexpected source asset hash')
    magic,version,length=struct.unpack_from('<III',data)
    if magic!=0x46546c67 or version!=2 or length!=len(data):raise ValueError('Invalid GLB')
    chunks={};offset=12
    while offset<len(data):
        n,kind=struct.unpack_from('<II',data,offset);offset+=8;chunks[kind]=data[offset:offset+n];offset+=n
    doc=json.loads(chunks[0x4e4f534a]);binary=chunks[0x004e4942]
    if doc.get('extensionsRequired') or doc.get('images') or doc.get('skins') or doc.get('animations'):raise ValueError('Unsupported asset features')
    return doc,binary

def accessor(doc,binary,index):
    a=doc['accessors'][index]
    if 'sparse' in a or a.get('normalized'):raise ValueError('Unsupported accessor')
    view=doc['bufferViews'][a['bufferView']]
    if view.get('buffer',0)!=0:raise ValueError('External buffers unsupported')
    dtype=np.dtype({5126:'<f4',5125:'<u4',5123:'<u2',5121:'u1'}[a['componentType']])
    width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]
    offset=view.get('byteOffset',0)+a.get('byteOffset',0);stride=view.get('byteStride',width*dtype.itemsize)
    return np.ndarray((a['count'],width),dtype=dtype,buffer=binary,offset=offset,strides=(stride,dtype.itemsize)).copy()

def node_transform(node):
    if 'matrix' in node:return np.asarray(node['matrix']).reshape(4,4).T
    x,y,z,w=node.get('rotation',[0,0,0,1]);norm=np.linalg.norm([x,y,z,w]);x,y,z,w=np.array([x,y,z,w])/norm
    r=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    t=np.eye(4);t[:3,:3]=r@np.diag(node.get('scale',[1,1,1]));t[:3,3]=node.get('translation',[0,0,0]);return t

def meshes(doc,binary):
    result=[]
    def visit(index,parent):
        node=doc['nodes'][index];t=parent@node_transform(node)
        if 'mesh' in node:
            for part,primitive in enumerate(doc['meshes'][node['mesh']]['primitives']):
                if primitive.get('mode',4)!=4:raise ValueError('Only triangle meshes supported')
                points=accessor(doc,binary,primitive['attributes']['POSITION'])
                points=points@t[:3,:3].T+t[:3,3]
                points=points[:,[2,0,1]]*np.array([1,-1,1])
                indices=accessor(doc,binary,primitive['indices']).ravel().astype(int)
                if len(indices)%3 or indices.min()<0 or indices.max()>=len(points):raise ValueError('Invalid triangles')
                normals=accessor(doc,binary,primitive['attributes']['NORMAL'])@np.linalg.inv(t[:3,:3])
                normals=normals[:,[2,0,1]]*np.array([1,-1,1]);normals/=np.linalg.norm(normals,axis=1,keepdims=True)
                result.append({'name':node.get('name',str(index)),'points':points,'normals':normals,'indices':indices,'material':primitive['material']})
        for child in node.get('children',[]):visit(child,t)
    for index in doc['scenes'][doc.get('scene',0)]['nodes']:visit(index,np.eye(4))
    return result

def main():
    source=ASSET/'sedan-original.glb';doc,binary=load_glb(source);parts=meshes(doc,binary)
    all_points=np.concatenate([m['points'] for m in parts]);lo=all_points.min(0);hi=all_points.max(0)
    offset=np.array([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]])
    fmt=lambda v:format(float(v),'.8g')
    vec=lambda v:'('+', '.join(fmt(x) for x in v)+')'
    lines=['#usda 1.0','(defaultPrim = "Sedan"',' metersPerUnit = 1',' upAxis = "Z")','def Xform "Sedan" {',' def Scope "Looks" {']
    for i,m in enumerate(doc['materials']):
        pbr=m.get('pbrMetallicRoughness',{});color=pbr.get('baseColorFactor',[1,1,1,1])
        lines += [f'  def Material "Mat{i}" {{',f'   token outputs:surface.connect = </Sedan/Looks/Mat{i}/Shader.outputs:surface>',
            '   def Shader "Shader" {','    uniform token info:id = "UsdPreviewSurface"',f'    color3f inputs:diffuseColor = {vec(color[:3])}',
            f'    float inputs:metallic = {fmt(pbr.get("metallicFactor",1))}',f'    float inputs:roughness = {fmt(pbr.get("roughnessFactor",1))}',
            '    token outputs:surface','   }','  }']
    lines+=[' }']
    for i,m in enumerate(parts):
        pts=m['points']-offset
        lines += [f' def Mesh "Part{i:03d}" (prepend apiSchemas = ["MaterialBindingAPI"]) {{',f'  custom string sourceName = {json.dumps(m["name"])}',
            '  point3f[] points = ['+', '.join(vec(p) for p in pts)+']',
            '  int[] faceVertexCounts = ['+', '.join(['3']*(len(m['indices'])//3))+']',
            '  int[] faceVertexIndices = ['+', '.join(str(n) for n in m['indices'])+']',
            '  normal3f[] normals = ['+', '.join(vec(n) for n in m['normals'])+'] (interpolation = "vertex")',
            '  uniform token subdivisionScheme = "none"','  uniform bool doubleSided = true',
            f'  rel material:binding = </Sedan/Looks/Mat{m["material"]}>',' }']
    lines+=['}'];destination=ASSET/'sedan.usda';destination.write_text('\n'.join(lines),encoding='utf-8')
    record={'source_url':'https://www.innerscene.com/tools/library/3d-parts/sedan-car-42109739','license':'CC0-1.0',
        'source_sha256':SOURCE_SHA256,'usd_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
        'source_bounds_xyz':[lo.tolist(),hi.tolist()],'normalized_size_xyz':(hi-lo).tolist(),
        'mesh_count':len(parts),'triangles':sum(len(m['indices'])//3 for m in parts),'paint_material':'Looks/Mat2',
        'axes':'X forward, Y left, Z up; centered XY; bottom at Z=0; meters',
        'ground_truth':'Appearance-only ablation uses original simple-car geometry and collision shapes in both variants.'}
    (ASSET/'asset.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))

if __name__=='__main__':main()
