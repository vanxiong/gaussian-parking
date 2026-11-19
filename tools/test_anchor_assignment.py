"""Exercise both assignment branches, input-order invariance and checkpoint safety."""
import torch


def check_assignment(model):
    from model.lifter.anchor_assignment import evaluation_assignment
    from model.lifter.gaussian_lifter import GaussianLifterLiDAR
    from model.utils.safe_ops import safe_sigmoid, safe_inverse_sigmoid, point_cloud_map
    for device in ['cpu', 'cuda']:
        for count in [0, 3, 8, 17]:
            coords=torch.stack([torch.arange(count), torch.arange(count)%3, torch.arange(count)%5],dim=1).to(device)
            rng=torch.random.get_rng_state().clone()
            cuda_rng=torch.cuda.get_rng_state().clone()
            destination,source=evaluation_assignment(coords,8)
            assert torch.equal(rng,torch.random.get_rng_state())
            assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
            assert len(destination)==len(source)==min(count,8)
            assert len(destination.unique())==len(destination)
            assert len(source.unique())==len(source)
            shuffle=torch.arange(count-1,-1,-1,device=device)
            d2,s2=evaluation_assignment(coords[shuffle],8)
            assert torch.equal(destination,d2)
            assert torch.equal(coords[source],coords[shuffle][s2])
    lifter=GaussianLifterLiDAR(num_anchor=8,embed_dims=4,semantics=True,semantic_dim=17).cuda().eval()
    state={k:v.clone() for k,v in lifter.state_dict().items()}
    coords=torch.tensor([[0,1,2,3],[0,2,3,4],[0,3,4,5],[1,1,2,3]],dtype=torch.int32,device='cuda')
    points=torch.tensor([[1.,2.,0.,255.,0.],[2.,3.,0.,64.,0.],[3.,4.,0.,128.,0.],[4.,5.,0.,255.,0.]],device='cuda')
    kwargs=dict(ms_img_feats=[torch.zeros(2,6,4,2,2,device='cuda')],voxel_lidar_feats=points,coors_batch=coords)
    with torch.inference_mode():
        first=lifter(**kwargs)['representation']
        torch.manual_seed(99)
        second=lifter(**kwargs)['representation']
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        for batch in [0,1]:
            targets=point_cloud_map(points[coords[:,0]==batch])
            assert (torch.cdist(targets,safe_sigmoid(first[batch,:,:3])).min(1).values<1e-4).all()
            destination,source=evaluation_assignment(coords[coords[:,0]==batch,1:],8)
            expected=safe_inverse_sigmoid(points[coords[:,0]==batch,3][source]/255.)
            torch.testing.assert_close(first[batch,destination,10],expected)
        lifter.train()
        torch.manual_seed(1);a=lifter(**kwargs)['representation']
        torch.manual_seed(2);b=lifter(**kwargs)['representation']
        assert not torch.equal(a,b),'Training should retain randomized assignment'
    for k,v in state.items():assert torch.equal(v,lifter.state_dict()[k])
    assert len(model.state_dict())==941
    return {'cpu_and_cuda_sparse_dense_empty_equal':True,'voxel_order_invariance':True,
            'global_cpu_and_cuda_rng_preserved':True,'multi_sample_coordinate_and_intensity_writes':True,
            'training_still_randomized':True,'checkpoint_parameters_unchanged':True}
