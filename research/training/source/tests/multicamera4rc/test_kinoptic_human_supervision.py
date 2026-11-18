import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from multicamera4rc.auxiliary_motion import AuxiliaryMotionOptions, auxiliary_motion_losses
from multicamera4rc.datasets.adapters.kinoptic_human import KinopticHumanAdapter
from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.kinoptic_human import attach_kinoptic_human_recipe
from multicamera4rc.datasets.mixture import MultiCameraAdapterDataset, collate_adapter_samples
from multicamera4rc.datasets.partial_dense import materialize_partial_dense_targets
from multicamera4rc.datasets.augmentation import crop_sample
from multicamera4rc.losses import MultiCamera4RCLoss


def fixture_pack(tmp_path, *, admitted=True, split='train', world_rotation=False):
    h,w=48,64;images=np.full((2,2,h,w,3),100,np.uint8);depth=np.full((2,2,h,w),2.,np.float32)
    K=np.tile(np.array([[60.,0,31.5],[0,60.,23.5],[0,0,1.]],np.float32),(2,2,1,1))
    E=np.tile(np.eye(4,dtype=np.float32),(2,2,1,1))
    if world_rotation:E[:,:,:3,:3]=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    E[1,:,0,3]=-.1
    joints=np.array([[-.2,0,2.1],[0,0,2.1],[.2,0,2.1]],np.float32)
    path=tmp_path/'clip.npz'
    np.savez_compressed(path,images=images,depths=depth,K=K,E=E,pixel_support=np.ones_like(depth,bool),
        frame_indices=[100,104],times_ms=[1000.,1133.5],joint_start_world_m=joints,
        joint_end_world_m=joints+np.array([.06,0,0],np.float32),joint_confidence=np.ones(3,np.float32),
        joint_person_id=np.zeros(3,np.int64),joint_index=np.arange(3),
        pseudo_source_uv=np.array([[25.,20.],[30.,20.]],np.float32),
        pseudo_delta_world_m=np.array([[.08,0,0],[.09,0,0]],np.float32),
        pseudo_uncertainty_m=np.array([.04,.04],np.float32),pseudo_cycle_error_px=np.array([.1,.2],np.float32),
        pseudo_cross_view_error_px=np.array([.2,.3],np.float32))
    row=dict(case='fixture',scene='train_capture',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status='prepared',split=split,pseudo_is_true_gt=False,camera_names=['00_01','00_02'],
        frames=[100,104],times_ms=[1000.,1133.5],image_hw=[h,w])
    manifest=dict(kind='kinoptic_human_pack_manifest',schema_version=1,training_admitted=admitted,
        inherited_scene_whitelist=['train_capture'],samples=[row])
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return manifest


def prepared(tmp_path, *, world_rotation=False):
    fixture_pack(tmp_path,world_rotation=world_rotation)
    adapter=KinopticHumanAdapter(tmp_path)
    clip=adapter.load_clip('fixture',[0,1,0,1])
    sample=prepare_clip(clip,num_queries=0,image_mode='native')
    original=sample['targets']['depth'].clone()
    attach_kinoptic_human_recipe(clip,sample,image_mode='native')
    return clip,sample,original


def test_depth_gt_not_replaced_by_joints_or_pseudo_labels(tmp_path):
    clip,sample,depth=prepared(tmp_path)
    t=sample['targets']
    assert torch.equal(t['depth'],depth) and t['depth_valid'].sum()==4*48*64
    assert not t['dense_pair_valid'].any()
    assert not any(k.startswith('sparse_') for k in t)
    assert t['joint_motion_queries'].shape==(12,3)
    assert t['pseudo_motion_queries'].shape==(2,3)
    assert not any('visibility' in k or k.endswith('trajectory') for k in t)
    assert sample['kinoptic_human']['pseudo_is_true_gt'] is False


def test_all_labels_use_depth_scale_and_reference_rotation(tmp_path):
    clip,sample,_=prepared(tmp_path,world_rotation=True)
    batch=collate_adapter_samples([sample]);target=materialize_partial_dense_targets(batch,torch.tensor([[0,1,2,3]]))
    scale=target['scene_scale'][0]
    torch.testing.assert_close(target['joint_motion_displacement'][0,0,1]*scale,torch.tensor([0.,.06,0.]),atol=1e-6,rtol=0)
    torch.testing.assert_close(target['pseudo_motion_displacement'][0,0,1]*scale,torch.tensor([0.,.08,0.]),atol=1e-6,rtol=0)
    torch.testing.assert_close(target['pseudo_motion_uncertainty'][0,0,1]*scale,torch.tensor(.04))
    assert not target['displacement_valid'].any()
    assert target['trajectory_valid'][:,0,0].any() and not target['trajectory_valid'][:,0,1].any()


def test_crop_updates_queries_and_masks_without_changing_motion(tmp_path):
    _,s,_=prepared(tmp_path);before=s['targets']['pseudo_motion_displacement'].clone()
    out=crop_sample(s,torch.tensor([[10,28,28,28]]*4))
    q=out['targets']['pseudo_motion_queries']
    torch.testing.assert_close(q[:,1:],torch.tensor([[-3.,10.],[2.,10.]]))
    assert out['targets']['pseudo_motion_query_valid'].tolist()==[False,True]
    assert not out['targets']['pseudo_motion_valid'][0].any()
    assert torch.equal(out['targets']['pseudo_motion_displacement'],before)


def test_collate_pads_each_label_family_independently(tmp_path):
    _,s,_=prepared(tmp_path);small=copy.deepcopy(s)
    for key in list(small['targets']):
        if key.startswith('pseudo_motion_'):small['targets'][key]=small['targets'][key][:1]
    batch=collate_adapter_samples([s,small])
    assert batch['targets']['joint_motion_queries'].shape==(2,12,3)
    assert batch['targets']['pseudo_motion_queries'].shape==(2,2,3)
    assert not batch['targets']['pseudo_motion_valid'][1,1].any()


def test_candidate_gate_hash_and_capture_whitelist(tmp_path):
    m=fixture_pack(tmp_path,admitted=False)
    with pytest.raises(ValueError,match='not admitted'):KinopticHumanAdapter(tmp_path)
    adapter=KinopticHumanAdapter(tmp_path,allow_candidate=True)
    with (tmp_path/'clip.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='changed'):adapter.load_clip('fixture',[0,1,0,1])
    m['samples'][0]['scene']='held_out_benchmark_capture';(tmp_path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='whitelist'):KinopticHumanAdapter(tmp_path,allow_candidate=True)


def test_real_mixture_route_keeps_all_three_supervision_families(tmp_path):
    fixture_pack(tmp_path)
    dataset=MultiCameraAdapterDataset(sources=[dict(adapter='kinoptic_human',root=str(tmp_path),split='train',
        cameras=2,num_times=2,dense_kind='kinoptic_human',strides=[1])],image_mode='native',
        num_times=2,num_queries=0,supervision='dense',epoch_size=2)
    sample=dataset[0]
    assert sample['targets']['depth_valid'].any()
    assert sample['targets']['joint_motion_valid'].any()
    assert sample['targets']['pseudo_motion_valid'].any()


def prediction_and_target(value=.1):
    parameter=torch.tensor(value,requires_grad=True)
    motion=torch.zeros(1,1,2,8,8,3);motion[:,:,:, :, :,0]=parameter
    pred=dict(displacement=motion,trajectory=motion,source_obs_idx=torch.tensor([[0]]),
        source_valid=torch.tensor([[True]]),source_time_idx=torch.tensor([[100]]),
        query_time_idx=torch.tensor([[100,104]]),motion_confidence=torch.ones(1,1,2,8,8))
    target={'scene_scale':torch.tensor([2.])}
    for prefix,truth in [('joint_motion',.1),('pseudo_motion',.09)]:
        target[prefix+'_queries']=torch.tensor([[[0,3.,3.]]])
        target[prefix+'_query_valid']=torch.tensor([[True]])
        target[prefix+'_displacement']=torch.tensor([[[[0.,0,0],[truth,0,0]]]])
        target[prefix+'_valid']=torch.tensor([[[False,True]]])
        target[prefix+'_confidence']=torch.ones(1,1,2)
    target['pseudo_motion_uncertainty']=torch.tensor([[[0.,.02]]])
    return parameter,pred,target


def test_uncertain_pseudo_band_has_zero_gradient():
    p,pred,target=prediction_and_target()
    terms,loss,active=auxiliary_motion_losses(pred,target,AuxiliaryMotionOptions(pseudo_weight=.02))
    loss.backward()
    assert active and p.grad==0 and loss==0
    assert terms['kinoptic_pseudo_motion_points']==1 and terms['kinoptic_joint_motion_points']==1


def test_real_joint_and_pseudo_losses_are_independent_and_differentiable():
    p,pred,target=prediction_and_target(.2)
    terms,loss,_=auxiliary_motion_losses(pred,target,AuxiliaryMotionOptions(joint_weight=.1,pseudo_weight=.02))
    loss.backward();assert torch.isfinite(p.grad) and p.grad>0
    truth=terms['kinoptic_joint_motion_loss'].detach().clone()
    target['pseudo_motion_displacement']*=100
    terms2,_,_=auxiliary_motion_losses(pred,target,AuxiliaryMotionOptions(joint_weight=.1,pseudo_weight=.02))
    torch.testing.assert_close(terms2['kinoptic_joint_motion_loss'],truth)


def test_invalid_unknown_and_unselected_queries_cannot_train():
    p,pred,target=prediction_and_target(.2)
    target['joint_motion_valid'].zero_();target['pseudo_motion_queries'][...,0]=1
    terms,loss,active=auxiliary_motion_losses(pred,target,AuxiliaryMotionOptions(joint_weight=.1,pseudo_weight=.02))
    assert not active and loss==0 and terms['kinoptic_pseudo_motion_points']==0


def test_loss_config_roundtrip_and_default_backward_compatibility():
    old=MultiCamera4RCLoss().to_config();assert 'auxiliary_motion_options' not in old
    criterion=MultiCamera4RCLoss(auxiliary_motion_options={'joint_weight':.1,'pseudo_weight':.02})
    config=criterion.to_config();assert MultiCamera4RCLoss(**config).to_config()==config


@pytest.mark.parametrize('prefix',['joint_motion','pseudo_motion'])
def test_auxiliary_motion_cannot_take_geometry_only_shortcut(prefix):
    from types import SimpleNamespace
    from multicamera4rc.train import _TrainingStep
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.p=torch.nn.Parameter(torch.tensor(1.));self.config=SimpleNamespace(intrinsics_sharing='none')
        def encode(self,*args,**kwargs):raise AssertionError('motion decoder was bypassed')
        def forward(self,*args,**kwargs):return {'probe':self.p*2}
    model=Model();step=_TrainingStep(model,lambda pred,target:{'loss':pred['probe']})
    targets={'displacement_valid':torch.zeros(1,1,2,8,8,dtype=torch.bool),prefix+'_valid':torch.tensor([[[False,True]]])}
    loss,_=step(torch.zeros(1,1,3,8,8),torch.tensor([[0]]),torch.tensor([[100]]),
        torch.tensor([[100,104]]),torch.tensor([[0]]),torch.tensor([[True]]),targets,False)
    loss.backward();assert model.p.grad==2


def test_empty_auxiliary_labels_support_geometry_only_predictions():
    _,pred,target=prediction_and_target()
    target['joint_motion_valid'].zero_();target['pseudo_motion_valid'].zero_()
    del pred['motion_confidence']
    _,loss,active=auxiliary_motion_losses(pred,target,AuxiliaryMotionOptions(joint_weight=.1,pseudo_weight=.02))
    assert loss==0 and not active


@pytest.mark.parametrize('options',[{'pseudo_weight':-1},{'huber_delta':0},{'pseudo_uncertainty_scale':.5},{'joint_weight':float('nan')}])
def test_invalid_loss_options_rejected(options):
    with pytest.raises(ValueError):AuxiliaryMotionOptions(**options)


def test_manifest_hash_and_capture_split_are_pinned(tmp_path):
    manifest=fixture_pack(tmp_path)
    with pytest.raises(ValueError,match='manifest changed'):
        KinopticHumanAdapter(tmp_path,manifest_sha256='0'*64)
    duplicate=dict(manifest['samples'][0],case='another_frame',split='validation')
    manifest['samples'].append(duplicate)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='capture cannot cross'):
        KinopticHumanAdapter(tmp_path)


def test_duplicate_case_across_splits_is_rejected(tmp_path):
    manifest=fixture_pack(tmp_path)
    manifest['samples'].append(dict(manifest['samples'][0],split='validation'))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='duplicate clip'):
        KinopticHumanAdapter(tmp_path)
