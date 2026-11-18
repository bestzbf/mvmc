from dataclasses import replace
from unittest.mock import patch
import json
import pytest
import torch
from multicamera4rc import MultiCamera4RC, MultiCamera4RCConfig
from multicamera4rc import model as model_module
from multicamera4rc import train
from multicamera4rc.datasets.demo import make_demo_data

@pytest.mark.parametrize('mode', ['none','no_camera_relation','no_relative_time','no_refiner'])
def test_strict_load_freezes_disabled_paths_and_keeps_time_grid(mode):
    torch.set_num_threads(1)
    cfg=MultiCamera4RCConfig.tiny(refinement_dim=16,refinement_iterations=1,refinement_levels=1,gradient_checkpointing=False)
    parent=MultiCamera4RC(cfg)
    with torch.no_grad():
        for n,p in parent.named_parameters():
            if n.endswith('camera_relation_weight') or n=='motion_decoder.camera_bias' or n.startswith('delta_projection.'):
                p.fill_(.2)
    m=MultiCamera4RC(replace(cfg,ablation=mode));m.load_state_dict(parent.state_dict(),strict=True)
    disabled=[]
    for n,p in m.named_parameters():
        zero=(mode=='no_camera_relation' and (n.endswith('camera_relation_weight') or n=='motion_decoder.camera_bias')) or (mode=='no_relative_time' and n.startswith('delta_projection.'))
        if zero:
            assert torch.count_nonzero(p)==0 and not p.requires_grad;disabled.append(n)
        if mode=='no_refiner' and n.startswith('trajectory_refiner.'):assert not p.requires_grad
    if mode in ('no_camera_relation','no_relative_time'):assert disabled
    images=torch.rand(1,4,3,28,28);cameras=torch.tensor([[0,0,1,1]]);times=torch.tensor([[0,2,0,2]])
    with patch.object(model_module,'build_metadata',wraps=model_module.build_metadata) as meta, patch.object(m.trajectory_refiner,'forward',wraps=m.trajectory_refiner.forward) as ref:
        out=m(images,cameras,times,torch.tensor([[2]]),torch.tensor([[0]]),return_visibility=True)
        assert meta.call_args.kwargs['all_observed_times'] is True
        assert ref.called == (mode!='no_refiner')
    loss=out['trajectory'].square().mean()+out['visibility'].mean();assert torch.isfinite(loss);loss.backward()
    for n,p in m.named_parameters():
        if n in disabled:assert p.grad is None
    # Native inference roundtrip must retain the mode, not reactivate the branch.
    payload={'architecture':'multicamera4rc','config':m.config.to_dict(),'model':m.state_dict()}
    restored=MultiCamera4RC.from_checkpoint(payload);assert restored.config.ablation==mode
    assert restored.refinement_active == (mode!='no_refiner')

@pytest.mark.parametrize('mode',['no_camera_relation','no_relative_time','no_refiner'])
def test_native_weights_only_starts_fresh_single_rank(tmp_path,mode):
    torch.set_num_threads(1)
    cfg=MultiCamera4RCConfig.tiny(refinement_dim=16,refinement_iterations=1,refinement_levels=1,gradient_checkpointing=False)
    m=MultiCamera4RC(cfg);parent=tmp_path/'parent.pt'
    torch.save(dict(architecture='multicamera4rc',config=cfg.to_dict(),model=m.state_dict(),step=215000,training_state={'world_size':7}),parent)
    data=make_demo_data(tmp_path/'data')
    result=train.main(['--data-config',str(data),'--initialize-native',str(parent),'--ablation',mode,'--output',str(tmp_path/'run'),'--steps','1','--device','cpu','--cpu-threads','1','--workers','0','--source-count','1','--max-source-time-pairs','4','--lr','0.0001','--backbone-lr','0.000001','--refinement-lr','0.0001','--geometry-lr-multiplier','0.3','--geometry-lr-warmup-steps','0','--no-checkpointing','--save-every','1'])
    x=torch.load(result,map_location='cpu',weights_only=True)
    assert x['step']==1 and x['training_state']['world_size']==1
    assert x['config']['ablation']==mode
    assert x['training_state']['weights_only_initialization']['parent_step']==215000
    assert x['training_state']['weights_only_initialization']['optimizer_reset']
