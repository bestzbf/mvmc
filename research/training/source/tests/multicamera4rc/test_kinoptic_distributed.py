"""Exercise the production reducer and real two-rank train/save/resume path."""
import json,math,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import distributed as dist
from multicamera4rc.auxiliary_motion import AUXILIARY_MOTION_METRICS
from multicamera4rc.distributed import aggregate_metrics
from multicamera4rc.losses import MultiCamera4RCLoss
from test_kinoptic_human_supervision import fixture_pack
from test_geometry_generalization_ddp import _WORKER
from test_geometry_generalization_integration import command


def test_reducer_registers_all_aux_metrics_and_excludes_unknown_label_diagnostics():
    context=SimpleNamespace(device=torch.device('cpu'),distributed=False,world_size=1)
    metrics={k:torch.tensor(0.) for k in AUXILIARY_MOTION_METRICS}
    metrics.update(kinoptic_joint_motion_points=torch.tensor(4.),kinoptic_joint_motion_epe_m=torch.tensor(.2),
                   kinoptic_pseudo_motion_epe_m=torch.tensor(float('nan')))
    values=aggregate_metrics(context,metrics)
    assert values['kinoptic_joint_motion_epe_m']==pytest.approx(.2)
    assert values['kinoptic_joint_motion_points']==4
    assert 'kinoptic_pseudo_motion_epe_m' not in values
    assert values['kinoptic_pseudo_motion_loss']==0


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),reason='Gloo unavailable')
def test_two_rank_kinoptic_auxiliary_training_save_and_resume(tmp_path):
    fixture_pack(tmp_path)
    config=tmp_path/'data.json';config.write_text(json.dumps(dict(
        sources=[dict(name='kinoptic_human_motion',adapter='kinoptic_human',root=str(tmp_path),split='train',
                      cameras=2,num_times=2,strides=[1],dense_kind='kinoptic_human')],
        image_mode='native',supervision='dense',num_times=2,num_queries=0,sampling='balanced_coverage',epoch_size=8)))
    loss=MultiCamera4RCLoss.dense().to_config();loss['auxiliary_motion_options']=dict(joint_weight=.1,pseudo_weight=.02)
    loss_path=tmp_path/'loss.json';loss_path.write_text(json.dumps(loss))
    worker=tmp_path/'worker.py';worker.write_text(_WORKER)
    repo=Path(__file__).resolve().parents[2]
    env={k:v for k,v in os.environ.items() if k not in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT')}
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONPATH=str(repo)+os.pathsep+env.get('PYTHONPATH',''))
    first=tmp_path/'first';second=tmp_path/'second'
    for output,steps,resume in ((first,2,None),(second,3,first/'latest.pt')):
        args=command(config,output,steps);args[args.index('--source-count')+1]='4'
        args+=['--amp','--amp-dtype','bfloat16','--save-every','100']
        if resume is None:args+=['--loss-config',str(loss_path),'--geometry-lr-multiplier','2','--geometry-lr-warmup-steps','4']
        else:args+=['--resume',str(resume)]
        argv=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--rdzv-backend=c10d',
              '--rdzv-endpoint=127.0.0.1:0','--local-addr=127.0.0.1','--nproc_per_node=2',str(worker),*args]
        result=subprocess.run(argv,cwd=repo,env=env,capture_output=True,text=True,timeout=90)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
        updates=[]
        for line in result.stdout.splitlines():
            try:r=json.loads(line)
            except ValueError:continue
            # Gloo startup logs can interleave and leave numeric-only lines.
            if isinstance(r, dict) and 'optimizer_step_skipped' in r:
                updates.append(r)
        assert updates and all(not r['optimizer_step_skipped'] for r in updates)
        for row in updates:
            assert row['kinoptic_joint_motion_points']>0 and row['kinoptic_pseudo_motion_points']>0
            assert all(math.isfinite(row[k]) for k in AUXILIARY_MOTION_METRICS)
        ranks=[json.loads((output/f'rank_{i}.json').read_text()) for i in range(2)]
        assert ranks[0]['digest']==ranks[1]['digest']
        assert all(x['finite'] and not x['skipped'] for rank in ranks for x in rank['gradient_steps'])
    saved=torch.load(second/'latest.pt',weights_only=True)
    assert saved['step']==3 and saved['training_state']['loss_config']['auxiliary_motion_options']['pseudo_weight']==.02
