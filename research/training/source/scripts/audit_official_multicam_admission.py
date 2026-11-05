"""Final, independent packed-GT and real mixed-loader admission checks.

Every MV-Kubric scene is checked; dense loader cases cover each archive and
C/T layout, and real Panoptic clips cover every admitted capture. This is
additional to the converter's all-frame raw-file and native-track checks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from multicamera4rc.datasets.adapters.mv_kubric import validate_full_conversion
from multicamera4rc.datasets.mv_kubric_inventory import expected_train_ids
from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples, read_data_config
from multicamera4rc.datasets.dense import materialize_dense_targets
from scripts.continue_quality_recovery import digest
from scripts.cutover_native_motion_stereo import write_json
from scripts.prepare_mv_kubric_full import inspect_pack, validate_pack_inventory


def rotation_xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x,y,z,w = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3,3))


def world_to_reference(points,c2w):
    """The training clip's world frame is its first physical camera frame."""
    inverse=np.linalg.inv(np.asarray(c2w,dtype=np.float64))
    return np.asarray(points) @ inverse[:3,:3].T+inverse[:3,3]


def audit_pack(path):
    record = inspect_pack(path)
    path = Path(path)
    with h5py.File(path, 'r') as f:
        receipt = json.loads(path.with_suffix('.json').read_text())
        if (receipt.get('status') != 'complete' or receipt['scene_id'] != record['scene_id']
                or receipt['packed_bytes'] != path.stat().st_size
                or receipt['source_scene_consumed_files_sha256'] != f.attrs['source_scene_consumed_files_sha256']):
            raise ValueError(f'pack/receipt provenance mismatch: {path}')
        raw_audit = json.loads(f.attrs['native_track_audit'])
        if (raw_audit != receipt['native_track_audit'] or raw_audit['native_points'] != 18000
                or not np.isfinite(list(raw_audit.values())).all()
                or raw_audit['rigid_max_error_m'] > 2e-4
                or raw_audit['reprojection_max_error_pixels'] > .05):
            raise ValueError(f'conversion native oracle failed: {path}')
        native = f['audit_native']
        ids = native['tracks_raw_scene_ids'][()]
        omitted=f['unobserved_object_raw_scene_ids'][()] if 'unobserved_object_raw_scene_ids' in f else np.array([],np.int64)
        omitted_mask=np.isin(ids,omitted)
        if (len(set(omitted.tolist()))!=len(omitted)
                or np.isin(omitted,f['object_raw_scene_ids'][()]).any()
                or not native['occlusion'][:,:,np.flatnonzero(omitted_mask)].all()
                or int(omitted_mask.sum())!=raw_audit.get('unobserved_native_points',0)):
            raise ValueError(f'pose-less object is not independently proven invisible: {path}')
        # Sample every object identity, with deterministic spread over its points.
        selected = np.unique(np.concatenate([np.flatnonzero(ids == i)[::max(1, int((ids == i).sum())//16)]
                                             for i in np.unique(ids)]))
        xyz = native['tracks_3d'][:, selected].astype(np.float64)
        raw_ids = f['object_raw_scene_ids'][()]
        poses = f['object_trajectories'][()].astype(np.float64)
        rotations = rotation_xyzw(poses[..., 3:])
        rigid_max = 0.
        for raw in np.unique(ids[selected]):
            if raw in omitted:
                continue  # no image observations and no exported pose: never assume static
            mask = ids[selected] == raw
            truth = xyz[:, mask]
            hit = np.flatnonzero(raw_ids == raw)
            if len(hit):
                i = int(hit[0]); p,R = poses[i,:,:3], rotations[i]
                local = (truth[0]-p[0]) @ R[0]
                expected = np.einsum('tij,nj->tni', R, local)+p[:,None]
            else:
                expected = np.broadcast_to(truth[:1], truth.shape)
            rigid_max = max(rigid_max, float(np.linalg.norm(expected-truth, axis=-1).max()))
        if rigid_max > 2e-4:
            raise ValueError(f'packed poses/native tracks disagree: {path}: {rigid_max}')
        reprojection_max, visible = 0., 0
        for camera in (record['scene_id'] % 10, (record['scene_id']+5) % 10):
            E = f['extrinsics'][camera].astype(np.float64)
            K = f['intrinsics'][camera].astype(np.float64)
            inverse = np.linalg.inv(E)
            camera_xyz = xyz @ inverse[:3,:3].T + inverse[:3,3]
            uv = native['tracks_2d'][:, camera, selected]
            occluded = native['occlusion'][:, camera, selected].astype(bool)
            mask = (camera_xyz[...,2] > 1e-4) & ~occluded & np.isfinite(uv).all(-1)
            if mask.any():
                projected = camera_xyz @ K.T
                projected = projected[mask,:2]/projected[mask,2,None]
                reprojection_max = max(reprojection_max, float(np.linalg.norm(projected-uv[mask],axis=-1).max()))
                visible += int(mask.sum())
            for t in (0, 23):
                depth, seg = f['depth'][t,camera], f['segmentation'][t,camera]
                if (not np.isfinite(depth).all() or (depth < 0).any()
                        or not (depth > 0).any() or int(seg.max()) > len(raw_ids)):
                    raise ValueError(f'invalid rendered depth/instance IDs: {path}, {camera}, {t}')
                if f['frames'][t,camera].shape != (512,512,3):
                    raise ValueError(f'RGB decode failed: {path}')
        if visible == 0 or reprojection_max > .05:
            raise ValueError(f'packed native projection failed: {path}: {reprojection_max}, visible={visible}')
    return dict(record, rigid_max_m=rigid_max, reprojection_max_px=reprojection_max,
                visible_track_samples=visible, raw_native_audit=raw_audit)


def loader_cases(dataset):
    """Freeze actual sampler indices, without overriding or mocking its plan."""
    mv, pan = {}, {}
    for si, source in enumerate(dataset.sources):
        if source['name'] not in ('mv_kubric','panoptic'):
            continue
        start = dataset.quota_ends[si-1] if si else 0
        for index in range(start, dataset.quota_ends[si]):
            plan = dataset.sample_plan(index)
            layout = (plan['cameras'], plan['num_times'])
            if source['name'] == 'mv_kubric':
                scene = int(plan['sequence'].rsplit('_',1)[-1])
                key = ((scene-1)//1000, *layout)
                mv.setdefault(key,(0,index))
            else:
                pan.setdefault(plan['sequence'],(0,index))
        if source['name'] == 'mv_kubric':
            expected = {(a,l['cameras'],l['num_times']) for a in range(5) for l in source['layouts']}
            if set(mv) != expected:
                raise ValueError('mixed sampler did not cover all MV archive/layout combinations')
        else:
            # Short captures need not appear in the first weighted epoch. Find
            # their exact position in the real coverage permutation rather
            # than dropping them or substituting an invented sampling plan.
            a,b=source['coverage_permutation'];n=source['coverage_size']
            for entry,(sequence,_,_) in enumerate(source['coverage_entries']):
                if sequence in pan:continue
                window=source['coverage_ends'][entry-1] if entry else 0
                ordinal=(pow(a,-1,n)*(window-b))%n
                epoch,local=divmod(ordinal,dataset.quotas[si])
                dataset.set_epoch(epoch);index=start+local
                if dataset.sample_plan(index)['sequence']!=sequence:
                    raise ValueError('coverage permutation inversion failed')
                pan[sequence]=(epoch,index)
            dataset.set_epoch(0)
            if len(pan)!=len(source['pool']):
                raise ValueError('mixed sampler did not cover every admitted Panoptic capture')
    if not mv or not pan:
        raise ValueError('both official sources are required')
    return list(mv.values())+list(pan.values())


def check_sample(dataset, index, epoch=0):
    dataset.set_epoch(epoch)
    plan = dataset.sample_plan(index)
    sample = dataset[index]
    batch = collate_adapter_samples([sample])
    source = torch.tensor([[0]])
    target = materialize_dense_targets(batch, source, normalize_scene=False)
    c,t = plan['cameras'], plan['num_times']
    if (batch['images'].shape[1] != c*t or batch['camera_id'].unique().numel() != c
            or batch['query_time_idx'].shape[1] != t or not torch.isfinite(batch['images']).all()
            or not target['camera_valid'].all()):
        raise ValueError('actual mixed-loader observations disagree with sampler plan')
    for camera in batch['camera_id'].unique():
        values = target['K'][0,batch['camera_id'][0] == camera]
        if not torch.allclose(values, values[:1],atol=1e-4,rtol=1e-6):
            raise ValueError('same camera has changing calibration across synchronized frames')
    name = plan['source']['name']
    row = dict(epoch=epoch,index=index, source=name, sequence=plan['sequence'], cameras=c, times=t,
               input_shape=list(batch['images'].shape))
    if name == 'panoptic':
        if any(target[k].any() for k in ('depth_valid','trajectory_valid','displacement_valid',
                                        'static_mask','camera_translation_valid')):
            raise ValueError('Panoptic supplied invented surface or metric-translation GT')
    else:
        if (not target['source_pixel_valid'].any() or not torch.isfinite(target['trajectory']).all()
                or target['depth_valid'][...,512:,:].any() or target['depth_valid'][...,:,512:].any()):
            raise ValueError('dense source labels or padding mask are invalid')
        identity = batch['query_time_idx'][0] == batch['time_idx'][0,0]
        if target['displacement'][0,0,identity].count_nonzero():
            raise ValueError('identity-time displacement is nonzero')
        path = Path(plan['source']['adapter']._name_to_record[plan['sequence']].h5_path)
        with h5py.File(path,'r') as f:
            camera = int(batch['camera_id'][0,0]); start = int(batch['time_idx'][0,0])
            # The maintained adapter exposes the original packed camera-axis
            # index as its physical integer ID; camera_ids stores names.
            ci = camera
            if not 0 <= ci < len(f['camera_ids']):
                raise ValueError('physical camera index is outside the original pack')
            radial,seg = f['depth'][start,ci],f['segmentation'][start,ci]
            K,E = f['intrinsics'][ci].astype(np.float64),f['extrinsics'][ci].astype(np.float64)
            y,x = np.indices(radial.shape)
            rays = np.stack((x+.5,y+.5,np.ones_like(x)),-1) @ np.linalg.inv(K).T
            points = rays/np.linalg.norm(rays,axis=-1,keepdims=True)*radial[...,None]
            world = points @ E[:3,:3].T+E[:3,3]
            valid = np.isfinite(radial)&(radial>0)&((seg>0)|(np.abs(world[...,2])<=.05))
            actual = target['source_pixel_valid'][0,0,:512,:512].numpy()
            if not np.array_equal(actual,valid):
                raise ValueError('dense ground/dome mask differs from independent raster unprojection')
            pixels = np.flatnonzero(valid)[::max(1,int(valid.sum())//256)]
            ys,xs = np.unravel_index(pixels,valid.shape)
            baseline = world[ys,xs]
            poses = f['object_trajectories'][()].astype(np.float64)
            R = rotation_xyzw(poses[...,3:]); labels=seg[ys,xs]
            error = 0.
            for slot,tt in enumerate(batch['query_time_idx'][0].tolist()):
                truth = baseline.copy()
                for oid in np.unique(labels):
                    if oid == 0:
                        continue
                    mask=labels==oid; i=int(oid)-1
                    local=(baseline[mask]-poses[i,start,:3]) @ R[i,start]
                    truth[mask]=local @ R[i,tt].T+poses[i,tt,:3]
                predicted_gt=target['trajectory'][0,0,slot,ys,xs].numpy()
                truth=world_to_reference(truth,E)
                error=max(error,float(np.linalg.norm(predicted_gt-truth,axis=-1).max()))
            if error > 2e-4:
                raise ValueError(f'dense loader GT disagrees with independent rigid motion: {error}')
            row.update(dense_trajectory_max_error_m=error,valid_source_pixels=int(valid.sum()))
    return row


def run(config_path, output, *, workers=2):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    config=read_data_config(config_path)
    source=next(s for s in config['sources'] if s['adapter']=='mv_kubric')
    root=Path(source['root']); paths=sorted(root.rglob('*.h5'))
    validate_full_conversion(root,[int(p.stem.rsplit('_',1)[-1]) for p in paths])
    write_json(output/'status.json',dict(stage='checking_all_packed_gt',scenes=len(paths)))
    records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool, (output/'packed_gt.jsonl').open('w') as log:
        for row in pool.map(audit_pack,paths,chunksize=8):
            records.append(row); log.write(json.dumps(row)+'\n'); log.flush()
            if len(records)%100==0:
                write_json(output/'status.json',dict(stage='checking_all_packed_gt',checked=len(records),scenes=len(paths)))
    manifest=json.loads((root/'conversion_manifest.json').read_text())
    validate_pack_inventory(records, expected_train_ids(manifest))
    torch.set_num_threads(2)
    # An admission check must surface a bad sample rather than exercising the
    # training loader's fallback to another sample of the same source.
    dataset=create_dataset(dict(config,skip_invalid=False)); indices=loader_cases(dataset)
    write_json(output/'status.json',dict(stage='checking_mixed_loader',cases=len(indices)))
    cases=[]
    with (output/'loader_gt.jsonl').open('w') as log:
        for epoch,index in indices:
            row=check_sample(dataset,index,epoch);cases.append(row);log.write(json.dumps(row)+'\n');log.flush();gc.collect()
    result=dict(passed=True,config=str(Path(config_path).resolve()),config_sha256=digest(config_path),
        conversion_manifest=str(root/'conversion_manifest.json'),conversion_manifest_sha256=digest(root/'conversion_manifest.json'),
        checked_scenes=len(records),loader_cases=len(cases),
        mv_loader_cases=sum(c['source']=='mv_kubric' for c in cases),
        panoptic_captures=len({c['sequence'] for c in cases if c['source']=='panoptic'}),
        rigid_max_m=max(r['rigid_max_m'] for r in records),reprojection_max_px=max(r['reprojection_max_px'] for r in records),
        source_inventory=[dict(name=s['display_name'],sequences=len(s['pool']),quota=dataset.quotas[i]) for i,s in enumerate(dataset.sources)],
        packed_gt_sha256=digest(output/'packed_gt.jsonl'),loader_gt_sha256=digest(output/'loader_gt.jsonl'))
    write_json(output/'admission.json',result);write_json(output/'status.json',dict(stage='passed',**result))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();print(json.dumps(run(args.config,args.output)))
