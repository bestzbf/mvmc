"""Immutable verified depth, official joints, and separate surface-pseudo packs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np

from .base import BaseAdapter, UnifiedClip
from .kinoptic_geometry import _intrinsic, _rigid


class KinopticHumanAdapter(BaseAdapter):
    dataset_name = 'kinoptic_human'

    def __init__(self, root, split='train', manifest='manifest.json', allow_candidate=False,
                 manifest_sha256=None):
        self.root = Path(root).resolve()
        path = Path(manifest)
        if not path.is_absolute(): path = self.root/path
        self.manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest_sha256 is not None and manifest_sha256 != self.manifest_sha256:
            raise ValueError('Kinoptic manifest changed after admission')
        value = json.loads(path.read_text())
        if value.get('kind') != 'kinoptic_human_pack_manifest' or value.get('schema_version') != 1:
            raise ValueError('unsupported Kinoptic human pack manifest')
        if not value.get('training_admitted', False) and not allow_candidate:
            raise ValueError('candidate Kinoptic packs are not admitted to production training')
        if split not in ('train', 'validation'):
            raise ValueError('Kinoptic pack split must be train or validation')
        allowed = set(value['inherited_scene_whitelist'])
        self.records = {}
        all_cases = set(); capture_splits = {}
        for row in value['samples']:
            if row['scene'] not in allowed:
                raise ValueError('capture outside inherited training-only whitelist')
            if row['case'] in all_cases or row['split'] not in ('train', 'validation'):
                raise ValueError('duplicate clip or invalid split in manifest')
            all_cases.add(row['case'])
            if capture_splits.setdefault(row['scene'], row['split']) != row['split']:
                raise ValueError('one capture cannot cross auxiliary label splits')
            if row['split'] != split: continue
            if row['case'] in self.records or row.get('pseudo_is_true_gt') is not False:
                raise ValueError('duplicate clip or invalid pseudo-label provenance')
            if row.get('status') != 'prepared':
                raise ValueError('unprepared clip in admission manifest')
            p = Path(row['path']).resolve()
            if not p.is_relative_to(self.root):
                raise ValueError('Kinoptic pack must be inside its immutable root')
            self.records[row['case']] = dict(row, path=str(p))
        if not self.records:
            raise ValueError('no Kinoptic human clips for requested split')
        self.sequences = sorted(self.records)

    def __len__(self): return len(self.sequences)
    def list_sequences(self): return self.sequences.copy()
    def get_sequence_name(self, index): return self.sequences[index]
    def get_num_frames(self, sequence_name): return 2

    def sanity_check(self, sequence_name):
        clip = self.load_clip(sequence_name, [0, 1, 0, 1])
        evidence = clip.metadata['kinoptic_human_evidence']
        return dict(frames=len(clip.images),
                    measured_depth_pixels=sum(int((d > 0).sum()) for d in clip.depths),
                    official_joints=len(evidence['joint_start_world_m']),
                    pseudo_points=len(evidence['pseudo_source_uv']),
                    pseudo_is_true_gt=False)

    def get_sequence_info(self, sequence_name):
        row = self.records[sequence_name]
        return dict(num_frames=2, num_cameras=2, cameras=row['camera_names'],
                    image_size=tuple(row['image_hw']), has_depth=True, has_tracks=False,
                    has_temporal_supervision=True, physical_scene=row['scene'])

    def load_clip(self, sequence_name, frame_indices, *, layout_num_cameras=2,
                  layout_num_times=2, sampling_seed=0, canonical_anchor_required=False):
        if canonical_anchor_required:
            raise ValueError('Kinoptic joint/pseudo labels are not a canonical material-point bank')
        if layout_num_cameras != 2 or layout_num_times != 2 or list(frame_indices) != [0, 1, 0, 1]:
            raise ValueError('Kinoptic human packs require their exact two cameras and two times')
        row = self.records[sequence_name];path = Path(row['path']);before = path.stat()
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Kinoptic pack changed after admission')
        with np.load(path, allow_pickle=False) as archive:
            a = {key: archive[key] for key in archive.files}
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Kinoptic pack changed while reading')
        images, depth, K, E = a['images'], a['depths'], a['K'], a['E']
        if images.dtype != np.uint8 or images.ndim != 5 or images.shape[:2] != (2, 2) or images.shape[-1] != 3:
            raise ValueError('invalid Kinoptic RGB shape/type')
        if list(images.shape[2:4]) != row['image_hw']:
            raise ValueError('Kinoptic manifest and RGB dimensions disagree')
        if depth.shape != images.shape[:-1] or a['pixel_support'].shape != depth.shape:
            raise ValueError('Kinoptic RGB/depth/support grids disagree')
        if not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError('invalid measured depth; zero holes must not be inpainted')
        if K.shape != (2, 2, 3, 3) or E.shape != (2, 2, 4, 4):
            raise ValueError('invalid Kinoptic calibration shapes')
        for k, e in zip(K.reshape(-1, 3, 3), E.reshape(-1, 4, 4)):
            _intrinsic(k); _rigid(e, 'E_w2c')
        if not np.array_equal(a['frame_indices'], row['frames']) or not np.allclose(a['times_ms'], row['times_ms'], atol=1e-6, rtol=0):
            raise ValueError('pack/manifest timestamps disagree')
        names = row['camera_names'];frames = list(map(int, row['frames']))
        camera_ids = [int(name.split('_')[1]) for name in names for _ in frames]
        metadata = dict(extrinsics_convention='w2c', pixel_coordinates='array_index_v1',
            slot_camera=[0, 0, 1, 1], slot_physical_camera_id=camera_ids,
            time_idx=frames*2, frame_indices=frames*2, valid_slot=[True]*4,
            camera_valid=[True]*4, has_depth=True, has_tracks=False, has_visibility=False,
            has_temporal_supervision=True, camera_translation_supervision=True,
            supervision_kind='kinoptic_human', world_units='metres',
            pixel_support=list(a['pixel_support'].reshape(4, *depth.shape[-2:])),
            kinoptic_human_evidence={key: a[key] for key in a if key.startswith(('joint_', 'pseudo_'))},
            kinoptic_manifest_sha256=self.manifest_sha256, kinoptic_pack_sha256=row['sha256'],
            kinoptic_capture=row['scene'], pseudo_is_true_gt=False,
            material_point_correspondence_gt=False, fps=30000/1001)
        return UnifiedClip(dataset_name=self.dataset_name, sequence_name=sequence_name,
            frame_paths=[f'{path}#camera={name}&frame={frame}' for name in names for frame in frames],
            images=list(images.reshape(4, *images.shape[2:])), depths=list(depth.reshape(4, *depth.shape[2:])),
            normals=None, trajs_2d=None, trajs_3d_world=None, valids=None, visibs=None,
            intrinsics=K.reshape(4, 3, 3), extrinsics=E.reshape(4, 4, 4), metadata=metadata)
