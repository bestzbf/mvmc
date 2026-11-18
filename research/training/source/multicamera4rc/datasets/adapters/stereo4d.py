"""Published Stereo4D pseudo trajectories in the official 0.063 m stereo gauge."""
import json
from pathlib import Path
import cv2
import numpy as np
from .base import BaseAdapter, UnifiedClip, InvalidClipError

class Stereo4DAdapter(BaseAdapter):
    dataset_name = 'stereo4d'
    def __init__(self, root, split='train', inventory=None, max_tracks=512, source_tolerance_px=2., stereo=False):
        if split != 'train': raise ValueError('training adapter requires the admitted train inventory')
        self.stereo=bool(stereo)
        self.root=Path(root);self.max_tracks=int(max_tracks);self.tolerance=float(source_tolerance_px)
        if not 1<=self.max_tracks<=4096 or not 0<self.tolerance<=3:raise ValueError('invalid Stereo4D point budget/tolerance')
        rows=[json.loads(x) for x in Path(inventory).read_text().splitlines()]
        if any(r['split']!='train' for r in rows):raise ValueError('heldout clip in train inventory')
        if self.stereo and any(not r.get('right_video') for r in rows):raise ValueError('stereo mode requires a paired train inventory')
        self.rows={r['clip']:r for r in rows};self.names=sorted(self.rows)
    def __len__(self):return len(self.names)
    def list_sequences(self):return self.names
    def get_sequence_name(self,index):return self.names[index]
    def get_sequence_info(self,name):return dict(num_frames=self.rows[name]['frames'])
    def get_num_frames(self,name):return int(self.rows[name]['frames'])
    def sanity_check(self,sequence_name):
        clip=self.load_clip(sequence_name,[0,1,2,3])
        return dict(frames=len(clip.images),tracks=int(clip.valids.shape[1]),source_points=int(clip.visibs.sum()))
    def load_clip(self,sequence_name,frame_indices):
        row=self.rows[sequence_name];ids=np.asarray(frame_indices,dtype=np.int64)
        if len(np.unique(ids))!=len(ids) or len(ids)<2:raise InvalidClipError('need distinct Stereo4D times')
        with np.load(row['npz'],allow_pickle=False) as z:
            lengths=z['track_lengths'];indices=z['track_indices'];coordinates=z['track_coordinates']
            poses=z['camera2world'][ids];timestamps=z['timestamps'][ids]
        if np.any(np.diff(timestamps)<=0):raise ValueError('nonmonotone Stereo4D timestamps')
        intervals=np.diff(timestamps)/np.diff(ids)
        if np.max(np.abs(intervals/np.median(intervals)-1))>.01:
            raise InvalidClipError('irregular frame timing cannot supervise frame-based velocity')
        ends=np.cumsum(lengths,dtype=np.int64);starts=ends-lengths
        if ends[-1]!=len(indices) or len(indices)!=len(coordinates):raise ValueError('corrupt ragged tracks')
        # Bound memory: never materialize the full N x T trajectory tensor.
        seed=int.from_bytes(sequence_name.encode()[:8].ljust(8,b'0'),'little') % 2**32
        candidates=np.flatnonzero(lengths>=2);np.random.default_rng(seed).shuffle(candidates)
        chosen=[];values=[];masks=[]
        for track in candidates:
            ix=indices[starts[track]:ends[track]]
            if np.any(np.diff(ix)<=0):raise ValueError('nonmonotone track frame indices')
            positions=np.searchsorted(ix,ids);ok=positions<len(ix)
            ok[ok]&=ix[positions[ok]]==ids[ok]
            if ok.sum()<2:continue
            xyz=np.zeros((len(ids),3),np.float32);xyz[ok]=coordinates[starts[track]+positions[ok]]
            ok &= np.isfinite(xyz).all(-1)
            if ok.sum()<2:continue
            chosen.append(int(track));values.append(xyz);masks.append(ok)
            if len(chosen)>=self.max_tracks*8:break
        if not chosen:raise InvalidClipError('no recorded temporal trajectories')
        world=np.stack(values,axis=1);valid=np.stack(masks,axis=1)
        c2w=np.broadcast_to(np.eye(4),(len(ids),4,4)).copy();c2w[:,:3]=poses
        camera=np.einsum('tnj,tjk->tnk',world-c2w[:,None,:3,3],c2w[:,:3,:3])
        # Official visualization uses extrs_rectified directly. rectified2rig
        # has already been applied when generating the rectified RGB videos.
        focal=256/np.tan(np.pi/6);K=np.array([[focal,0,255.5],[0,focal,255.5],[0,0,1]],np.float32)
        uv=camera[...,:2]/np.maximum(camera[...,2:],1e-8)*focal+255.5
        in_view=valid & (camera[...,2]>0) & (camera[...,2]<=20) & (uv>=3).all(-1) & (uv<=508).all(-1)
        cap=cv2.VideoCapture(row['left_video']);images=[]
        try:
            if int(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=int(row['frames']):raise InvalidClipError('RGB/annotation frame count mismatch')
            for index in ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,image=cap.read()
                if not ok or image.shape[:2]!=(512,512):raise InvalidClipError('invalid rectified RGB frame')
                images.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        finally:cap.release()
        gray=[cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in images];certified=np.zeros_like(valid);errors=[]
        # Source correspondences are checked independently on RGB. This is a
        # conservative pseudo-label quality gate, not an occlusion annotation.
        for a in range(len(ids)-1):
            b=a+1;keep=np.flatnonzero(in_view[a]&in_view[b])
            if not len(keep):continue
            q=uv[a,keep].astype(np.float32)[:,None,:];guess=uv[b,keep].astype(np.float32)[:,None,:]
            nxt,status,_=cv2.calcOpticalFlowPyrLK(gray[a],gray[b],q,guess.copy(),winSize=(21,21),maxLevel=3,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            back,back_status,_=cv2.calcOpticalFlowPyrLK(gray[b],gray[a],nxt,q.copy(),winSize=(21,21),maxLevel=3,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            error=np.linalg.norm(nxt[:,0]-guess[:,0],axis=-1);fb=np.linalg.norm(back[:,0]-q[:,0],axis=-1)
            good=status[:,0].astype(bool)&back_status[:,0].astype(bool)&(error<=self.tolerance)&(fb<=1.)
            certified[a,keep[good]]=True;certified[b,keep[good]]=True;errors.extend(error[good].tolist())
        selected=np.flatnonzero(certified.any(0))[:self.max_tracks]
        if len(selected)<16:raise InvalidClipError('insufficient RGB-checked Stereo4D source correspondences')
        world=world[:,selected];valid=valid[:,selected];uv=uv[:,selected];certified=certified[:,selected]
        world[~valid]=0;uv[~np.isfinite(uv).all(-1)]=0
        metadata=dict(extrinsics_convention='c2w',pixel_coordinates='array_index_v1',frame_indices=ids.tolist(),
            supervision_kind='camera_tracks',track_source_valid=certified,source_label_kind='RGB-checked pseudo correspondences',
            timestamp_unit='microseconds',timestamps_us=timestamps.tolist(),track_ids=np.asarray(chosen)[selected].tolist(),
            has_temporal_supervision=True,source_reprojection_error_px=errors)
        clip = UnifiedClip(dataset_name=self.dataset_name,sequence_name=sequence_name,frame_paths=None,images=images,
            depths=None,normals=None,trajs_3d_world=world,trajs_2d=uv,valids=valid,visibs=certified,
            visibility_valids=certified,intrinsics=np.repeat(K[None],len(ids),axis=0),extrinsics=c2w.astype(np.float32),metadata=metadata)

        return self._add_right_eye(clip, row, ids) if self.stereo else clip

    def _add_right_eye(self, clip, row, ids):
        # Official track_optimization.py computes Z = f * 0.063 / disparity.
        # Rectified axes coincide; the right centre is +X in the left frame.
        right_pose = clip.extrinsics.copy()
        right_pose[:, :3, 3] += .063 * right_pose[:, :3, 0]
        xyz = np.einsum('tnj,tjk->tnk', clip.trajs_3d_world-right_pose[:, None, :3, 3], right_pose[:, :3, :3])
        uv = xyz[..., :2]/np.maximum(xyz[..., 2:], 1e-8)*clip.intrinsics[:, None, 0, 0, None] + 255.5
        cap=cv2.VideoCapture(row['right_video']);images=[]
        try:
            if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != int(row['frames']):
                raise InvalidClipError('right RGB/annotation frame count mismatch')
            for index in ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,image=cap.read()
                if not ok or image.shape[:2] != (512,512):raise InvalidClipError('invalid right rectified RGB')
                images.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        finally:cap.release()
        matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=5,
            P1=8*25,P2=32*25,disp12MaxDiff=1,uniquenessRatio=12,speckleWindowSize=50,speckleRange=2)
        certified=np.zeros_like(clip.valids);residuals=[]
        for t in range(len(ids)):
            left=cv2.cvtColor(clip.images[t],cv2.COLOR_RGB2GRAY);right=cv2.cvtColor(images[t],cv2.COLOR_RGB2GRAY)
            left_uv=clip.trajs_2d[t]
            keep=np.flatnonzero(clip.visibs[t] & (left_uv[:,0]>130) & (left_uv[:,0]<505)
                & (left_uv[:,1]>5) & (left_uv[:,1]<505) & (uv[t]>=3).all(-1) & (uv[t]<=508).all(-1))
            if not len(keep):continue
            disparity=matcher.compute(left,right).astype(np.float32)/16
            pix=np.rint(left_uv[keep]).astype(int);d=disparity[pix[:,1],pix[:,0]]
            keep=keep[d>1];d=d[d>1]
            if not len(keep):continue
            q=left_uv[keep].astype(np.float32)[:,None,:];guess=q.copy();guess[:,0,0]-=d
            nxt,status,_=cv2.calcOpticalFlowPyrLK(left,right,q,guess,winSize=(15,15),maxLevel=2,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            back,bs,_=cv2.calcOpticalFlowPyrLK(right,left,nxt,q.copy(),winSize=(15,15),maxLevel=2,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            error=np.linalg.norm(nxt[:,0]-uv[t,keep],axis=-1)
            good=status[:,0].astype(bool)&bs[:,0].astype(bool)&(np.linalg.norm(back[:,0]-q[:,0],axis=-1)<=1)
            good &= (abs(nxt[:,0,1]-q[:,0,1])<=1)&(abs(q[:,0,0]-nxt[:,0,0]-d)<=1.5)&(error<=1.5)
            certified[t,keep[good]]=True;residuals.extend(error[good].tolist())
        if certified.sum()<32 or certified.any(0).sum()<16:
            raise InvalidClipError('insufficient independently RGB-verified right-eye queries')
        count=len(ids);left_source=clip.visibs.copy()
        clip.images += images
        clip.extrinsics=np.concatenate([clip.extrinsics,right_pose])
        clip.intrinsics=np.concatenate([clip.intrinsics,clip.intrinsics])
        clip.trajs_3d_world=np.concatenate([clip.trajs_3d_world,clip.trajs_3d_world])
        clip.trajs_2d=np.concatenate([clip.trajs_2d,uv.astype(np.float32)])
        clip.valids=np.concatenate([clip.valids,clip.valids])
        clip.visibs=np.concatenate([left_source,certified]);clip.visibility_valids=clip.visibs.copy()
        clip.metadata.update(slot_camera=np.repeat([0,1],count),time_idx=ids.tolist()*2,
            frame_indices=ids.tolist()*2,track_source_valid=clip.visibs.copy(),stereo_baseline_m=.063,
            stereo_scale_kind='official_fixed_baseline_pseudo_metric',right_source_error_px=residuals,
            right_source_check='independent SGBM and forward-backward LK; 1.5px GT residual gate')
        return clip
