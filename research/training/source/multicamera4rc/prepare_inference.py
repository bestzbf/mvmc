"""Prepare native RGB image/video clips with explicit camera and shared time IDs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .metadata import build_metadata


def _integer(value, name, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must be an integer, not a name or timestamp')
    if nonnegative and value < 0:
        raise ValueError(f'{name} must be nonnegative')
    return value


def _path(base, value):
    return (base / str(value)).resolve()


def _video_frames(path, frames):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('video input requires opencv-python-headless; install the data extra') from exc
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'cannot open video: {path}')
    result = {}
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        wanted = set(frames)
        for index in range(max(wanted) + 1):
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f'cannot decode source frame {index}: {path}')
            if index in wanted:
                result[index] = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                 dict(video=str(path), frame_index=index, fps=fps,
                                      decoder_pts_ms=float(cap.get(cv2.CAP_PROP_POS_MSEC))))
    finally:
        cap.release()
    return result


def prepare(config, output):
    config = Path(config).resolve(); output = Path(output).resolve()
    if (output / 'clips.json').exists() or (output / 'images').exists():
        raise FileExistsError(f'prepared output already exists; choose a new directory: {output}')
    plan = json.loads(config.read_text())
    clips = plan.get('clips')
    if not isinstance(clips, list) or not clips:
        raise ValueError('config requires a nonempty clips list')
    prepared = []
    for clip_index, clip in enumerate(clips):
        observations = []; pixels = []; camera_ids = set()
        cameras = clip.get('cameras')
        if not isinstance(cameras, list) or not cameras:
            raise ValueError('each clip requires a nonempty cameras list')
        for camera in cameras:
            cid = _integer(camera['camera_id'], 'camera_id', nonnegative=True)
            if cid in camera_ids:
                raise ValueError('each camera_id must appear once in the cameras list')
            camera_ids.add(cid)
            name = str(camera.get('name', cid))
            if ('images' in camera) == ('video' in camera):
                raise ValueError('each camera needs either images or video, exclusively')
            if 'images' in camera:
                rows = camera['images']
                if not isinstance(rows, list) or not rows:
                    raise ValueError('camera images must be a nonempty list')
                for row in rows:
                    t = _integer(row['time_idx'], 'time_idx')
                    path = _path(config.parent, row['image'])
                    with Image.open(path) as im:
                        rgb = np.array(im.convert('RGB'), copy=True)
                    observations.append(dict(camera_id=cid, camera_name=name, time_idx=t,
                                             source_image=str(path), valid=True))
                    pixels.append(rgb)
            else:
                frames = camera.get('frame_indices')
                times = camera.get('time_idx')
                if not isinstance(frames, list) or not frames or not isinstance(times, list) or len(times) != len(frames):
                    raise ValueError('video requires matching nonempty frame_indices and shared time_idx lists')
                frames = [_integer(f, 'frame_index', nonnegative=True) for f in frames]
                times = [_integer(t, 'time_idx') for t in times]
                if len(set(frames)) != len(frames):
                    raise ValueError('source video frame_indices must be unique')
                path = _path(config.parent, camera['video'])
                extracted = _video_frames(path, frames)
                for frame, t in zip(frames, times):
                    rgb, provenance = extracted[frame]
                    observations.append(dict(camera_id=cid, camera_name=name, time_idx=t, valid=True,
                                             **provenance))
                    pixels.append(rgb)
        if not observations:
            raise ValueError('clip has no RGB observations')
        camera = [o['camera_id'] for o in observations]; times = [o['time_idx'] for o in observations]
        query = clip.get('query_time_idx', sorted(set(times)))
        if not isinstance(query, list) or not query:
            raise ValueError('query_time_idx must be a nonempty list')
        query = [_integer(t, 'query_time_idx') for t in query]
        sources = clip.get('source_obs_idx')
        if sources is None:
            # One source at each camera's earliest supplied time; retain config order.
            sources = [min((s for s, c in enumerate(camera) if c == cid), key=lambda s: times[s])
                       for cid in dict.fromkeys(camera)]
        if not isinstance(sources, list) or not sources:
            raise ValueError('source_obs_idx must be a nonempty list')
        sources = [_integer(s, 'source_obs_idx', nonnegative=True) for s in sources]
        build_metadata(torch.tensor([camera]), torch.tensor([times]), query, sources)
        h = max(rgb.shape[0] for rgb in pixels); w = max(rgb.shape[1] for rgb in pixels)
        h = max(28, (h + 13)//14*14); w = max(28, (w + 13)//14*14)
        directory = output / 'images' / f'clip_{clip_index:06d}'; directory.mkdir(parents=True)
        for s, (rgb, obs) in enumerate(zip(pixels, observations)):
            native_h, native_w = rgb.shape[:2]
            padded = np.zeros((h, w, 3), dtype=np.uint8)
            padded[:native_h, :native_w] = rgb
            target = directory / f'observation_{s:06d}.png'
            Image.fromarray(padded).save(target)
            obs.update(image=str(target.relative_to(output)), native_hw=[native_h, native_w],
                       rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        prepared.append(dict(id=str(clip.get('id', clip_index)), observations=observations,
                             query_time_idx=query, source_obs_idx=sources, prepared_hw=[h, w],
                             preprocessing='native decoded RGB; zero padding on bottom/right only'))
    manifest = output / 'clips.json'
    manifest.write_text(json.dumps(dict(clips=prepared), indent=2, ensure_ascii=False) + '\n')
    (output / 'preparation.json').write_text(json.dumps(dict(config=str(config), manifest=str(manifest),
        pixel_resize=False, distortion_correction=False, synchronization_estimated=False,
        clips=[dict(id=c['id'],observations=len(c['observations']),camera_ids=list(dict.fromkeys(o['camera_id'] for o in c['observations'])),
                    times=c['query_time_idx'],source_obs_idx=c['source_obs_idx'],prepared_hw=c['prepared_hw']) for c in prepared]),
        indent=2, ensure_ascii=False) + '\n')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='JSON image/video plan with explicit camera_id and shared time_idx')
    parser.add_argument('--output', required=True, help='New directory for padded PNGs and clips.json')
    args = parser.parse_args(argv)
    path = prepare(args.config, args.output)
    print(json.dumps(dict(manifest=str(path)), ensure_ascii=False))
    return path


if __name__ == '__main__': main()
