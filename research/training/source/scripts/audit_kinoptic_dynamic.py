"""CPU diagnostics of human-region RGB/depth alignment, never new motion GT.

Body joints select times and approximate image ROIs only. Edge distances,
temporal offsets and unwarped patch correlation are diagnostic proxies, not
surface depth, segmentation, visibility or material-point ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import tarfile
import time
import traceback

import cv2
import numpy as np


class UnsupportedReplay(ValueError):
    """Frozen views/times lack support after filtering; never replace them."""

    def __init__(self,reason,*,rejected_cameras=None,reader_error_type=None):
        super().__init__(reason)
        self.rejected_cameras=[] if rejected_cameras is None else rejected_cameras
        self.reader_error_type=reader_error_type

    @classmethod
    def from_reader(cls,error):
        return cls(str(error),rejected_cameras=getattr(error,"rejected_cameras",None),
                   reader_error_type=type(error).__name__)


def gt_source_files(runtime):
    """Bind transitive dataset code and the admission/mixture/audit entrypoints."""
    runtime=Path(runtime)
    names={path.relative_to(runtime).as_posix()
           for path in (runtime/"multicamera4rc/datasets").rglob("*.py")}
    names.update("scripts/"+name for name in ("audit_kinoptic_dynamic.py",
        "audit_kinoptic_hd_sync.py","build_kinoptic_mixture.py"))
    missing=sorted(name for name in names if not (runtime/name).is_file())
    if missing:
        raise ValueError(f"GT source snapshot is incomplete: {missing}")
    return sorted(names)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False))
    os.replace(temporary, path)


def body_joints(body, confidence=.2):
    """Read official centimetre joints into metre coordinates, for ROI only."""
    key = "joints19" if "joints19" in body else "joints15"
    expected = 19 if key == "joints19" else 15
    value = np.asarray(body.get(key, []), dtype=np.float64)
    if value.size != expected * 4:
        return None
    value = value.reshape(expected, 4).copy()
    value[:, :3] *= .01
    valid = np.isfinite(value).all(axis=-1) & (value[:, 3] >= confidence)
    if valid.sum() < 6:
        return None
    return dict(id=body.get("id"), joints=value, valid=valid, skeleton=key)


def motion_proxy(previous, current):
    """Confidence-gated joint velocity ranks motion; it is never scene-flow GT."""
    delta_seconds = (current["time_ms"] - previous["time_ms"]) / 1000
    if not 0 < delta_seconds <= 1.:
        return None
    speeds = []
    old = {body["id"]: body for body in previous["bodies"] if body["id"] is not None}
    for body in current["bodies"]:
        before = old.get(body["id"])
        if before is None or body["skeleton"] != before["skeleton"]:
            continue
        good = body["valid"] & before["valid"]
        if good.sum() < 6:
            continue
        value = np.linalg.norm(body["joints"][good, :3] - before["joints"][good, :3], axis=-1)
        speeds.extend((value / delta_seconds).tolist())
    if not speeds:
        return None
    score = float(np.percentile(speeds, 75))
    return score if score <= 8. else None  # Ignore implausible track jumps as motion candidates.


def collect_body_candidates(tar_path, frames, times, *, stride=10, confidence=.2):
    mapping = {int(f): float(t) for f, t in zip(frames, times)}
    candidates = []
    report = dict(body_tar=str(tar_path), body_stride=stride, sampled_json=0,
                  timing_mismatches=0, empty_or_low_confidence=0, candidates=0)
    before = Path(tar_path).stat()
    with tarfile.open(tar_path, "r:") as archive:
        for member in archive:
            match = re.search(r"(?:^|/)body3DScene_(\d+)\.json$", member.name)
            if not member.isfile() or not match:
                continue
            frame = int(match.group(1))
            if frame not in mapping or frame % stride:
                continue
            report["sampled_json"] += 1
            if not 0 < member.size <= 4 * 1024 * 1024:
                raise ValueError("body JSON outside bounded metadata size")
            with archive.extractfile(member) as stream:
                raw = stream.read()
            data = json.loads(raw)
            value = data.get("univTime")
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0 or abs(value - mapping[frame]) > 1.):
                report["timing_mismatches"] += 1
                continue
            bodies = [value for body in data.get("bodies", [])
                      if (value := body_joints(body, confidence)) is not None]
            if not bodies:
                report["empty_or_low_confidence"] += 1
                continue
            candidates.append(dict(frame=frame, time_ms=float(data["univTime"]), bodies=bodies,
                body_member=member.name, body_sha256=hashlib.sha256(raw).hexdigest()))
    after = Path(tar_path).stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("body annotation source changed during read-only audit")
    candidates.sort(key=lambda x: x["frame"])
    for index, candidate in enumerate(candidates):
        candidate["motion_proxy_mps"] = motion_proxy(candidates[index-1], candidate) if index else None
    report.update(candidates=len(candidates), body_times_used_as_surface_depth_gt=False,
                  first_last_body_frame=[candidates[0]["frame"], candidates[-1]["frame"]] if candidates else None)
    return candidates, report


def choose_samples(candidates, count=4):
    if not candidates:
        return []
    selected = []
    for label, fraction in (("early", .1), ("middle", .5), ("late", .9)):
        index = min(len(candidates)-1, round((len(candidates)-1)*fraction))
        if candidates[index]["frame"] not in {x["frame"] for x in selected}:
            selected.append(dict(candidates[index], stratum=label))
    moving = [x for x in candidates if x["motion_proxy_mps"] is not None]
    if moving:
        threshold = float(np.percentile([x["motion_proxy_mps"] for x in moving], 95))
        unused = [x for x in moving if x["frame"] not in {s["frame"] for s in selected}]
        if unused:
            picked = min(unused, key=lambda x: (abs(x["motion_proxy_mps"]-threshold), x["frame"]))
            selected.append(dict(picked, stratum="high_joint_motion_95pct"))
    for candidate in candidates:
        if len(selected) >= count:
            break
        if candidate["frame"] not in {x["frame"] for x in selected}:
            selected.append(dict(candidate, stratum="additional_body_presence"))
    return selected[:count]


def projected_rois(bodies, K, E, shape):
    height, width = shape
    rois = []
    for body in bodies:
        world = body["joints"][body["valid"], :3]
        camera = world @ np.asarray(E)[:3, :3].T + np.asarray(E)[:3, 3]
        valid = camera[:, 2] > .05
        if valid.sum() < 6:
            continue
        camera = camera[valid]
        uvh = camera @ np.asarray(K).T
        uv = uvh[:, :2] / uvh[:, 2, None]
        low, high = uv.min(axis=0), uv.max(axis=0)
        padding = np.maximum(16., .2*(high-low))
        low = np.floor(low-padding).astype(int)
        high = np.ceil(high+padding).astype(int)+1
        x0, y0 = np.maximum(low, [0, 0]); x1, y1 = np.minimum(high, [width, height])
        if x1 <= x0 or y1 <= y0 or (x1-x0)*(y1-y0) < 256:
            continue
        rois.append(dict(body_id=body["id"], bbox=[int(x0), int(y0), int(x1), int(y1)],
            projected_joints=uv.tolist(), joint_camera_depth_m=camera[:, 2].tolist(),
            approximate_roi_only=True))
    return rois


def roi_mask(rois, shape):
    mask = np.zeros(shape, dtype=bool)
    for roi in rois:
        x0, y0, x1, y1 = roi["bbox"]
        mask[y0:y1, x0:x1] = True
    return mask


def core_roi_mask(rois, shape):
    """A central/upper box proxy reduces floor contamination, not a body mask."""
    boxes = []
    for roi in rois:
        x0,y0,x1,y1 = roi["bbox"]
        width,height = x1-x0,y1-y0
        boxes.append(dict(bbox=[round(x0+.2*width),round(y0+.15*height),
                                round(x1-.2*width),round(y0+.65*height)]))
    return roi_mask(boxes,shape)


def adapter_options(value):
    if value is None:
        return {}
    try:
        result=json.loads(value)
    except json.JSONDecodeError:
        result=json.loads(Path(value).read_text())
    if not isinstance(result,dict):
        raise ValueError("adapter-options must contain a JSON object")
    reserved={"root","eligibility_manifest","cache_dir","exclude_videos","index_workers"}
    if reserved & set(result):
        raise ValueError("adapter-options cannot override audited dataset roots, admission, or caches")
    return result


def read_replay(path):
    """Accept the original baseline selection, enriching views/seed from its report."""
    path=Path(path)
    content=json.loads(path.read_text())
    companion=path.with_name("report.json")
    report=json.loads(companion.read_text()) if companion.exists() else {"scenes":[]}
    measured={(scene["scene"],sample["frame"]):sample
              for scene in report["scenes"] for sample in scene.get("samples",[])}
    rows={}
    for scene in content["scenes"]:
        if scene["scene"] in rows:
            raise ValueError("replay requires unique scene records")
        chosen=[]
        for original in scene["samples"]:
            item=dict(original)
            previous=measured.get((scene["scene"],item["frame"]),{})
            for key in ("sampling_seed","camera_names"):
                if key not in item and key in previous:
                    item[key]=previous[key]
            if "sampling_seed" not in item or not item.get("camera_names"):
                raise ValueError("exact replay requires original seed and camera_names, or a companion report")
            if len(set(item["camera_names"]))!=len(item["camera_names"]):
                raise ValueError("replay camera_names must be unique")
            item["bodies"]=[dict(body,joints=np.asarray(body["joints"],float),
                                valid=np.asarray(body["valid"],bool)) for body in item["bodies"]]
            chosen.append(item)
        if len({x["frame"] for x in chosen})!=len(chosen):
            raise ValueError("replay frame labels must be unique")
        rows[scene["scene"]]=chosen
    return rows


def fixed_camera_clip(adapter,scene,logical,camera_names,seed):
    """New reader pins views explicitly; legacy reader must reproduce exactly."""
    arguments=dict(layout_num_cameras=len(camera_names),layout_num_times=1,sampling_seed=seed)
    if "camera_names" in inspect.signature(adapter.load_clip).parameters:
        arguments["camera_names"]=list(camera_names)
    clip=adapter.load_clip(scene,[logical]*len(camera_names),**arguments)
    if clip.metadata["camera_names"]!=list(camera_names):
        raise UnsupportedReplay("exact replay camera views changed; refusing replacement views")
    return clip


def camera_seed(available, scores, *, trials=512):
    available = sorted(available)
    def key(seed):
        order = np.random.default_rng(seed).permutation(available).tolist()
        values = [scores.get(name, 0.) for name in order[:4]]
        return (sum(value > 0 for value in values), sum(np.log1p(values)), min(values, default=0.))
    return max(range(trials), key=key)


def structure_edges(depth, roi):
    """Observed inverse-depth curvature; a slanted plane is not an edge.

    Inverse Z is affine in pinhole image coordinates on a plane. Requiring
    known observations on both sides avoids interpreting depth holes or the
    ordinary first derivative of a slanted floor as a physical boundary.
    """
    valid = np.isfinite(depth) & (depth > 0)
    inverse=np.divide(1.,depth,out=np.zeros_like(depth),where=valid)
    result=np.zeros(depth.shape,bool)
    for radius in (1,2):
        for axis in (0,1):
            left=np.roll(inverse,radius,axis=axis);right=np.roll(inverse,-radius,axis=axis)
            good=valid & np.roll(valid,radius,axis=axis) & np.roll(valid,-radius,axis=axis)
            if axis==0:
                good[:radius]=False;good[-radius:]=False
            else:
                good[:,:radius]=False;good[:,-radius:]=False
            departure=np.abs(left+right-2*inverse)*depth*depth/(radius*radius)
            result |= good & (departure>(.03+.01*depth))
    return result & roi


def image_content_diagnostics(image,support=None):
    """Reject decoded constant-colour placeholders, without semantic segmentation."""
    image=np.asarray(image)
    support=np.ones(image.shape[:2],bool) if support is None else np.asarray(support,bool)
    if image.ndim!=3 or image.shape[2]!=3 or support.shape!=image.shape[:2]:
        raise ValueError("RGB content diagnostics require RGB HxWx3 and matching support")
    pixels=image[support]
    if len(pixels)<64:
        return dict(valid=False,status="insufficient_pixel_support",sampled_pixels=len(pixels))
    pixels=pixels[::max(1,len(pixels)//16384)].astype(np.float64)
    span=np.percentile(pixels,99,axis=0)-np.percentile(pixels,1,axis=0)
    gray=pixels@np.array([.299,.587,.114]);std=float(gray.std())
    uniform=bool((span<=1.).all() and std<=1.)
    return dict(valid=not uniform,status="constant_colour_placeholder" if uniform else "informative",
                sampled_pixels=len(pixels),rgb_p01_p99_span=span.tolist(),gray_std=std,
                median_rgb=np.median(pixels,axis=0).tolist(),semantic_mask_gt=False)


def edge_diagnostics(images, depth, roi, support=None):
    points = np.argwhere(structure_edges(depth, roi))
    result = dict(depth_structure_points=len(points), diagnostic_only=True,
                  hole_boundaries_are_not_surface_edges=True,
                  structure_method="observed_inverse_depth_second_difference_v1",
                  temporal_controls=[], pixel_controls=[])
    content=image_content_diagnostics(images[0],support)
    if not content["valid"]:
        result.update(status="invalid_rgb_content",rgb_content=content)
        return result
    if len(points) < 20:
        result["status"] = "insufficient_depth_edge_evidence"
        return result
    if len(points) > 10000:
        points = points[np.linspace(0, len(points)-1, 10000).astype(int)]
    y, x = points.T
    fields = {}
    for offset, image in sorted(images.items()):
        content=image_content_diagnostics(image,support)
        if not content["valid"]:
            result["temporal_controls"].append(dict(offset_frames=offset,status="invalid_rgb_content",median_px=None))
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rgb_edges = cv2.Canny(gray, 60, 160) > 0
        if not rgb_edges.any():
            result["temporal_controls"].append(dict(offset_frames=offset,status="insufficient_rgb_edges",median_px=None))
            continue
        fields[offset] = cv2.distanceTransform((~rgb_edges).astype(np.uint8), cv2.DIST_L2, 3)
        values = fields[offset][y, x]
        result["temporal_controls"].append(dict(offset_frames=offset, median_px=float(np.median(values)),
                                                p90_px=float(np.percentile(values, 90))))
    if 0 not in fields:
        result["status"]="insufficient_rgb_edge_evidence"
        return result
    for dx, dy in ((-8, 0), (-4, 0), (-2, 0), (0, 0), (2, 0), (4, 0), (8, 0), (0, -4), (0, 4)):
        yy, xx = y+dy, x+dx
        good = (yy >= 0)&(yy < depth.shape[0])&(xx >= 0)&(xx < depth.shape[1])
        if not good.any():
            continue
        values = fields[0][yy[good], xx[good]]
        result["pixel_controls"].append(dict(dx=dx, dy=dy, median_px=float(np.median(values))))
    measured=[value for value in result["temporal_controls"] if value["median_px"] is not None]
    best_time = min(measured, key=lambda x:(x["median_px"],abs(x["offset_frames"]),x["offset_frames"]))
    best_pixel = min(result["pixel_controls"], key=lambda x:(x["median_px"],abs(x["dx"])+abs(x["dy"])))
    zero_time = next(value for value in result["temporal_controls"] if value["offset_frames"] == 0)
    zero_pixel = next(value for value in result["pixel_controls"] if value["dx"] == value["dy"] == 0)
    result.update(status="measured", best_time_offset_frames=best_time["offset_frames"],
                  best_time_improvement_px=zero_time["median_px"]-best_time["median_px"],
                  best_pixel_offset=best_pixel,
                  best_pixel_improvement_px=zero_pixel["median_px"]-best_pixel["median_px"])
    return result


def patch_ncc(source, target, uv_source, uv_target, radius=3):
    if not len(uv_source):
        return dict(textured_pairs=0, median_ncc=None)
    a = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).astype(np.float32)
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    delta = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    pa, pb = np.asarray(uv_source)[:, None]+delta, np.asarray(uv_target)[:, None]+delta
    patches = []
    for gray, coordinates in ((a, pa), (b, pb)):
        patches.append(cv2.remap(gray, coordinates[..., 0].astype(np.float32),
            coordinates[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
    p, q = patches
    p -= p.mean(axis=1, keepdims=True); q -= q.mean(axis=1, keepdims=True)
    good = (p.std(axis=1) >= 8.) & (q.std(axis=1) >= 8.)
    if not good.any():
        return dict(textured_pairs=0, median_ncc=None)
    p, q = p[good], q[good]
    values = (p*q).sum(axis=1)/np.sqrt((p*p).sum(axis=1)*(q*q).sum(axis=1))
    return dict(textured_pairs=int(good.sum()), median_ncc=float(np.median(values)),
                positive_half_fraction=float((values >= .5).mean()),
                unwarped_patches_viewpoint_biased=True)


def cross_view_diagnostics(clip, roi_masks):
    result = []
    for src in range(len(clip.images)):
        depth = clip.depths[src]
        safe = (depth > 0) & roi_masks[src]
        safe[:4] = False; safe[-4:] = False; safe[:, :4] = False; safe[:, -4:] = False
        yx = np.argwhere(safe)
        if len(yx) > 512:
            yx = yx[np.linspace(0, len(yx)-1, 512).astype(int)]
        y, x = yx.T if len(yx) else (np.array([], int), np.array([], int))
        pixels = np.stack((x, y, np.ones(len(y))), axis=-1)
        local = (pixels @ np.linalg.inv(clip.intrinsics[src]).T)*depth[y, x, None]
        C = np.linalg.inv(clip.extrinsics[src]); world = local @ C[:3, :3].T+C[:3, 3]
        for dst in range(len(clip.images)):
            if src == dst:
                continue
            E = clip.extrinsics[dst]; p = world @ E[:3, :3].T+E[:3, 3]
            uvh = p @ clip.intrinsics[dst].T
            uv = uvh[:, :2]/np.maximum(uvh[:, 2, None], 1e-8)
            h, w = clip.depths[dst].shape
            good = (p[:, 2] > .05)&(uv[:, 0] >= 4)&(uv[:, 0] < w-4)&(uv[:, 1] >= 4)&(uv[:, 1] < h-4)
            rounded = np.rint(uv[good]).astype(int); z = p[good, 2]
            target = clip.depths[dst][rounded[:, 1], rounded[:, 0]]
            tolerance = .02+.01*np.minimum(z, np.where(target > 0, target, z))
            closer = (target > 0)&(target < z-tolerance)
            consistent = (target > 0)&(np.abs(target-z) <= tolerance)
            ncc = patch_ncc(clip.images[src], clip.images[dst],
                np.stack((x, y), axis=-1)[good][consistent], uv[good][consistent])
            result.append(dict(source=clip.metadata["camera_names"][src],
                target=clip.metadata["camera_names"][dst], source_points=len(y),
                projected_in_bounds=int(good.sum()), target_depth_missing=int((target <= 0).sum()),
                target_observed_closer_count=int(closer.sum()),
                consistent_depth_points=int(consistent.sum()), rgb_patch_ncc=ncc,
                observed_depth_only_not_complete_hd_visibility_gt=True))
    return result


def save_overlay(path, clip, rois):
    valid = [depth[depth > 0] for depth in clip.depths if (depth > 0).any()]
    if not valid:
        return
    low, high = np.percentile(np.concatenate(valid), [1, 99])
    rows = []
    for index in range(min(2, len(clip.images))):
        image = cv2.cvtColor(clip.images[index], cv2.COLOR_RGB2BGR)
        for roi in rois[index]:
            x0,y0,x1,y1 = roi["bbox"]; cv2.rectangle(image,(x0,y0),(x1-1,y1-1),(255,255,0),2)
            for x,y in roi["projected_joints"]:
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(image,(round(x),round(y)),2,(0,255,0),-1)
        depth = clip.depths[index]; valid = depth > 0
        colors = cv2.applyColorMap(np.clip((depth-low)/max(high-low,1e-6)*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
        colors[~valid] = 24
        overlay = image.copy(); overlay[valid] = (.4*image[valid]+.6*colors[valid]).astype(np.uint8)
        row = np.concatenate((image,colors,overlay),axis=1)
        cv2.putText(row,clip.metadata["camera_names"][index]+" RGB + joint ROI | measured GT | overlay",
                    (12,24),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
        rows.append(row)
    cv2.imwrite(str(path),np.concatenate(rows,axis=0))


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise ValueError("disable CUDA_VISIBLE_DEVICES for this CPU audit")
    cv2.setNumThreads(1)
    sys.path.insert(0, str(args.runtime))
    from multicamera4rc.datasets.adapters.panoptic_kinoptic import PanopticKinopticAdapter
    from multicamera4rc.datasets.adapters.base import InvalidClipError
    output = args.output.resolve()
    if any(root.resolve() == output or root.resolve() in output.parents for root in (args.root,args.body_root)):
        raise ValueError("audit output must be outside read-only dataset roots")
    output.mkdir(parents=True, exist_ok=False)
    source_files = gt_source_files(args.runtime)
    sources = {name:hashlib.sha256((args.runtime/name).read_bytes()).hexdigest() for name in source_files}
    report = dict(kind="kinoptic_dynamic_human_region_diagnostics_v1",status="indexing",pid=os.getpid(),
        started_unix=time.time(),runtime=str(args.runtime),source_sha256=sources,
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_versions=dict(opencv=cv2.__version__,numpy=np.__version__,python=sys.version),
        limitations=["Joint boxes are approximate ROIs, not silhouette or depth GT",
            "Patch NCC is viewpoint biased and not a truth or automatic acceptance criterion",
            "Observed-depth visibility does not certify unobserved HD occlusion",
            "No motion, static, segmentation or material-point labels are created"],
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},scenes=[])
    options=adapter_options(args.adapter_options)
    replay=read_replay(args.replay_selection) if args.replay_selection else None
    report.update(adapter_options=options,
        replay_selection_sha256=hashlib.sha256(args.replay_selection.read_bytes()).hexdigest() if replay else None)
    save(output/"report.json",report)
    admission = json.loads(args.manifest.read_text())
    scenes = list(replay) if replay is not None else admission["eligible_scenes"]
    if args.scene_list:
        path = Path(args.scene_list)
        scenes = json.loads(path.read_text()) if path.is_file() else args.scene_list.split(",")
    if len(set(scenes)) != len(scenes) or not set(scenes) <= set(admission["eligible_scenes"]):
        raise ValueError("scene-list requires unique independently admitted training scenes")
    exclude = json.loads(args.exclude_videos.read_text()) if args.exclude_videos else []
    parameters=dict(root=args.root, eligibility_manifest=args.manifest,
        max_long_side=960, cache_dir=args.cache, min_cameras=2, strict=True,
        exclude_videos=exclude,index_workers=1,video_cache_items=4,map_cache_items=4)
    parameters.update(options)
    adapter = PanopticKinopticAdapter(**parameters)
    selected_manifest = dict(scenes=[])
    started = time.monotonic()
    report.update(status="running",expected_scenes=scenes)
    for scene in scenes:
        row = dict(scene=scene,status="running",samples=[])
        report["scenes"].append(row);save(output/"report.json",report)
        try:
            record = adapter.records[scene]
            admitted_tar = Path(adapter.admission[scene]["body_tar"])
            body_tar = args.body_root/scene/admitted_tar.name
            if body_tar.resolve() != admitted_tar.resolve():
                raise ValueError("body-root does not match independently admitted body source")
            if replay is None:
                candidates, scan = collect_body_candidates(body_tar,
                    record["hd_frames"],record["hd_times_ms"],stride=args.body_stride)
                selected = choose_samples(candidates,args.samples)
            else:
                selected=replay.get(scene,[])[:args.samples]
                scan=dict(source="frozen_replay_body_roi_evidence",body_tar=str(body_tar),
                    candidates=len(selected),body_times_used_as_surface_depth_gt=False)
            row["body_metadata"] = scan
            row["selected_sample_count"] = len(selected)
            if not selected:
                row["status"] = "insufficient_body_evidence";continue
            lookup = {int(frame):i for i,frame in enumerate(record["hd_frames"])}
            selection_entry=dict(scene=scene,samples=json_safe(selected))
            selected_manifest["scenes"].append(selection_entry)
            save(output/"selection.json",selected_manifest)
            for sample_index,chosen in enumerate(selected):
                sample = dict(frame=chosen["frame"],time_ms=chosen["time_ms"],stratum=chosen["stratum"],
                    motion_proxy_mps=chosen["motion_proxy_mps"],body_member=chosen["body_member"],
                    body_sha256=chosen["body_sha256"],body_count=len(chosen["bodies"]),status="running")
                row["samples"].append(sample);save(output/"report.json",report)
                try:
                    if chosen["frame"] not in lookup:
                        raise UnsupportedReplay("frozen HD time no longer admitted; no replacement time sampled")
                    logical=lookup[chosen["frame"]]
                    if abs(float(record["hd_times_ms"][logical])-chosen["time_ms"])>1.:
                        raise ValueError("frozen body timestamp no longer matches original HD frame")
                    scores = {}
                    for name,camera in record["cameras"].items():
                        width,height = camera["resolution"];scale=min(1.,960/max(width,height))
                        w,h=round(width*scale),round(height*scale)
                        A=np.array([[w/width,0,.5*w/width-.5],[0,h/height,.5*h/height-.5],[0,0,1.]])
                        boxes=projected_rois(chosen["bodies"],A@camera["K"],camera["E"],(h,w))
                        scores[name]=sum((b["bbox"][2]-b["bbox"][0])*(b["bbox"][3]-b["bbox"][1]) for b in boxes)
                    seed=chosen["sampling_seed"] if replay is not None else camera_seed(record["cameras"],scores)
                    sample.update(sampling_seed=seed,roi_camera_scores=scores)
                    if replay is not None:
                        names=chosen["camera_names"]
                        sample["requested_camera_names"]=names
                        if not set(names)<=set(record["cameras"]):
                            raise UnsupportedReplay("frozen HD camera no longer available; no replacement view sampled")
                        try:
                            clip=fixed_camera_clip(adapter,scene,logical,names,seed)
                            sample["C4_supported"]=len(names)==4
                        except InvalidClipError as error:
                            raise UnsupportedReplay.from_reader(error) from error
                    else:
                        try:
                            clip=adapter.load_clip(scene,[logical]*4,layout_num_cameras=4,layout_num_times=1,sampling_seed=seed)
                            sample["C4_supported"]=True
                        except InvalidClipError as error:
                            sample.update(C4_supported=False,C4_reason=str(error))
                            clip=adapter.load_clip(scene,[logical]*2,layout_num_cameras=2,layout_num_times=1,sampling_seed=seed)
                    selection_entry["samples"][sample_index].update(sampling_seed=seed,
                        camera_names=clip.metadata["camera_names"])
                    save(output/"selection.json",selected_manifest)
                    rois=[];masks=[];views=[]
                    for index,name in enumerate(clip.metadata["camera_names"]):
                        depth=clip.depths[index];image=clip.images[index]
                        boxes=projected_rois(chosen["bodies"],clip.intrinsics[index],clip.extrinsics[index],depth.shape)
                        mask=roi_mask(boxes,depth.shape);rois.append(boxes);masks.append(mask)
                        core=core_roi_mask(boxes,depth.shape)
                        view=dict(camera=name,body_rois=boxes,valid_depth_fraction=float((depth>0).mean()),
                            roi_pixels=int(mask.sum()),roi_valid_depth_pixels=int(((depth>0)&mask).sum()),
                            roi_valid_depth_fraction=float(((depth>0)&mask).sum()/max(mask.sum(),1)),
                            core_bbox_pixels=int(core.sum()),core_bbox_valid_depth_pixels=int(((depth>0)&core).sum()),
                            core_bbox_valid_depth_fraction=float(((depth>0)&core).sum()/max(core.sum(),1)),
                            core_bbox_is_region_proxy_not_human_segmentation_gt=True,
                            per_body_coverage=[])
                        view["rgb_content"]=image_content_diagnostics(image,clip.metadata["pixel_support"][index])
                        view["reader_quality"]=clip.metadata.get("geometry_quality",{}).get(name,[])
                        for box in boxes:
                            bm=roi_mask([box],depth.shape)
                            view["per_body_coverage"].append(dict(body_id=box["body_id"],
                                roi_pixels=int(bm.sum()),valid_depth_pixels=int(((depth>0)&bm).sum())))
                        camera=record["cameras"][name]
                        offsets=[dt for dt in (-2,-1,0,1,2) if 0<=chosen["frame"]+dt<record["hd_video_frames"]]
                        raw=adapter._read_frames(camera,[chosen["frame"]+dt for dt in offsets])
                        mx,my,support,_,_=adapter._rectification(scene,name,camera)
                        ims={dt:cv2.remap(rgb,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                             for dt,rgb in zip(offsets,raw)}
                        view["edge_alignment"]=edge_diagnostics(ims,depth,mask,support)
                        if -2 in ims and 2 in ims:
                            diff=np.abs(cv2.cvtColor(ims[2],cv2.COLOR_RGB2GRAY).astype(float)
                                        -cv2.cvtColor(ims[-2],cv2.COLOR_RGB2GRAY).astype(float))
                            view["rgb_activity_proxy"]=dict(threshold_intensity=12,
                                roi_changed_fraction=float((diff[mask]>12).mean()) if mask.any() else None,
                                outside_roi_changed_fraction=float((diff[~mask]>12).mean()) if (~mask).any() else None,
                                estimated_activity_not_segmentation_gt=True)
                        views.append(view)
                    sample.update(camera_names=clip.metadata["camera_names"],views=views,
                        c2_first_two_roi_valid_depth_pixels=sum(v["roi_valid_depth_pixels"] for v in views[:2]),
                        c4_roi_valid_depth_pixels=sum(v["roi_valid_depth_pixels"] for v in views) if len(views)==4 else None,
                        depth_measurements=clip.metadata["depth_measurements"],
                        reader_geometry_quality=clip.metadata.get("geometry_quality",{}),
                        reader_rgb_verification=clip.metadata.get("rgb_verification"),
                        cross_view_diagnostics=cross_view_diagnostics(clip,masks))
                    filename=f"{scene}_{chosen['stratum']}_frame{chosen['frame']:08d}.png"
                    save_overlay(output/filename,clip,rois);sample["overlay"]=filename
                    bad_rgb=[v["camera"] for v in views if not v["rgb_content"]["valid"]]
                    sample["invalid_rgb_cameras"]=bad_rgb
                    sample["status"]=("invalid_rgb_content" if bad_rgb else
                        "measured" if any(v["roi_valid_depth_pixels"] for v in views) else "no_body_roi_depth_evidence")
                except UnsupportedReplay as error:
                    sample.update(status="unsupported_after_filter",reason=str(error),
                        replacement_time_or_camera_sampled=False,
                        rejected_cameras=error.rejected_cameras,
                        reader_error_type=error.reader_error_type)
                except Exception as error:
                    sample.update(status="failed",reason=str(error),traceback=traceback.format_exc())
                save(output/"report.json",report)
                print(json.dumps(dict(scene=scene,stratum=chosen["stratum"],frame=chosen["frame"],status=sample["status"])),flush=True)
            row["status"]="completed"
        except Exception as error:
            row.update(status="failed",reason=str(error),traceback=traceback.format_exc())
        finally:
            report["elapsed_seconds"]=time.monotonic()-started
            save(output/"report.json",report)
    adapter.close()
    report.update(status="completed",elapsed_seconds=time.monotonic()-started,
        source_hashes_unchanged=all(hashlib.sha256((args.runtime/name).read_bytes()).hexdigest()==value
                                  for name,value in sources.items()))
    save(output/"report.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("runtime","root","body-root","manifest","cache","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--scene-list",help="JSON list file or comma-separated admitted scene names")
    parser.add_argument("--exclude-videos",type=Path)
    parser.add_argument("--adapter-options",default="{}",help="Inline JSON object or JSON file passed to the reader")
    parser.add_argument("--replay-selection",type=Path,help="Replay frozen frames, seeds and exact HD views")
    parser.add_argument("--samples",type=int,default=4)
    parser.add_argument("--body-stride",type=int,default=10)
    args=parser.parse_args(argv)
    if args.samples<4 or args.body_stride<1:
        parser.error("samples must be >=4 and body-stride positive")
    run(args)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
