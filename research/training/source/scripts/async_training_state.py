"""Durable validation queue, bounded checkpoint retention and live log audit."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re

from scripts.continue_refinement_training import next_boundary
from scripts.cutover_native_motion_stereo import write_json
from scripts.independent_validation import file_identity, require_identity
from scripts.native_training_log import _STEP
from scripts.query_pair_main_admission import read


class UpdateAudit:
    """Read only newly appended complete records; incomplete writes stay buffered."""
    def __init__(self, path, begin, world, accumulation, batch):
        self.path, self.begin, self.last = Path(path), begin, begin
        self.world, self.accumulation, self.batch = world, accumulation, batch
        self.offset, self.pending, self.identity = 0, b'', None
        self.latest = None

    def poll(self):
        if not self.path.exists():
            return
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if (self.identity is not None and identity != self.identity) or stat.st_size < self.offset:
            raise ValueError('live training log was replaced or truncated')
        self.identity = identity
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            raw = self.pending + stream.read()
            self.offset = stream.tell()
        complete, separator, self.pending = raw.rpartition(b'\n')
        if not separator:
            self.pending = raw
            return
        decoder = json.JSONDecoder()
        for line in complete.decode('utf-8').splitlines():
            position = 0
            while match := _STEP.search(line, position):
                row, position = decoder.raw_decode(line, match.start())
                self.accept(row)

    def accept(self, row):
        if type(row.get('step')) is not int or row['step'] != self.last + 1:
            raise ValueError('missing, repeated or out-of-order native update')
        if (row['world_size'] != self.world or row['gradient_accumulation'] != self.accumulation
                or row['effective_batch_size'] != self.batch or len(row['rank_batches']) != self.batch
                or row['optimizer_step_skipped'] or row['supervision'] != 'dense'):
            raise ValueError('native execution, optimizer update or dense supervision changed')
        if any(not math.isfinite(row[k]) for k in ('loss', 'grad_norm', 'lr', 'backbone_lr', 'refinement_lr')):
            raise ValueError('nonfinite native update')
        for sample in row['rank_batches']:
            m = sample['metrics']
            if any(not math.isfinite(v) for v in m.values() if isinstance(v, (int, float))):
                raise ValueError('nonfinite sample metric')
            geometry = 'depth_abs_rel' in m and 'depth_mae_normalized' in m
            camera = 'intrinsics_log_focal_error' in m and 'camera_rotation_degrees' in m
            if m.get('supervised_source_pixels', 0) <= 0 and not geometry and not camera:
                raise ValueError('sample has no admitted supervision')
        self.last, self.latest = row['step'], {k: row[k] for k in (
            'step', 'loss', 'grad_norm', 'lr', 'backbone_lr', 'refinement_lr')}


def validation_steps(begin, initial, target):
    result = []
    while begin < target:
        begin = next_boundary(begin, initial, target)
        result.append(begin)
    return result


class ValidationQueue:
    """Only completed immutable files enter the queue. Publication is atomic."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def entries(self):
        return [read(path) for path in sorted(self.directory.glob('step_*/entry.json'))]

    def enqueue(self, source, step, job):
        directory = self.directory / f'step_{step:08d}'
        directory.mkdir(exist_ok=True)
        entry_path, pin = directory / 'entry.json', directory / f'step_{step:08d}.pt'
        source_identity = file_identity(source)
        if entry_path.exists():
            entry = read(entry_path)
            if entry['step'] != step or entry['checkpoint_identity'] != source_identity:
                raise ValueError('validation step already refers to a different checkpoint')
            require_identity(entry['checkpoint'], entry['checkpoint_identity'])
            return entry
        # Same filesystem is intentional: pinning 18 GB must not stall training
        # or duplicate storage. Never fall back to an unverified partial copy.
        try:
            os.link(source, pin)
        except FileExistsError:
            require_identity(pin, source_identity)
        require_identity(pin, source_identity)
        frozen = deepcopy(job)
        frozen.update(control_dir=str(directory), validation_checkpoint=str(pin), validation_step=step,
            validation_label='refinement_' + str(step), checkpoint_identity=source_identity)
        frozen.pop('pid', None)
        write_json(directory / 'job.json', frozen)
        entry = dict(step=step, checkpoint=str(pin), checkpoint_identity=source_identity,
            directory=str(directory), status='queued', attempts=0)
        write_json(entry_path, entry)
        return entry

    def update(self, entry, **changes):
        entry = dict(entry, **changes)
        write_json(Path(entry['directory']) / 'entry.json', entry)
        return entry

    def pending(self):
        return [e for e in self.entries() if e['status'] not in ('completed', 'pruned')]


def retention_candidates(run, entries, protected=(), keep_latest=6):
    """Only this run's numbered, completed saves; retain queued/running pins."""
    numbered = []
    for path in Path(run).glob('step_*.pt'):
        match = re.fullmatch(r'step_(\d+)\.pt', path.name)
        if match and not path.is_symlink():
            numbered.append((int(match[1]), path))
    numbered.sort()
    keep = {str(Path(p).resolve()) for p in protected}
    pinned = {(e['checkpoint_identity']['device'], e['checkpoint_identity']['inode']) for e in entries
              if e['status'] not in ('completed', 'pruned')}
    latest = {p for _, p in numbered[-keep_latest:]}
    return [p for step, p in numbered if p not in latest and step % 10000
            and str(p.resolve()) not in keep and (p.stat().st_dev, p.stat().st_ino) not in pinned]
