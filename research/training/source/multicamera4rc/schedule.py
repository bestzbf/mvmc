"""Explicit, checkpointed learning-rate phases for supervised continuation."""

from dataclasses import asdict, dataclass, replace
import math


@dataclass
class LearningRatePlan:
    mode: str
    start_step: int
    end_step: int
    base_lrs: list[float]
    warmup_steps: int = 0
    min_ratio: float = .1
    warmup_start_ratio: float = 0.
    group_names: list[str] | None = None
    group_multipliers: dict | None = None

    def __post_init__(self):
        if self.mode not in ("constant", "cosine"):
            raise ValueError("learning-rate mode must be constant/cosine")
        if self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError("learning-rate phase end must exceed its start")
        if self.warmup_steps < 0 or self.warmup_steps >= self.end_step-self.start_step:
            raise ValueError("warmup must fit inside the learning-rate phase")
        if not math.isfinite(self.min_ratio) or not 0 < self.min_ratio <= 1:
            raise ValueError("min learning-rate ratio must lie in (0,1]")
        if not math.isfinite(self.warmup_start_ratio) or not 0 <= self.warmup_start_ratio <= 1:
            raise ValueError("warmup start ratio must lie in [0,1]")
        if any(not math.isfinite(lr) or lr <= 0 for lr in self.base_lrs):
            raise ValueError("base learning rates must be finite and positive")
        if self.group_names is not None:
            if (len(self.group_names) != len(self.base_lrs)
                    or any(not isinstance(name, str) or not name for name in self.group_names)
                    or len(set(self.group_names)) != len(self.group_names)):
                raise ValueError('learning-rate group names must be unique and match base rates')
        if self.group_multipliers is not None:
            if self.group_names is None or not isinstance(self.group_multipliers, dict):
                raise ValueError('group multipliers require named learning-rate groups')
            for name, policy in self.group_multipliers.items():
                if name not in self.group_names or not isinstance(policy, dict) or set(policy) != {
                        'start_step', 'multiplier', 'warmup_steps'}:
                    raise ValueError('invalid learning-rate group multiplier policy')
                if (not isinstance(policy['start_step'], int) or policy['start_step'] < 0
                        or not isinstance(policy['warmup_steps'], int) or policy['warmup_steps'] < 0
                        or not math.isfinite(policy['multiplier']) or policy['multiplier'] <= 0):
                    raise ValueError('invalid learning-rate group multiplier values')

    def factor(self, step):
        position = max(0, step-self.start_step)
        if position < self.warmup_steps:
            return self.warmup_start_ratio + (1-self.warmup_start_ratio)*(position+1)/self.warmup_steps
        if self.mode == "constant":
            return 1.
        span = max(1, self.end_step-self.start_step-self.warmup_steps-1)
        progress = min(1., max(0., (position-self.warmup_steps)/span))
        return self.min_ratio+(1-self.min_ratio)*(1+math.cos(math.pi*progress))/2

    def apply(self, optimizer, step):
        if len(optimizer.param_groups) != len(self.base_lrs):
            raise ValueError("learning-rate plan does not match optimizer groups")
        if self.group_names is not None and [g.get('name') for g in optimizer.param_groups] != self.group_names:
            raise ValueError('learning-rate group identity/order differs from the saved plan')
        for index, (group, lr) in enumerate(zip(optimizer.param_groups, self.base_lrs)):
            name = self.group_names[index] if self.group_names is not None else None
            group["lr"] = lr*self.factor(step)*self.group_factor(name, step)

    def group_factor(self, name, step):
        policy = (self.group_multipliers or {}).get(name)
        if policy is None or step < policy['start_step']:
            return 1.
        warmup = policy['warmup_steps']
        progress = min(1., (step-policy['start_step'])/warmup) if warmup else 1.
        return 1.+(policy['multiplier']-1.)*progress

    def to_config(self):
        config = asdict(self)
        # Keep legacy serialization unchanged when no group-specific phase exists.
        return {key: value for key, value in config.items()
                if key not in ('group_names', 'group_multipliers') or value is not None}


def migrate_geometry_plan(plan, old_group_names, new_group_names, *, start_step,
                          multiplier=2., warmup_steps=100):
    """Split geometry off heads without restarting the saved cosine phase.

    The new group inherits the heads' original base LR. Only a named multiplier
    ramps from one at ``start_step`` to its target after ``warmup_steps``. Its
    absolute anchor is checkpointed, so ordinary resume never repeats warmup.
    """
    old_group_names, new_group_names = list(old_group_names), list(new_group_names)
    if (len(old_group_names) != len(plan.base_lrs) or len(set(old_group_names)) != len(old_group_names)
            or len(set(new_group_names)) != len(new_group_names)
            or 'heads' not in old_group_names or 'geometry' not in new_group_names):
        raise ValueError('geometry LR migration requires complete, unique old/new group names')
    if plan.group_names is not None and plan.group_names != old_group_names:
        raise ValueError('saved LR plan group identity does not match source optimizer')
    if 'geometry' in old_group_names:
        raise ValueError('geometry LR phase already exists; restore its saved plan without restarting warmup')
    if set(new_group_names) != set(old_group_names) | {'geometry'}:
        raise ValueError('geometry LR migration may only add the geometry group')
    if not isinstance(start_step, int) or start_step < 0 or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError('geometry LR migration requires nonnegative integer step/warmup')
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('geometry LR multiplier must be finite and positive')
    rates = dict(zip(old_group_names, plan.base_lrs))
    rates['geometry'] = rates['heads']
    policies = dict(plan.group_multipliers or {})
    policies['geometry'] = dict(start_step=start_step, multiplier=float(multiplier), warmup_steps=warmup_steps)
    return replace(plan, base_lrs=[rates[name] for name in new_group_names],
                   group_names=new_group_names, group_multipliers=policies)
