"""CPU toy contracts only: synthetic CUDA bytes test structure, not Philox validity."""
from collections import Counter
from copy import deepcopy
import ast
import importlib.util
import contextlib
import io
import json
import tempfile
import math
import os
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
NATIVE = Path(os.environ["EXPANSION_NATIVE_ROOT"])


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


expand = load("expansion_v14", HERE / "expand_checkpoint_v14.py")
schedule = load("expansion_native_schedule", NATIVE / "multicamera4rc/schedule.py")
optim = load("expansion_native_optim", NATIVE / "multicamera4rc/optim.py")
balance_module = load("expansion_native_balance", NATIVE / "multicamera4rc/load_balance.py")
balance_module.sample_cost = lambda dataset, index, *args: index % 17
SOURCE_SHA = "a" * 64


def state(size=1000, epoch=2, cursor=24, anchor=0, offset=7):
    return dict(world_size=5, batch_size=1, gradient_accumulation=1,
        effective_batch_size=5, sampler_kind="distributed_v1", dataset_size=size,
        data_epoch=epoch, batches_consumed=cursor, optimizer_updates=140000, seed=49,
        loss_config=dict(depth=1., query_pair=.1), loss_profile="custom",
        lr_plan=dict(mode="cosine", start_step=95000, end_step=200000,
            base_lrs=[1e-7, 1e-6, 1e-4, 1e-6], warmup_steps=500, min_ratio=.1,
            group_names=["backbone", "heads", "refinement", "geometry"],
            group_multipliers=dict(geometry=dict(start_step=139500, multiplier=2., warmup_steps=100))),
        geometry_lr_options=dict(multiplier=2., warmup_steps=100),
        data_migration=dict(step=139500, cursor_policy="explicit_reset_once"),
        data_config_sha256="b" * 64, augmentation=dict(enabled=True),
        load_balancing=dict(version=1, window_batches=8, world_size=5,
            dataset_size=size, start_epoch=epoch, start_batch=anchor,
            historical_padding_sample_offset=offset, image_hw={}, source_count=4,
            max_source_time_pairs=64))


def rng(seed):
    n = np.random.RandomState(seed).get_state()
    return dict(torch=torch.Generator().manual_seed(seed).get_state(),
        python=random.Random(seed).getstate(),
        numpy=(n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])),
        cuda=torch.arange(16, dtype=torch.uint8) + seed % 100)


def payload():
    groups, adam = [], {}
    for index, name in enumerate(("backbone", "heads", "refinement", "geometry")):
        groups.append(dict(name=name, params=[index], param_names=[name + ".weight"],
            lr=[1e-7, 1e-6, 1e-4, 1e-6][index], betas=(.9, .999),
            fused=None, foreach=None, weight_decay=.01))
        adam[index] = dict(step=torch.tensor(140000.),
            exp_avg=torch.full((2, 3), .2 + index), exp_avg_sq=torch.full((2, 3), .3 + index))
    return dict(architecture="multicamera4rc", format_version=2, step=140000,
        model={"weight": torch.arange(6.).reshape(2, 3)},
        optimizer=dict(state=adam, param_groups=groups),
        config=dict(gradient_checkpointing=True, reuse_decoder_kv=False),
        training_state=state(), rank_rng=[rng(49 + i) for i in range(5)], grad_scaler={})


def envelope():
    return dict(state=rng(54), provenance=dict(schema="guarded-cuda-generator-rank-rng-v1",
        rank=5, seed=54, source_checkpoint_sha256=SOURCE_SHA,
        method="torch.Generator(device=cuda).manual_seed(seed).get_state()",
        physical_gpu_uuid="GPU-00000000-0000-0000-0000-000000000007",
        torch_version=str(torch.__version__), guard_verified=True,
        test_fixture_only=True))


class ExpansionTests(unittest.TestCase):
    def test_real_native_sampler_and_native_resume_preserve_real_prefix_suffix(self):
        tree = ast.parse((NATIVE / "multicamera4rc/train.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_resume_data_cursor")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "native_resume_cursor", "exec"), namespace)
        for size in (127, 131, 997, 1000, 1001):
            for epoch in (0, 2):
                before = state(size=size, epoch=epoch)
                plan = expand.plan_expansion(before)
                after = expand.migrated_training_state(before, plan)
                cut = before["batches_consumed"] * 5
                def stream(world, balance, cursor):
                    ranks = []
                    for rank in range(world):
                        sampler = balance_module.BalancedDistributedSampler(range(size), balance=balance,
                            num_replicas=world, rank=rank, seed=49)
                        sampler.set_epoch(epoch)
                        sampler.start_batch = cursor
                        ranks.append(list(sampler))
                    return [i for row in zip(*ranks) for i in row]
                old = stream(5, before["load_balancing"], 0)
                new = stream(6, after["load_balancing"], after["batches_consumed"])
                raw = torch.randperm(size, generator=torch.Generator().manual_seed(49 + epoch)).tolist()
                padded = raw + raw[:expand.padded_epoch(size, 6) - size]
                self.assertEqual(Counter(old[:cut]), Counter(raw[:cut]))
                self.assertEqual(Counter(new[:cut]), Counter(raw[:cut]))
                self.assertEqual(Counter(new[cut:]), Counter(padded[cut:]))
                self.assertTrue(set(raw[cut:]) <= set(new[cut:]))
                self.assertEqual(expand.consumed_samples(before), expand.consumed_samples(after))
                self.assertEqual(namespace["_resume_data_cursor"](after, 140000, size, 6, 1,
                    sampler_kind="distributed_v1"), (epoch, cut // 6))

    def test_epoch_tail_padding_offset_and_completed_window_alignment(self):
        for size in (97, 100, 101, 997, 1000):
            before = state(size=size, cursor=math.ceil(size / 5))
            after = expand.migrated_training_state(before, expand.plan_expansion(before))
            self.assertEqual((after["data_epoch"], after["batches_consumed"]), (3, 0))
            self.assertEqual(expand.consumed_samples(before), expand.consumed_samples(after))
            self.assertEqual(after["load_balancing"][expand.OFFSET], 7 + 3 * (
                expand.padded_epoch(size, 5) - expand.padded_epoch(size, 6)))
        with self.assertRaisesRegex(ValueError, "fully consumed"):
            expand.plan_expansion(state(cursor=25))
        with self.assertRaisesRegex(ValueError, "not aligned"):
            expand.plan_expansion(state(cursor=32))
        with self.assertRaisesRegex(ValueError, "fully consumed"):
            expand.plan_expansion(state(cursor=24, anchor=1))
        step, plan = expand.next_aligned_step(state(size=1000, cursor=25))
        self.assertEqual(step, 140023)
        self.assertEqual(plan["batches_consumed"], 40)
        # Odd activation anchor has no common8-window/6-divisibility boundary.
        step, _ = expand.next_aligned_step(state(size=1000, cursor=25, anchor=1))
        self.assertEqual(step, 140175)

    def test_four_group_full_payload_and_retained_rng_are_exact_and_source_immutable(self):
        before = payload()
        snapshot = deepcopy(before)
        cpu_rng = torch.get_rng_state().clone()
        result = expand.transform_payload(before, envelope(), source_sha256=SOURCE_SHA)
        expand._equal(before, snapshot, "immutable_source")
        expand._equal(before["rank_rng"], result["rank_rng"][:5], "retained5")
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        for name in ("model", "optimizer", "config", "grad_scaler", "step"):
            expand._equal(before[name], result[name], name)
        for name in ("lr_plan", "loss_config", "geometry_lr_options", "data_migration",
                     "data_config_sha256", "augmentation", "optimizer_updates"):
            expand._equal(before["training_state"][name], result["training_state"][name], name)
        self.assertEqual(len(result["rank_rng"]), 6)
        self.assertFalse(result[expand.LINEAGE]["checkpoint_written"])

    def test_rng_bad_structure_provenance_and_payload_mutation_fail_closed(self):
        for mutate in (
            lambda e: e["state"].pop("cuda"),
            lambda e: e["state"].update(cuda=torch.ones(8, dtype=torch.uint8)),
            lambda e: e["provenance"].update(seed=49),
            lambda e: e["provenance"].update(guard_verified=False),
            lambda e: e["state"].update(numpy=("MT19937", [0], 0, 0, 0.)),
        ):
            value = envelope(); mutate(value)
            with self.assertRaises((ValueError, RuntimeError)):
                expand.transform_payload(payload(), value, source_sha256=SOURCE_SHA)
        before = payload()
        result = expand.transform_payload(before, envelope(), source_sha256=SOURCE_SHA)
        bad = deepcopy(result)
        bad["optimizer"]["state"][0]["exp_avg"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "tensor changed"):
            expand.verify_transformed_payload(before, bad, envelope(), source_sha256=SOURCE_SHA)
        bad = deepcopy(result); bad["rank_rng"][0]["cuda"][0] += 1
        with self.assertRaisesRegex(ValueError, "tensor changed"):
            expand.verify_transformed_payload(before, bad, envelope(), source_sha256=SOURCE_SHA)

    def test_cli_toy_dry_run_writes_only_optional_json_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "toy_source.pt"
            rng_path = directory / "toy_rng.pt"
            report = directory / "receipt.json"
            torch.save(payload(), source)
            source_sha = expand.sha256_file(source)
            value = envelope()
            value["provenance"]["source_checkpoint_sha256"] = source_sha
            torch.save(value, rng_path)
            before = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                expand.main(["dry-run", "--source", str(source),
                    "--source-sha256", source_sha, "--new-rank-rng", str(rng_path),
                    "--new-rank-rng-sha256", expand.sha256_file(rng_path),
                    "--report", str(report)])
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(set(p.name for p in directory.iterdir()),
                {"toy_source.pt", "toy_rng.pt", "receipt.json"})
            receipt = json.loads(report.read_text())
            self.assertFalse(receipt["checkpoint_written"])
            self.assertTrue(receipt["dry_run_verified"])
            self.assertEqual(receipt["rank_rng_count"], 6)

    def test_native_geometry_lr_keeps_global_cosine_and_absolute_warm_anchor(self):
        old = schedule.LearningRatePlan(mode="cosine", start_step=95000, end_step=200000,
            base_lrs=[1e-7, 1e-6, 1e-4], warmup_steps=500, min_ratio=.1,
            warmup_start_ratio=.19018677306955883)
        start = 140000
        new = schedule.migrate_geometry_plan(old, ["backbone", "heads", "refinement"],
            ["backbone", "heads", "refinement", "geometry"], start_step=start,
            multiplier=2., warmup_steps=100)
        for step in (start - 1, start, start + 1, start + 50, start + 100, 199999):
            self.assertEqual(new.factor(step), old.factor(step))
        self.assertEqual(new.group_factor("geometry", start), 1.)
        self.assertEqual(new.group_factor("geometry", start + 1), 1.01)
        self.assertEqual(new.group_factor("geometry", start + 50), 1.5)
        self.assertEqual(new.group_factor("geometry", start + 100), 2.)
        self.assertEqual(new.base_lrs, [1e-7, 1e-6, 1e-4, 1e-6])
        restored = schedule.LearningRatePlan(**new.to_config())
        self.assertEqual(restored.to_config(), new.to_config())
        with self.assertRaisesRegex(ValueError, "already exists"):
            schedule.migrate_geometry_plan(new, new.group_names, new.group_names,
                start_step=start + 20)

    def test_native_named_adam_three_to_four_and_four_to_four_preserve_moments(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Module()
                self.backbone.pretrained = torch.nn.Linear(2, 2)
                self.head = torch.nn.Linear(2, 2)
                self.cam_dec = torch.nn.Linear(2, 2)
                self.motion_head = torch.nn.Linear(2, 2)
                self.trajectory_refiner = torch.nn.Linear(2, 2)
        torch.manual_seed(73)
        model = Model()
        old = torch.optim.AdamW(optim.optimizer_groups(model, 1e-7, 1e-6, 1e-4))
        for i, parameter in enumerate(model.parameters()):
            parameter.grad = torch.full_like(parameter, .1 + i)
        old.step()
        saved = deepcopy(old.state_dict())
        new = torch.optim.AdamW(optim.optimizer_groups(model, 1e-7, 1e-6, 1e-4, geometry_lr=1e-6))
        receipt = optim.load_optimizer_state(new, saved, model, model_state=model.state_dict())
        self.assertTrue(receipt["geometry_split"])
        by_name = {name: saved["state"][index] for group in saved["param_groups"]
                   for name, index in zip(group["param_names"], group["params"])}
        four = deepcopy(new.state_dict())
        for group in four["param_groups"]:
            for name, index in zip(group["param_names"], group["params"]):
                expand._equal(by_name[name], four["state"][index], name)
        restored = torch.optim.AdamW(optim.optimizer_groups(model, 1e-7, 1e-6, 1e-4, geometry_lr=1e-6))
        receipt = optim.load_optimizer_state(restored, four, model)
        self.assertFalse(receipt["geometry_split"])
        expand._equal(four, restored.state_dict(), "ordinary4group_resume")

    def test_data_composer_within_family_camera_bucket_preserves_mono_mass(self):
        mono = .3781580347498024
        base = {("kubric_variable", 1): mono * .4, ("pointodyssey", 1): mono * .6,
                ("kubric_variable", 2): (1 - mono) * .3, ("other", 2): (1 - mono) * .7}
        rows = []
        for (family, camera), weight in base.items():
            long_fraction = .2 if family in ("kubric_variable", "pointodyssey") else 0.
            rows += [(family, camera, 16, weight * (1 - long_fraction)),
                     (family, camera, 32, weight * long_fraction)]
        for key, weight in base.items():
            self.assertAlmostEqual(sum(w for f, c, t, w in rows if (f, c) == key), weight)
        self.assertAlmostEqual(sum(w for _, c, _, w in rows if c == 1), mono)
        self.assertAlmostEqual(sum(w for _, _, _, w in rows), 1.)


if __name__ == "__main__":
    print("TOY CPU CONTRACTS: synthetic CUDA bytes are not evidence of valid CUDA RNG")
    print("torch_version", torch.__version__)
    unittest.main(verbosity=2)
