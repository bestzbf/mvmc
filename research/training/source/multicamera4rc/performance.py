"""Optional synchronized phase timings for short, matched speed experiments."""
import time
import torch


class StepTimer:
    def __init__(self, device, detailed=False):
        self.device, self.detailed = device, detailed
        self.last = time.perf_counter()
        self.begin = self.last
        self.phases = {}

    def mark(self, name):
        if self.detailed and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        self.phases[name] = now - self.last
        self.last = now

    def finish(self):
        now = time.perf_counter()
        result = {'seconds': now - self.begin}
        if self.detailed:
            result['phases'] = dict(self.phases)
        self.begin = self.last = now
        self.phases.clear()
        return result
