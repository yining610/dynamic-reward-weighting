from typing import List, Any, Dict, Union, Optional
import math
import torch
import wandb

class Metric:
    def __init__(self):
        pass

    def add(self, val):
        raise NotImplementedError

    def val(self) -> float:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def compute(self, val: Any):
        return val

    def __add__(self, other):
        raise NotImplementedError
    
    def __radd__(self, other):
        return self.__add__(other)


class MeanMetric(Metric):
    def __init__(self, num=0, denom=0):
        self.numerator = num
        self.denominator: int = denom

    def add(self, val: Any):
        self.numerator += self.compute(val)
        self.denominator += 1

    def many(self, vals: List[Any], denoms: Optional[List[int]] = None):
        if denoms is None:
            denoms = [1] * len(vals)
        assert len(vals) == len(denoms)

        for v, n in zip(vals, denoms):
            self.numerator += self.compute(v)
            self.denominator += n
    
    def val(self):
        if self.denominator == 0:
            return 0
        return self.numerator / self.denominator

    def reset(self):
        self.numerator = self.denominator = 0

    def __add__(self, other: 'MeanMetric'):
        return MeanMetric(self.numerator + other.numerator, self.denominator + other.denominator)

class SumMetric(Metric):
    def __init__(self, sum_=0):
        self.sum_ = sum_

    def add(self, val):
        self.sum_ += self.compute(val)

    def many(self, vals: List[Any]):
        self.sum_ += sum(self.compute(v) for v in vals)

    def val(self):
        return self.sum_

    def reset(self):
        self.sum_ = 0

    def __add__(self, other: 'SumMetric'):
        return SumMetric(self.sum_ + other.sum_)

class RealtimeMetric(Metric):
    def __init__(self, val=0):
        self.v = val

    def add(self, val):
        self.v = self.compute(val)
        
    def many(self, vals: List[Any]):
        self.add(vals[-1])
    
    def val(self):
        return self.v

    def reset(self):
        self.v = 0

    def __add__(self, other):
        return RealtimeMetric(self.v)

def StringMetric(Metric):
    def __init__(self, val=[]):
        self.v = val
    
    def add(self, val):
        self.v.append(val)
    
    def many(self, vals: List[Any]):
        self.v.extend(vals)
    
    def val(self):
        return self.v
    
    def reset(self):
        self.v = []
    
    def __add__(self, other):
        return StringMetric(self.v + other.v)

class Metrics():

    def __init__(self, mode):
        self.metrics = {}
        self.mode = mode

    def create_metric(self, metric_name: str, metric_obj: Metric):
        assert metric_name not in self.metrics
        self.metrics[metric_name] = metric_obj

    def record_metric(self, metric_name: str, val: Any):
        self.metrics[metric_name].add(val)

    def record_metric_many(self, metric_name: str, vals: List[Any], counts: Optional[List[int]] = None):
        if counts is None:
            self.metrics[metric_name].many(vals)
        else:
            self.metrics[metric_name].many(vals, counts)

    def reset(self, no_reset = ['global_num_examples']):
        for k, v in self.metrics.items():
            if k not in no_reset:
                v.reset()
                
    def all_gather_metrics(self):
        with torch.no_grad():
            metrics_tensor = {k: torch.tensor([v.val()]) if isinstance(v.val(), Union[int, float]) else v.val() for k, v in self.metrics.items()}
            gathered_metrics = metrics_tensor             
            gathered_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in gathered_metrics.items()}
        return gathered_metrics