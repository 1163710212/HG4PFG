import numpy as np
import torch


EXPOSURE_METRICS = ('ad', 'coverage')


def get_exposure_metrics(environment, action_indices, active_mask=None):
    """Compute per-step exposure metrics for the evaluated users only.

    ``coverage`` follows the simulator's existing definition: the number of
    distinct items exposed in the current batch. ``ad`` is the absolute gap
    between popular- and long-tail-item exposure shares. During user-exit
    evaluation, ``active_mask`` excludes users after their first exit.
    """
    actions = action_indices
    if active_mask is not None:
        actions = actions[active_mask]
    if actions.numel() == 0:
        return {}

    metrics = {
        'coverage': float(torch.unique(actions).numel()),
    }
    if hasattr(environment, 'item_types'):
        item_types = environment.item_types[actions.reshape(-1)].float()
        popular_share = torch.mean(item_types)
        metrics['ad'] = float(
            torch.abs(popular_share - (1.0 - popular_share)).item()
        )
    return metrics


def append_exposure_metrics(samples, metrics):
    for key in EXPOSURE_METRICS:
        if key in metrics:
            samples.setdefault(key, []).append(float(metrics[key]))


def add_exposure_metrics_to_report(report, samples):
    for key in EXPOSURE_METRICS:
        values = samples.get(key, [])
        if values:
            report[key] = float(np.mean(values))
