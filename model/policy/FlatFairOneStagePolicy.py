import math

import torch

from model.policy.OneStagePolicy_OnPolicy import OneStagePolicy_OnPolicy


class FlatFairOneStagePolicy(OneStagePolicy_OnPolicy):
    """The train_ppo one-stage policy with a catalogue-entropy diagnostic."""

    def __init__(self, args, environment):
        super().__init__(args, environment)
        if self.item_num <= 1:
            raise ValueError(
                'normalized fairness entropy requires at least two items'
            )
        self.display_name = 'FlatFairOneStagePolicy'

    @staticmethod
    def normalized_catalogue_entropy(scores):
        """Return H(softmax(scores)) / log(catalogue size) per user."""
        if scores.shape[-1] <= 1:
            raise ValueError(
                'normalized fairness entropy requires at least two items'
            )
        log_probabilities = torch.log_softmax(scores, dim=-1)
        probabilities = log_probabilities.exp()
        entropy = -torch.sum(
            probabilities * log_probabilities, dim=-1
        )
        return entropy / math.log(scores.shape[-1])

    def _extra_policy_outputs(self, scores):
        return {
            'fairness_reward': self.normalized_catalogue_entropy(scores),
        }
