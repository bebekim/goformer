"""Minimal self-play experience buffer: (state, visit-count target,
final reward) triples, saved/loaded as a single .npz so train.py can
fine-tune a checkpoint without re-running self-play."""
import numpy as np


class ZeroExperienceCollector:
    def __init__(self):
        self.states = []
        self.visit_counts = []
        self.rewards = []
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def record_decision(self, state, visit_counts):
        self._current_episode_states.append(state)
        self._current_episode_visit_counts.append(visit_counts)

    def complete_episode(self, reward):
        num_moves = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.visit_counts += self._current_episode_visit_counts
        self.rewards += [reward] * num_moves
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def to_arrays(self):
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.visit_counts, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
        )

    def save(self, path):
        states, visit_counts, rewards = self.to_arrays()
        np.savez_compressed(path, states=states, visit_counts=visit_counts, rewards=rewards)


def combine_experience(collectors):
    combined = ZeroExperienceCollector()
    for c in collectors:
        combined.states += c.states
        combined.visit_counts += c.visit_counts
        combined.rewards += c.rewards
    return combined


def load_experience(path):
    data = np.load(path)
    return data['states'], data['visit_counts'], data['rewards']
