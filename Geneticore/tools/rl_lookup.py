from Geneticore.action_parsers import DiscreteAction
import numpy as np

#Tailored to rocket league, taken from rlgym
class RL_LookupAction(DiscreteAction):
    def __init__(self, bins=[(-1, 0, 1)] * 5):
        self.act_bins = bins

        self._lookup_table = self.make_lookup_table(self.act_bins)

        self.bins = len(self._lookup_table)

    def make_lookup_table(self, bins):
        actions = []
        # Ground
        for throttle in bins[0]:
            for steer in bins[1]:
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in bins[2]:
            for yaw in bins[3]:
                for roll in bins[4]:
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])

        actions = np.array(actions)
        return actions

    def return_act_space(self):
      return self.bins

    def parse_actions(self, activations) -> np.ndarray:
        action = self.parse(activations)
        return self._lookup_table[action]