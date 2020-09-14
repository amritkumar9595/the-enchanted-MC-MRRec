

import argparse
import pathlib


class Args(argparse.ArgumentParser):
    """
    Defines global default arguments.
    """

    def __init__(self, **overrides):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.add_argument('--seed', default=42, type=int, help='Seed for random number generators')

        
        self.add_argument('--sample-rate', type=float, default=1.,
                          help='Fraction of total volumes to include')

        self.add_argument('--center-fractions', nargs='+', default=[0.08], type=float,
                          help='Fraction of low-frequency k-space columns to be sampled. Should '
                               'have the same length as accelerations')

        # Override defaults with passed overrides
        self.set_defaults(**overrides)
