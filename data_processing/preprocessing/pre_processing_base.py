import abc
import collections
from typing import List, Callable, Optional, OrderedDict, Tuple

import pandas as pd


class PreProcessingBase:
    def __init__(self,
                 df: pd.DataFrame,
                 actions: Optional[OrderedDict[Callable, Tuple]] = None):
        self._df = df
        self._actions = actions
        if self._actions is None:
            self._actions = collections.OrderedDict()

    @abc.abstractmethod
    def _get_actions(self) -> OrderedDict[Callable, Tuple]:
        raise NotImplementedError

    def setup(self):
        self._actions = self._get_actions()
        return self

    def run(self) -> pd.DataFrame:
        for action, args in self._actions.items():
            self._df = self._df.apply(action, args=args)

        return self._df

