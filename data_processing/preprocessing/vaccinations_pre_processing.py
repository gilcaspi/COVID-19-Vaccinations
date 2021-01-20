import collections
from typing import OrderedDict, Callable, Tuple

from data_processing.preprocessing.pre_processing_actions import missing_data_imputation, \
    filter_cities_with_missing_vaccinations_data, convert_to_int
from data_processing.preprocessing.pre_processing_base import PreProcessingBase


class VaccinationsPreProcessing(PreProcessingBase):
    def _get_actions(self) -> OrderedDict[Callable, Tuple]:
        actions = collections.OrderedDict()
        # Key = callable function to perform
        # Value = Arguments as a tuple
        actions[filter_cities_with_missing_vaccinations_data] = (1,)
        actions[missing_data_imputation] = ('< 15', 1)
        actions[convert_to_int] = ('City_Name', None)

        return actions
