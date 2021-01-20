import os
import pandas as pd
from data import PROJECT_RAW_DATA_FOLDER


class DataLoaders:
    @staticmethod
    def get_cases_by_city() -> pd.DataFrame:
        cities_data_path = os.path.join(PROJECT_RAW_DATA_FOLDER,
                                        'corona_city_table_ver_0035.csv')
        cities_df = pd.read_csv(cities_data_path, encoding='utf-8-sig')
        return cities_df

    @staticmethod
    def get_vaccinations_by_age_and_city() -> pd.DataFrame:
        vaccination_data_path = os.path.join(PROJECT_RAW_DATA_FOLDER,
                                             'vaccinated_by_age_2021_01_13.csv')
        vaccination_df = pd.read_csv(vaccination_data_path, encoding='utf-8-sig')
        return vaccination_df

    @staticmethod
    def get_population_age_groups_by_city() -> pd.DataFrame:
        population_path = os.path.join(PROJECT_RAW_DATA_FOLDER,
                                       'israel-city_pop.csv')
        pop_df = pd.read_csv(population_path, encoding='utf-8-sig')
        return pop_df

    @staticmethod
    def get_socioeconomic_rank_by_city() -> pd.DataFrame:
        rank_data_path = os.path.join(PROJECT_RAW_DATA_FOLDER,
                                      'se_index_2017.csv')
        rank_df = pd.read_csv(rank_data_path, encoding='utf-8-sig')
        return rank_df
