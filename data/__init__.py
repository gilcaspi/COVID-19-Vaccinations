import os

PROJECT_ROOT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   os.pardir))

PROJECT_DATA_FOLDER = os.path.join(PROJECT_ROOT_FOLDER,
                                   'data')

PROJECT_PROCESSED_DATA_FOLDER = os.path.join(PROJECT_DATA_FOLDER,
                                             'processed')

PROJECT_RAW_DATA_FOLDER = os.path.join(PROJECT_DATA_FOLDER,
                                       'raw')
