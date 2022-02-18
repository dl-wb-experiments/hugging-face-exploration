import logging

from constants import LOGS_PATH


def setup_logging(is_mp=False):
    LOGS_PATH.mkdir(exist_ok=True)

    file_name = 'check_hf_mp.log' if is_mp else 'check_hf.log'
    logging.basicConfig(filename=str(LOGS_PATH / file_name),
                        level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filemode='w')
