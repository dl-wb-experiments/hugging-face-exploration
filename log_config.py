import logging

from constants import LOGS_PATH


def setup_logging(is_mp=False, log_filename=None):
    LOGS_PATH.mkdir(exist_ok=True)

    if log_filename is None:
        file_name = 'check_hf_mp.log' if is_mp else 'check_hf.log'
    else:
        file_name = log_filename
    logging.basicConfig(filename=str(LOGS_PATH / file_name),
                        level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filemode='w')
