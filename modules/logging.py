import os
from tqdm import tqdm
import logging
import datetime

def get_logger(task_name: str, log_dir: str):

    current_time = datetime.datetime.now()
    time_str = current_time.strftime('%Y-%m-%d %H-%M-%S')
    

    # console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, 'logger.log'), mode="a", encoding="utf-8")

    console_fmt = "(%(levelname)s) %(asctime)s - %(name)s:\n%(message)s"
    file_fmt = "(%(levelname)s) %(asctime)s  - %(name)s:\n%(message)s"

    console_formatter = logging.Formatter(fmt = console_fmt)
    file_formatter = logging.Formatter(fmt = file_fmt)

    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setLevel(logging.INFO)
    tqdm_handler.setFormatter(fmt = console_formatter)
    file_handler.setFormatter(fmt = file_formatter)

    logging.basicConfig(level='INFO', handlers=[tqdm_handler, file_handler])
    logger = logging.getLogger(f'{task_name}_Logger')

    return logger

class TqdmLoggingHandler(logging.StreamHandler):

    def __init__(self, level = logging.NOTSET):
        logging.StreamHandler.__init__(self)

    def emit(self, record):

        msg = self.format(record)
        tqdm.write(msg)
        self.flush()

class TqdmLogger():

    def __init__(self, name: str, res_dir: str) -> None:

        self.name = name
        current_time = datetime.datetime.now()
        time_str = current_time.strftime('%Y-%m-%d %H-%M-%S')
        self.log_dir = os.path.join(res_dir, time_str)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        self.logger = get_logger(f'{self.name}_Logger', self.log_dir)

    def info(self, log_str):
        
        self.logger.info(log_str)