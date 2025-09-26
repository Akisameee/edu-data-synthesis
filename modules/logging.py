import os
from tqdm import tqdm
import logging
import datetime

def get_logger(task_name: str, log_dir: str):
    
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(f'{task_name}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    logger.handlers.clear()
    file_handler = logging.FileHandler(os.path.join(log_dir, 'logger.log'), mode="a", encoding="utf-8")
    tqdm_handler = TqdmLoggingHandler()
    
    console_fmt = "(%(levelname)s) %(asctime)s - %(name)s:\n%(message)s"
    file_fmt = "(%(levelname)s) %(asctime)s  - %(name)s:\n%(message)s"
    
    console_formatter = logging.Formatter(fmt=console_fmt)
    file_formatter = logging.Formatter(fmt=file_fmt)
    
    tqdm_handler.setLevel(logging.INFO)
    tqdm_handler.setFormatter(fmt=console_formatter)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt=file_formatter)
    
    logger.addHandler(tqdm_handler)
    logger.addHandler(file_handler)
    
    return logger

class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self, level=logging.NOTSET):
        super().__init__()
        self.setLevel(level)

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
        os.makedirs(self.log_dir, exist_ok = True)

        self.logger = get_logger(f'{self.name}_logger', self.log_dir)

    def info(self, log_str):
        self.logger.info(log_str)