import logging
import os
from src.utils.common import read_config
def logging_dir(config):
    log_dir = config["logs"]["logs_dir"]
    general_log_dir = config["logs"]["general_logs"]
    log_dir_path = os.path.join(log_dir, general_log_dir)
    os.makedirs(log_dir_path, exist_ok=True)
    log_name = config["logs"]["log_name"]
    path_to_log = os.path.join(log_dir_path, log_name)
    logging.basicConfig(filename=path_to_log, filemode='w', format='%(name)s - %(levelname)s - %(message)s',level=logging.INFO)
    logging.warning('This will get logged to a file')
    return log_dir