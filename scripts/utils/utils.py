from pathlib import Path
import yaml
import logging

def parse_config(config_file:yaml):
    '''
    Reads a yaml file [yaml] and returns a dictionary with contents [dict]
    '''
    with open(config_file, 'rb') as f:
        config = yaml.safe_load(f)
    return config



def configure_logger(logger_name, log_path):
    '''
    Arguments:
        - log_path [str] - where the log should be saved (e.g. "log/train.log")
    
    Outputs:
        - a configured logger [logger]
    '''
    log_path = Path(log_path)
    #in case user wants so more hierarchy levels in the log directory -> create the neccesary folders
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # __name__ ensures each script will have a separate logger
    logger = logging.getLogger(logger_name)
    # DEBUG is the lowest logging level -> it will log everything
    logger.setLevel(logging.DEBUG) 
    # Set the file handler, formatter and finally the logger
    file_handler = logging.FileHandler(log_path, mode='w')
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

