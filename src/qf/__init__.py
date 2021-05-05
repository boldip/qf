import logging

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(funcName)s, line %(lineno)4d]: %(message)s', 
        datefmt='%d/%m/%Y %H:%M:%S')

__version__ = "1.0"
