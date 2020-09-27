import os
from datetime import datetime
import logging

def initialize(logdir='logs', prefix=None):
    """Function to create and initialize logger. Create log directory if not present."""
    
    #timestamp for file
    strft = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    if prefix:
        filename = f'{prefix}_run_{strft}.log'
    else:
        filename = f'run_{strft}.log'

    logging.basicConfig(filename=os.path.join(logdir, filename),filemode='w', 
                        format='[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    
    
def log(message):
    logging.info(message)
    print(message)