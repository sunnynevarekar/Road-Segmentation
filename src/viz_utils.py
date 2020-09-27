import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def print_table(*arrays, header):
    #print header
    assert len(arrays) == len(header)
    num_rows = len(arrays[0])
    for column in header:
        print(f'{column:<20}', end='')
    print()
    
    
    for j in range(num_rows):
        for i in range(len(header)):
            ele = arrays[i][j]
            if isinstance(ele, str) or isinstance(ele, int):
                print(f'{ele:<20}', end='')
            else:
                print(f'{ele:<20.4f}', end='')
        print()    
    
    
def print_model_summary(model, logger=None):
    col_lenth = 130
    if logger:
        print_fn = logger.log
    else:
        print_fn = print
       
        
    print_fn('_'*col_lenth)
    print_fn('{:60s}{:40s}{:10s}{:>20s}'.format('Layer', 'Size', 'Parameters', 'Trainable'))
    print_fn('_'*col_lenth)
    total_params = 0
    for name, parameter in model.named_parameters():
        mult = 1
        for p in parameter.size():
            mult *= p
        print_fn(f'{name[:50]:60}{str(parameter.size()):40s}{mult:10d}{str(parameter.requires_grad):>20s}')
        print_fn('-'*col_lenth)
        total_params += mult

    print_fn()    
    print_fn('{:100s}{:10d}'.format('Total parameters', total_params))
    print_fn('_'*col_lenth)    