"""

split txt file with ratios

alfred cab split -f all.txt -r 0.1,0.8,0.1 -n train,val,test

"""
import os
import glob
from alfred.utils.log import logger as logging
import numpy as np


def split_txt_file(f, ratios, names):
    assert os.path.exists(f), '{} not exist.'.format(f)
    if not ratios:
        ratios = [0.2, 0.8]
    else:
        ratios = [float(i) for i in ratios.split(',')]
    logging.info('split ratios: {}'.format(ratios))

    if not names:
        names = ['part_{}'.format(i) for i in range(len(ratios))]
    else:
        names = names.split(',')
    names = [i+'.txt' for i in names]
    logging.info('split save to names: {}'.format(names))

    a = sum(ratios)
    if a != 1.:
        logging.info(
            'ratios: {} does not sum to 1. you must change it first.'.format(ratios))
        exit(1)

    # read txt file
    with open(f, 'r') as f:
        lines = f.readlines()
        logging.info('to split file have all {} lines.'.format(len(lines)))
        # split with ratios
        last_lines = 0
        for i, r in enumerate(ratios):
            one = lines[last_lines: last_lines+int(r * len(lines))]
            with open(names[i], 'w') as ff:
                ff.writelines(one)
                logging.info('Part {} saved into: {}. portion: {}/{}={}'.format(
                    i, names[i], len(one), len(lines), len(one)/(len(lines))))
            last_lines += len(one)
    logging.info('split done.')
