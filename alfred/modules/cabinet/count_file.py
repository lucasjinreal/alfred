"""

count how many certain files under dir

"""
import os
import glob
from alfred.utils.log import logger as logging



def count_file(d, f_type):
    assert os.path.exists(d), '{} not exist.'.format(d)
    # f_type can be jpg,png,pdf etc, connected by comma
    all_types = f_type.split(',')
    logging.info('count all file types: {} under: {}'.format(all_types, d))
    all_files = []
    for t in all_types:
        t = t.replace('.', '')
        one = glob.glob(os.path.join(d, '*.{}'.format(t)))
        one = [i for i in one if os.path.isfile(i)]
        logging.info('{} num: {}'.format(t, len(one)))
        all_files.extend(one)
    logging.info('file types: {}, total num: {}'.format(all_types, len(all_files)))
