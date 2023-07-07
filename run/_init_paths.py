# ------------------------------------------------------------------------------
# Written by Miao
# ------------------------------------------------------------------------------

import os.path as osp
import sys


def add_path():

    this_dir = osp.dirname(__file__)
    lib_path = osp.join(this_dir, '..', 'lib')

    if this_dir not in sys.path:
        sys.path.insert(0, lib_path)


