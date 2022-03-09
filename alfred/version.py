# Copyright (c) Lucas Jin. All rights reserved.

<<<<<<< HEAD
__version__ = '2.10.1'
=======
__version__ = '2.10.4'
>>>>>>> 024a75357a3836c4de031658f005628a03f16978
short_version = __version__


def parse_version_info(version_str):
    version_info = []
    for x in version_str.split('.'):
        if x.isdigit():
            version_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            version_info.append(int(patch_version[0]))
            version_info.append(f'rc{patch_version[1]}')
    return tuple(version_info)


version_info = parse_version_info(__version__)
