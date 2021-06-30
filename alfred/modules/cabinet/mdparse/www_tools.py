"""
Some functions useful for the working with URLs and network.
"""

import requests
from typing import Optional
import re
import os
from mimetypes import guess_extension
from .string_tools import slugify


def is_url(url: str, allowed_url_prefixes=('http', 'ftp')) -> bool:
    """
    Check url for prefix match.
    """

    for prefix in set(allowed_url_prefixes):
        if url.startswith(prefix):
            return True

    return False


def download_from_url(url: str, timeout=None):
    """
    Download file from the URL.
    :param url: URL to download.
    :param timeout: timeout before fail.
    """

    try:
        response = requests.get(url, allow_redirects=True, timeout=timeout)
    except requests.exceptions.SSLError:
        print('Incorrect SSL certificate, trying to download without verifying...')
        response = requests.get(url, allow_redirects=True, verify=False,
                                timeout=timeout)

    if response.status_code != 200:
        raise OSError(str(response))

    return response


def get_filename_from_url(req: requests.Response) -> Optional[str]:
    """
    Get filename from url and, if not found, try to get from content-disposition.
    """

    if req.url.find('/'):
        result = req.url.rsplit('/', 1)[1]
    else:
        cd = req.headers.get('content-disposition')

        if cd is None:
            return None

        file_name = re.findall('filename=(.+)', cd)

        if len(file_name) == 0:
            return None

        result = file_name[0]

    f_name, f_ext = os.path.splitext(result)

    result = f'{slugify(f_name)}{guess_extension(req.headers["content-type"].partition(";")[0].strip())}' if not f_ext\
        else f'{slugify(f_name)}.{slugify(f_ext)}'

    return result
