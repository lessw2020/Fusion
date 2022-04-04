# downloads from URL with progress bar

from pathlib import Path, PurePath
import requests
import shutil
from tqdm.auto import tqdm
import os


def preview_file(file_url):
    """preview a file from a URL - verify access, existence and check size"""
    if isinstance(file_url, PurePath):
        file_url = str(file_url)

    # connect to URL
    with requests.get(file_url, stream=True) as req:
        status = req.status_code
        print(f"connection status = {req.status_code}")

        if status == 403:
            print(f"Object permissions are blocking from download (Forbidden)")
            return

        # check header to get content length, in bytes
        file_length = int(req.headers.get("Content-Length"))
        file_name = os.path.basename(req.url)
        file_size_in_MB = round(file_length / (1024 * 1024), 4)

        print(f"File is available, size of {file_name} is {file_size_in_MB} MB ")
        return


def get_file(file_url, download=True):
    """download a file with progress bar

    download = False - will check file size and verify access to file
    download = True - downloads file with progress bar"""

    if isinstance(file_url, PurePath):
        file_url = str(file_url)

    # http request
    with requests.get(file_url, stream=True) as req:
        status = req.status_code
        print(f"connection status = {req.status_code}")

        if status == 403:
            print(f"Object permissions are blocking from download (Forbidden)")
            return

        # check header to get content length, in bytes
        file_length = int(req.headers.get("Content-Length"))
        file_name = os.path.basename(req.url)
        file_size_in_MB = round(file_length / (1024 * 1024), 4)
        if not download:

            print(f"File size of {file_name} is {file_size_in_MB} MB ")
            return

        with tqdm.wrapattr(
            req.raw, "read", total=file_length, desc=f"Downloading {file_name} "
        ) as raw:

            with open(f"{file_name}", "wb") as output:
                shutil.copyfileobj(raw, output)
