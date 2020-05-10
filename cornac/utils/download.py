# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import shutil
import zipfile
import tarfile
from urllib import request

from tqdm.auto import tqdm


def _urlretrieve(url, fpath):
    """Retrieve data from given url

    Parameters
    ----------
    url: str
        The url to the data.

    fpath: str
        The path to file where data is stored.

    """
    opener = request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]

    with tqdm(unit="B", unit_scale=True) as progress:

        def report(chunk, chunksize, total):
            progress.total = total
            progress.update(chunksize)

        request.install_opener(opener)
        request.urlretrieve(url, fpath, reporthook=report)


def _extract_archive(file_path, extract_path="."):
    """Extracts an archive.
    """
    for archive_type in ["zip", "tar"]:
        if archive_type == "zip":
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile
        elif archive_type == "tar":
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(extract_path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(extract_path):
                        if os.path.isfile(extract_path):
                            os.remove(extract_path)
                        else:
                            shutil.rmtree(extract_path)
                    raise


def get_cache_path(relative_path, cache_dir=None):
    """Return the absolute path to the cached data file
    """
    if cache_dir is None and os.access(os.path.expanduser("~"), os.W_OK):
        cache_dir = os.path.join(os.path.expanduser("~"), ".cornac")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    if not os.access(cache_dir, os.W_OK):
        cache_dir = os.path.join("/tmp", ".cornac")
    cache_path = os.path.join(cache_dir, relative_path)

    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))

    return cache_path, cache_dir


def cache(url, unzip=False, relative_path=None, cache_dir=None):
    """Download the data and cache to file

    Parameters
    ----------
    url: str
        The url to the data.

    unzip: bool, optional, default: False
        Whether the data is a zip file and going to be unzipped after the download.

    relative_path: str
        Relative path to the data file after finishing the download.
        If unzip=True, relative_path is the path to unzipped file.

    cache_dir: str, optional, default: None
        The path to cache folder. If `None`, either ~/.cornac or /tmp/.cornac will be used.

    """
    if relative_path is None:
        relative_path = url.split("/")[-1]
    cache_path, cache_dir = get_cache_path(relative_path, cache_dir)
    if os.path.exists(cache_path):
        return cache_path

    print("Data from", url)
    print("will be cached into", cache_path)

    if unzip:
        tmp_path = os.path.join(cache_dir, "file.tmp")
        _urlretrieve(url, tmp_path)
        print("Unzipping ...")
        _extract_archive(tmp_path, cache_dir)
        os.remove(tmp_path)
    else:
        _urlretrieve(url, cache_path)

    print("File cached!")
    return cache_path
