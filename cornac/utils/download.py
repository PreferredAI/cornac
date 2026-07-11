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
import sys
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


def _safe_extract(archive, extract_path, extracted):
    """Extract archive members one at a time, allowing only regular files and directories.
    Blocks traversal paths, symlinks, hardlinks, FIFOs, and device files.
    Appends each successfully written path to `extracted` for scoped cleanup on failure.
    """
    extract_path = os.path.realpath(extract_path)
    members = archive.getmembers() if hasattr(archive, "getmembers") else archive.infolist()
    for member in members:
        member_name = member.name if hasattr(member, "name") else member.filename
        # Whitelist: allow only regular files and directories — blocks symlinks, hardlinks,
        # FIFOs, character devices, block devices, and any other special TAR types
        if hasattr(member, "isreg") and not (member.isreg() or member.isdir()):
            raise ValueError(f"Blocked special member type in archive: {member_name}")
        target = os.path.realpath(os.path.join(extract_path, member_name))
        # Skip the root directory entry — extracting it can apply tar-owned permissions
        if target == extract_path:
            continue
        if not target.startswith(extract_path + os.sep):
            raise ValueError(f"Blocked path traversal attempt in archive: {member_name}")
        # Record the target before writing so a partially-written file (failed mid-extract)
        # is still cleaned up. Extract one member at a time so each write is validated
        # against the live filesystem. Use filter="data" on Python 3.12+ as a defence-in-depth
        # layer (safer than fully_trusted).
        extracted.append(target)
        if isinstance(archive, tarfile.TarFile) and sys.version_info >= (3, 12):
            archive.extract(member, extract_path, filter="data")
        else:
            archive.extract(member, extract_path)


def _extract_archive(file_path, extract_path="."):
    """Extracts an archive."""
    for archive_type in ["zip", "tar"]:
        if archive_type == "zip":
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile
        elif archive_type == "tar":
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                extracted = []
                try:
                    _safe_extract(archive, extract_path, extracted)
                except BaseException:
                    # Any failure (corrupt archive, blocked member, interrupt, decompression
                    # error, etc.) triggers cleanup, then the original exception is re-raised.
                    # Remove only files written during this extraction — never wipe the full cache
                    for path in reversed(extracted):
                        try:
                            if os.path.isfile(path) or os.path.islink(path):
                                os.remove(path)
                            elif os.path.isdir(path) and not os.listdir(path):
                                os.rmdir(path)
                        except OSError:
                            pass
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
