"""Subcommand for fetching UndoDB recordings from Evergreen."""
import os
from typing import List, Optional
import subprocess
import tempfile
from urllib.request import urlopen
from shutil import copyfileobj
import tarfile

import evergreen
from evergreen import RetryingEvergreenApi
from buildscripts.resmokelib.plugin import Subcommand


def _is_jira_ticket(asset: str) -> bool:
    maybe_ticket = asset.upper()
    return maybe_ticket.startswith("BF-") or maybe_ticket.startswith("BFG-")


class Fetch(Subcommand):
    """Fetch UndoDB recordings from a given Evergreen task ID or BF ticket."""

    def __init__(self, asset: str):
        """Constructor."""
        if _is_jira_ticket(asset):
            self._ticket = asset
            self._task_id = None
        else:
            self._ticket = None
            self._task_id = asset

    def execute(self) -> None:
        """
        Work your magic.

        :return: None
        """
        if self._ticket:
            raise NotImplementedError("Fetching recordings from JIRA tickets not yet implemented")

        assert self._task_id

        evg = RetryingEvergreenApi.get_api(use_config_file=True)
        artifacts = evg.task_by_id(self._task_id).artifacts
        url = _find_undodb_artifact_url(artifacts)
        if not url:
            print(
                f"Evergreen task '{self._task_id}' does not have an UndoDB recordings archive attached to it"
            )
            return

        local_file = _download_archive(url)
        if not local_file:
            print(f"Failed to download archive from '{url}'")
            return

        _extract_archive(local_file)
        _cleanup(local_file)


def _find_undodb_artifact_url(artifacts: List[evergreen.task.Artifact]) -> Optional[str]:
    for artifact in artifacts:
        if artifact.name.startswith("UndoDB Recordings - Execution "):
            return artifact.url

    return None


def _download_archive(url: str) -> Optional[str]:
    """Download the archive from the given url.

    :return path to the downloaded archive, or None if nothing was downloaded
    """
    fname = os.path.basename(url)
    out_file = os.path.join(tempfile.gettempdir(), fname)
    if os.path.exists(out_file):
        print(f"Output file '{out_file}' exists, assuming that it's valid")
        return out_file
    try:
        print(f"Downloading to '{out_file}'")
        with urlopen(url) as fsrc, open(out_file, "wb") as fdst:
            copyfileobj(fsrc, fdst)  # type: ignore

    except Exception as ex:
        if ex is KeyboardInterrupt:
            print("Cancelled download")
        _cleanup(out_file)
        raise ex

    return out_file


def _extract_archive(fname: str):
    print(f"Extracting to '{os.getcwd()}'")
    with tarfile.open(fname) as tfile:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tfile)


def _cleanup(out_file: str):
    if os.path.isfile(out_file):
        print("Cleaning up temporary files")
        os.remove(out_file)
