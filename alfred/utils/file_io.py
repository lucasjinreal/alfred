#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import errno
import logging
import os
import glob
import shutil
from collections import OrderedDict
from typing import IO, Any, Dict, List, MutableMapping, Optional
from urllib.parse import urlparse
import portalocker
import logging
import shutil
from typing import Callable, Optional
from urllib import request


__all__ = ["PathManager", "get_cache_dir", "file_lock"]


def download(
    url: str, dir: str, *, filename: Optional[str] = None, progress: bool = True
):
    """
    Download a file from a given URL to a directory. If file exists, will not
        overwrite the existing file.

    Args:
        url (str):
        dir (str): the directory to download the file
        filename (str or None): the basename to save the file.
            Will use the name in the URL if not given.
        progress (bool): whether to use tqdm to draw a progress bar.

    Returns:
        str: the path to the downloaded file or the existing one.
    """
    os.makedirs(dir, exist_ok=True)
    if filename is None:
        filename = url.split("/")[-1]
        assert len(filename), "Cannot obtain filename from url {}".format(url)
    fpath = os.path.join(dir, filename)
    logger = logging.getLogger(__name__)

    if os.path.isfile(fpath):
        logger.info("File {} exists! Skipping download.".format(filename))
        return fpath

    tmp = fpath + ".tmp"  # download to a tmp file first, to be more atomic.
    try:
        logger.info("Downloading from {} ...".format(url))
        if progress:
            import tqdm

            # todo: change tqdm to rich for download progress

            def hook(t: tqdm.tqdm) -> Callable[[int, int, Optional[int]], None]:
                last_b = [0]

                def inner(b: int, bsize: int, tsize: Optional[int] = None) -> None:
                    if tsize is not None:
                        t.total = tsize
                    t.update((b - last_b[0]) * bsize)  # type: ignore
                    last_b[0] = b

                return inner

            with tqdm.tqdm(  # type: ignore
                unit="B", unit_scale=True, miniters=1, desc=filename, leave=True
            ) as t:
                tmp, _ = request.urlretrieve(url, filename=tmp, reporthook=hook(t))

        else:
            tmp, _ = request.urlretrieve(url, filename=tmp)
        statinfo = os.stat(tmp)
        size = statinfo.st_size
        if size == 0:
            raise IOError("Downloaded an empty file from {}!".format(url))
        # download to tmp first and move to fpath, to make this function more
        # atomic.
        shutil.move(tmp, fpath)
    except IOError:
        logger.error("Failed to download {}".format(url))
        raise
    finally:
        try:
            os.unlink(tmp)
        except IOError:
            pass

    logger.info("Successfully downloaded " + fpath + ". " + str(size) + " bytes.")
    return fpath


def get_cache_dir(cache_dir: Optional[str] = None) -> str:
    """
    Returns a default directory to cache static files
    (usually downloaded from Internet), if None is provided.

    Args:
        cache_dir (None or str): if not None, will be returned as is.
            If None, returns the default cache directory as:

        1) $DL_LIB_CACHE, if set
        2) otherwise ~/.torch/dl_lib_cache
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(
            os.getenv("DL_LIB_CACHE", "~/.torch/dl_lib_cache")
        )
    return cache_dir


def file_lock(path: str):  # type: ignore
    """
    A file lock. Once entered, it is guaranteed that no one else holds the
    same lock. Others trying to enter the lock will block for 30 minutes and
    raise an exception.

    This is useful to make sure workers don't cache files to the same location.

    Args:
        path (str): a path to be locked. This function will create a lock named
            `path + ".lock"`

    Examples:

        filename = "/path/to/file"
        with file_lock(filename):
            if not os.path.isfile(filename):
                do_create_file()
    """
    dirname = os.path.dirname(path)
    try:
        os.makedirs(dirname, exist_ok=True)
    except OSError:
        # makedir is not atomic. Exceptions can happen when multiple workers try
        # to create the same dir, despite exist_ok=True.
        # When this happens, we assume the dir is created and proceed to creating
        # the lock. If failed to create the directory, the next line will raise
        # exceptions.
        pass
    return portalocker.Lock(path + ".lock", timeout=1800)  # type: ignore


class PathHandler:
    """
    PathHandler is a base class that defines common I/O functionality for a URI
    protocol. It routes I/O for a generic URI which may look like "protocol://*"
    or a canonical filepath "/foo/bar/baz".
    """

    def _get_supported_prefixes(self) -> List[str]:
        """
        Returns:
            List[str]: the list of URI prefixes this PathHandler can support
        """
        raise NotImplementedError()

    def _get_local_path(self, path: str) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk. In this case, this function is meant to be
        used with read-only resources.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            local_path (str): a file path which exists on the local file system
        """
        raise NotImplementedError()

    def _open(self, path: str, mode: str = "r") -> IO[Any]:
        """
        Open a stream to a URI, similar to the built-in `open`.

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.

        Returns:
            file: a file-like object.
        """
        raise NotImplementedError()

    def _copy(self, src_path: str, dst_path: str, overwrite: bool = False) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file

        Returns:
            status (bool): True on success
        """
        raise NotImplementedError()

    def _exists(self, path: str) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        raise NotImplementedError()

    def _isfile(self, path: str) -> bool:
        """
        Checks if the resource at the given URI is a file.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a file
        """
        raise NotImplementedError()

    def _isdir(self, path: str) -> bool:
        """
        Checks if the resource at the given URI is a directory.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a directory
        """
        raise NotImplementedError()

    def _ls(self, path: str) -> List[str]:
        """
        List the contents of the directory at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            List[str]: list of contents in given path
        """
        raise NotImplementedError()

    def _mkdirs(self, path: str) -> None:
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.

        Args:
            path (str): A URI supported by this PathHandler
        """
        raise NotImplementedError()

    def _rm(self, path: str) -> None:
        """
        Remove the file (not directory) at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler
        """
        raise NotImplementedError()


class NativePathHandler(PathHandler):
    """
    Handles paths that can be accessed using Python native system calls. This
    handler uses `open()` and `os.*` calls on the given path.
    """

    def _get_local_path(self, path: str) -> str:
        return path

    def _open(self, path: str, mode: str = "r") -> IO[Any]:
        return open(path, mode)

    def _copy(self, src_path: str, dst_path: str, overwrite: bool = False) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file

        Returns:
            status (bool): True on success
        """
        if os.path.exists(dst_path) and not overwrite:
            logger = logging.getLogger(__name__)
            logger.error("Destination file {} already exists.".format(dst_path))
            return False

        try:
            shutil.copyfile(src_path, dst_path)
            return True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error("Error in file copy - {}".format(str(e)))
            return False

    def _exists(self, path: str) -> bool:
        return os.path.exists(path)

    def _isfile(self, path: str) -> bool:
        return os.path.isfile(path)

    def _isdir(self, path: str) -> bool:
        return os.path.isdir(path)

    def _ls(self, path: str) -> List[str]:
        return os.listdir(path)

    def _mkdirs(self, path: str) -> None:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            # EEXIST it can still happen if multiple processes are creating the dir
            if e.errno != errno.EEXIST:
                raise

    def _rm(self, path: str) -> None:
        os.remove(path)


class HTTPURLHandler(PathHandler):
    """
    Download URLs and cache them to disk.
    """

    def __init__(self) -> None:
        self.cache_map: Dict[str, str] = {}

    def _get_supported_prefixes(self) -> List[str]:
        return ["http://", "https://", "ftp://"]

    def _get_local_path(self, path: str) -> str:
        """
        This implementation downloads the remote resource and caches it locally.
        The resource will only be downloaded if not previously requested.
        """
        if path not in self.cache_map or not os.path.exists(self.cache_map[path]):
            logger = logging.getLogger(__name__)
            parsed_url = urlparse(path)
            dirname = os.path.join(
                get_cache_dir(), os.path.dirname(parsed_url.path.lstrip("/"))
            )
            filename = path.split("/")[-1]
            cached = os.path.join(dirname, filename)
            with file_lock(cached):
                if not os.path.isfile(cached):
                    logger.info("Downloading {} ...".format(path))
                    cached = download(path, dirname, filename=filename)
            logger.info("URL {} cached in {}".format(path, cached))
            self.cache_map[path] = cached
        return self.cache_map[path]

    def _open(self, path: str, mode: str = "r") -> IO[Any]:
        assert mode in (
            "r",
            "rb",
        ), "{} does not support open with {} mode".format(self.__class__.__name__, mode)
        local_path = self._get_local_path(path)
        return open(local_path, mode)


class PathManager:
    """
    A class for users to open generic paths or translate generic paths to file names.
    """

    _PATH_HANDLERS: MutableMapping[str, PathHandler] = OrderedDict()
    _NATIVE_PATH_HANDLER = NativePathHandler()

    @staticmethod
    def __get_path_handler(path: str) -> PathHandler:
        """
        Finds a PathHandler that supports the given path. Falls back to the native
        PathHandler if no other handler is found.

        Args:
            path (str): URI path to resource

        Returns:
            handler (PathHandler)
        """
        for p in PathManager._PATH_HANDLERS.keys():
            if path.startswith(p):
                return PathManager._PATH_HANDLERS[p]
        return PathManager._NATIVE_PATH_HANDLER

    @staticmethod
    def open(path: str, mode: str = "r") -> IO[Any]:
        """
        Open a stream to a URI, similar to the built-in `open`.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            file: a file-like object.
        """
        return PathManager.__get_path_handler(path)._open(path, mode)

    @staticmethod
    def copy(src_path: str, dst_path: str, overwrite: bool = False) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file

        Returns:
            status (bool): True on success
        """

        # Copying across handlers is not supported.
        assert PathManager.__get_path_handler(
            src_path
        ) == PathManager.__get_path_handler(dst_path)
        return PathManager.__get_path_handler(src_path)._copy(
            src_path, dst_path, overwrite
        )

    @staticmethod
    def get_local_path(path: str) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            local_path (str): a file path which exists on the local file system
        """
        return PathManager.__get_path_handler(path)._get_local_path(path)

    @staticmethod
    def exists(path: str) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        return PathManager.__get_path_handler(path)._exists(path)

    @staticmethod
    def isfile(path: str) -> bool:
        """
        Checks if there the resource at the given URI is a file.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a file
        """
        return PathManager.__get_path_handler(path)._isfile(path)

    @staticmethod
    def isdir(path: str) -> bool:
        """
        Checks if the resource at the given URI is a directory.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a directory
        """
        return PathManager.__get_path_handler(path)._isdir(path)

    @staticmethod
    def ls(path: str) -> List[str]:
        """
        List the contents of the directory at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            List[str]: list of contents in given path
        """
        return PathManager.__get_path_handler(path)._ls(path)

    @staticmethod
    def mkdirs(path: str) -> None:
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.

        Args:
            path (str): A URI supported by this PathHandler
        """
        return PathManager.__get_path_handler(path)._mkdirs(path)

    @staticmethod
    def rm(path: str) -> None:
        """
        Remove the file (not directory) at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler
        """
        return PathManager.__get_path_handler(path)._rm(path)

    @staticmethod
    def register_handler(handler: PathHandler) -> None:
        """
        Register a path handler associated with `handler._get_supported_prefixes`
        URI prefixes.

        Args:
            handler (PathHandler)
        """
        assert isinstance(handler, PathHandler), handler
        for prefix in handler._get_supported_prefixes():
            assert prefix not in PathManager._PATH_HANDLERS
            PathManager._PATH_HANDLERS[prefix] = handler

        # Sort path handlers in reverse order so longer prefixes take priority,
        # eg: http://foo/bar before http://foo
        PathManager._PATH_HANDLERS = OrderedDict(
            sorted(
                PathManager._PATH_HANDLERS.items(),
                key=lambda t: t[0],
                reverse=True,
            )
        )


PathManager.register_handler(HTTPURLHandler())


class SourceIter:
    def __init__(self, src, exit_auto=True):
        self.src = src
        self.srcs = []
        self.crt_index = 0
        self.video_mode = False
        self.webcam_mode = False
        self.cap = None
        self.ok = True
        self.exit_auto = exit_auto

    def __len__(self):
        return len(self.src)

    def __next__(self):
        if self.video_mode or self.webcam_mode:
            assert (
                self.cap is not None
            ), "video mode on but cap is None. video open failed."
            ret, frame = self.cap.read()
            self.crt_index += 1
            if not ret:
                if self.exit_auto:
                    print("Seems iteration done. bye~")
                    exit(0)
                else:
                    self.ok = False
            else:
                return frame
        else:
            if self.crt_index < len(self.srcs):
                p = self.srcs[self.crt_index]
                self.crt_index += 1
                return p
            else:
                if self.exit_auto:
                    print("Seems iteration done. bye~")
                    exit(0)
                else:
                    self.ok = False
                # raise StopIteration


class ImageSourceIter(SourceIter):
    def __init__(self, src, exit_auto=True):
        super(ImageSourceIter, self).__init__(src, exit_auto)

        import cv2 as cv

        self._index_sources()
        self.is_written = False
        self.save_f = None
        assert len(self.srcs) > 0, "srcs indexed empty: {}".format(self.srcs)
        self.lens = len(self.srcs)
        if self.video_mode and not self.webcam_mode:
            self.is_save_video_called = False
            fourcc = cv.VideoWriter_fourcc(*"XVID")
            self.video_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH) + 0.5)
            self.video_frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
            self.video_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT) + 0.5)
            if self.video_mode:
                self.filename = os.path.basename(src).split(".")[0]
                self.save_f = os.path.join(
                    os.path.dirname(src), self.filename + "_result.mp4"
                )
                self.lens = self.video_frame_count
            else:
                os.makedirs("results", exist_ok=True)
                self.save_f = os.path.join("results/webcam_result.mp4")
            self.video_writter = cv.VideoWriter(
                self.save_f, fourcc, 25.0, (self.video_width, self.video_height)
            )

    def get_specific_frames(self, frame_indexes, verbose=True):
        """
        get specific frames by a series frames idxes
        CAUTION: this will drain your memory, better not use
        """
        frames = []
        for i in frame_indexes:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, i - 1)
            res, frame = self.cap.read()
            if verbose:
                print(f"\r{i}/{len(frame_indexes)}", end="", flush=True)
            if res:
                frames.append(frame)
            else:
                print("failed to get frame at index: ", i)
        return frames

    def get_specific_frame_at(self, frame_index):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_index - 1)
        res, frame = self.cap.read()
        if res:
            return frame
        else:
            print("get frame at failed: ", frame_index)
            return

    def get_new_video_writter(self, new_width, new_height, save_f=None):
        """
        for users want save a video with new width and height
        """
        fourcc = cv.VideoWriter_fourcc(*"XVID")
        video_writter = cv.VideoWriter(save_f, fourcc, 25.0, (new_width, new_height))
        return video_writter

    def _is_video(self, p):
        suffix = os.path.basename(p).split(".")[-1]
        if suffix.lower() in ["mp4", "avi", "flv", "wmv", "mpeg", "mov"]:
            return True
        else:
            return False

    def save_res_image_or_video_frame(self, res):
        if self.video_mode:
            self.is_save_video_called = True
            self.video_writter.write(res)
            if not self.is_written:
                self.is_written = True
        else:
            return NotImplementedError

    def _index_sources(self):
        from natsort import natsorted

        if str(self.src).isdigit():
            self.webcam_mode = True
            self.video_mode = True
            self.cap = cv.VideoCapture(int(self.src))
            # self.cap = cv.VideoCapture(0)
            self.srcs = [self.src]
        else:
            assert os.path.exists(self.src), f"{self.src} not exist."
            if os.path.isfile(self.src) and self._is_video(self.src):
                self.video_mode = True
                self.cap = cv.VideoCapture(self.src)
                self.srcs = [self.src]
            elif os.path.isfile(self.src):
                self.srcs = [self.src]
            elif os.path.isdir(self.src):
                for ext in ("*.bmp", "*.png", "*.jpg", "*.jpeg"):
                    self.srcs.extend(glob.glob(os.path.join(self.src, ext)))
                # sort srcs with natural order
                self.srcs = natsorted(self.srcs)
            else:
                TypeError("{} must be dir or file".format(self.src))

    def __del__(self) -> None:
        if self.video_mode:
            self.cap.release()
            if not self.webcam_mode:
                self.video_writter.release()
            if self.is_written and self.is_save_video_called:
                print("your wrote video result file should saved into: ", self.save_f)
            else:
                if self.save_f and os.path.exists(self.save_f):
                    # clean up remove saved file.
                    os.remove(self.save_f)


class ImageSourceIterAsync(SourceIter):
    """
    reading frames in threads if on video mode
    using queue to to contains readed frames
    """

    def __init__(self, src, exit_auto=True):
        super(ImageSourceIter, self).__init__(src, exit_auto)
        import cv2 as cv
        
        self._index_sources()
        self.is_written = False
        self.save_f = None
        assert len(self.srcs) > 0, "srcs indexed empty: {}".format(self.srcs)
        self.lens = len(self.srcs)
        if self.video_mode and not self.webcam_mode:
            self.is_save_video_called = False
            fourcc = cv.VideoWriter_fourcc(*"XVID")
            self.video_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH) + 0.5)
            self.video_frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
            self.video_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT) + 0.5)
            if self.video_mode:
                self.filename = os.path.basename(src).split(".")[0]
                self.save_f = os.path.join(
                    os.path.dirname(src), self.filename + "_result.mp4"
                )
            else:
                os.makedirs("results", exist_ok=True)
                self.save_f = os.path.join("results/webcam_result.mp4")
            self.video_writter = cv.VideoWriter(
                self.save_f, fourcc, 25.0, (self.video_width, self.video_height)
            )

    def get_specific_frames(self, frame_indexes, verbose=True):
        """
        get specific frames by a series frames idxes
        CAUTION: this will drain your memory, better not use
        """
        frames = []
        for i in frame_indexes:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, i - 1)
            res, frame = self.cap.read()
            if verbose:
                print(f"\r{i}/{len(frame_indexes)}", end="", flush=True)
            if res:
                frames.append(frame)
            else:
                print("failed to get frame at index: ", i)
        return frames

    def get_specific_frame_at(self, frame_index):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_index - 1)
        res, frame = self.cap.read()
        if res:
            return frame
        else:
            print("get frame at failed: ", frame_index)
            return

    def get_new_video_writter(self, new_width, new_height, save_f=None):
        """
        for users want save a video with new width and height
        """
        fourcc = cv.VideoWriter_fourcc(*"XVID")
        video_writter = cv.VideoWriter(save_f, fourcc, 25.0, (new_width, new_height))
        return video_writter

    def _is_video(self, p):
        suffix = os.path.basename(p).split(".")[-1]
        if suffix.lower() in ["mp4", "avi", "flv", "wmv", "mpeg", "mov"]:
            return True
        else:
            return False

    def save_res_image_or_video_frame(self, res):
        if self.video_mode:
            self.is_save_video_called = True
            self.video_writter.write(res)
            if not self.is_written:
                self.is_written = True
        else:
            return NotImplementedError

    def _index_sources(self):
        from natsort import natsorted

        if str(self.src).isdigit():
            self.webcam_mode = True
            self.video_mode = True
            self.cap = cv.VideoCapture(int(self.src))
            # self.cap = cv.VideoCapture(0)
            self.srcs = [self.src]
        else:
            assert os.path.exists(self.src), f"{self.src} not exist."
            if os.path.isfile(self.src) and self._is_video(self.src):
                self.video_mode = True
                self.cap = cv.VideoCapture(self.src)
                self.srcs = [self.src]
            elif os.path.isfile(self.src):
                self.srcs = [self.src]
            elif os.path.isdir(self.src):
                for ext in ("*.bmp", "*.png", "*.jpg", "*.jpeg"):
                    self.srcs.extend(glob.glob(os.path.join(self.src, ext)))
                # sort srcs with natural order
                self.srcs = natsorted(self.srcs)
            else:
                TypeError("{} must be dir or file".format(self.src))

    def __del__(self) -> None:
        if self.video_mode:
            self.cap.release()
            if not self.webcam_mode:
                self.video_writter.release()
            if self.is_written and self.is_save_video_called:
                print("your wrote video result file should saved into: ", self.save_f)
            else:
                if self.save_f and os.path.exists(self.save_f):
                    # clean up remove saved file.
                    os.remove(self.save_f)
