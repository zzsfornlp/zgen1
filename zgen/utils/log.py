#

# about print and log

__all__ = [
    "Logger", "zlog", "zwarn", "zfatal", "zcheck",
]

from typing import Iterable
from io import FileIO
import platform, time, logging, sys
from .file import zopen

# the simple logger class
class Logger:
    _instance = None  # singleton instance
    _logger_heads = {
        "plain": "-- ", "time": "## ", "io": "== ", "result": ">> ", "report": "** ", "config": "CC ",
        "warn": "!! ", "error": "ER ", "fatal": "KI ",
        "debug": "DE ", "nope": "", "": "",
    }
    MAGIC_CODE = "?THE-NATURE-OF-HUMAN-IS?"  # REPEATER?

    @staticmethod
    def get_singleton_logger():
        assert Logger._instance is not None, "Not initialized!!"
        return Logger._instance

    @staticmethod
    def init(files: Iterable = (), level=0, use_magic_file=False, log_last=False):
        s = "LOG-%s-%s.txt" % (platform.uname().node, '-'.join(time.ctime().split()[-4:]))
        s = '-'.join(s.split(':'))      # ':' raise error in Win
        log_files = [s] if use_magic_file else []
        log_files.extend([f for f in files if (f is not None) and ((not isinstance(f, str)) or len(f)>0)])
        Logger._instance = Logger(log_files, level=level, log_last=log_last)
        # zlog("START!!", func="plain")

    # =====
    def __init__(self, log_files: Iterable, level=0, log_last=False):
        self.log_files = list(log_files)
        self.log_last = log_last
        self.fds = []
        self.cached_lines = []  # store them if log_last
        # the managing of open files (except outside handlers like stdio) is by this one
        for f in log_files:
            if isinstance(f, str):
                if not self.log_last:
                    one_fd = zopen(f, mode="w")
                else:
                    one_fd = None
            else:  # should be already a fd
                one_fd = f
            if one_fd:
                self.fds.append(one_fd)
        self.level = level

    def flush_cached_logs(self, clear=False):
        if self.log_last:  # write out logs!
            for f in self.log_files:
                zlog(f'Flush {len(self.cached_lines)} lines to {f}')
                if isinstance(f, str):
                    with zopen(f, mode='w') as fd:  # todo(+?): w or a?
                        for line in self.cached_lines:
                            fd.write(line)
            if clear:  # by default no clear!
                self.cached_lines.clear()
        # --

    def __del__(self):
        for f, fd in zip(self.log_files, self.fds):
            if isinstance(f, str):
                # print(f"Delete file-logger {f}", file=sys.stderr)
                fd.close()
        # --

    def __repr__(self):
        return f"Logger({self.log_files})"

    def set_level(self, level=0):
        if level != self.level:
            zlog(f"Change log level from {self.level} to {level}", func="config")
            self.level = level
        return self.level

    def log(self, s: object, func="", end="\n", flush=True, timed=False, level=0):
        if level >= self.level:
            head = Logger._logger_heads.get(func, func)
            if level!=0:  # special ones
                head += f"[L{level}] "
            if timed:
                ss = f"{head}[{'-'.join(time.ctime().split()[-4:])}] {s}"
            else:
                ss = f"{head}{s}"
            for f in self.fds:
                if (not self.log_last) or (f is sys.stderr):
                    # note: sometimes there are unprinted ones from yield's generators which will trigger error,
                    #  but this does not matter anyway ...
                    # if f.closed:
                    #     print(f"WARNING: {self.__dict__} {ss}")
                    print(ss, end=end, file=f, flush=flush)
            if self.log_last:
                self.cached_lines.append(ss+end)

    # =====
    # system's logging module

    cached_loggers = {}

    @staticmethod
    def get_sys_logger(name="main", level=logging.INFO, handler=sys.stderr,
                       formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
        if name not in Logger.cached_loggers:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter(formatter)
            stream_handler = logging.StreamHandler(handler)
            stream_handler.setLevel(level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
            Logger.cached_loggers[name] = logger
        return Logger.cached_loggers[name]

# =====
# shortcuts
def zlog(s: object, **kwargs):
    Logger.get_singleton_logger().log(s, **kwargs)

def zfatal(ss=""):
    zlog(ss, func="fatal", timed=True, level=1)
    raise RuntimeError(ss)

def zwarn(ss=""):
    zlog(ss, func="warn", timed=True, level=1)

def zcheck(v, s="", error=False):
    if not v:
        zwarn(s)
        if error:
            raise RuntimeError(error)
