#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# -*- coding: utf-8 -*-

from time import perf_counter
from typing import List, NamedTuple, Optional, Union
import time
from collections import namedtuple
from abc import ABC
import tabulate


class Timer(object):
    """A simple timer."""

    def __init__(self, name: str):
        self.name = name
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.perf_counter()

    def toc(self, average=True):
        self.diff = time.perf_counter() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.


class TimerManager(ABC):

    def __init__(self, timer_names: Union[List[str], str]):
        self.timer_names = timer_names
        self.tpls = []

        if isinstance(self.timer_names, list):
            for tn in self.timer_names:
                setattr(self, tn, Timer(tn))
                self.tpls.append(tn)

    def collect_avg(self):
        for tn in self.timer_names:
            obj = getattr(self, tn)
            print(f'{obj.name}: {obj.average_time*1000:.3f}ms')
        print('')


'''

timers = create_some_timers(['det', 'mesh', 'reg'])

timers.det.tic()
...
timers.det.toc()


timers.mesh.tic()
...
timers.mesh.toc()


timers.collect_avg()

'''


class ATimer:
    records = {}
    tmp = None

    @classmethod
    def tic(cls):
        cls.tmp = time.time()

    @classmethod
    def toc(cls):
        res = (time.time() - cls.tmp) * 1000
        cls.tmp = None
        return res

    @classmethod
    def report(cls):
        header = ['', 'Time(ms)']
        contents = []
        for key, val in cls.records.items():
            contents.append(
                ['{:20s}'.format(key), '{:.2f}'.format(sum(val)/len(val))])
        print(tabulate.tabulate(contents, header, tablefmt='fancy_grid'))

    def __init__(self, name, silent=False):
        self.name = name
        self.silent = silent
        if name not in ATimer.records.keys():
            ATimer.records[name] = []

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        end = time.time()
        ATimer.records[self.name].append((end-self.start)*1000)
        if not self.silent:
            t = (end - self.start)*1000
            if t > 1000:
                print('-> [{:20s}]: {:5.1f}s'.format(self.name, t/1000))
            elif t > 1e3*60*60:
                print('-> [{:20s}]: {:5.1f}min'.format(self.name, t/1e3/60))
            else:
                print('-> [{:20s}]: {:5.1f}ms'.format(self.name,
                      (end-self.start)*1000))
