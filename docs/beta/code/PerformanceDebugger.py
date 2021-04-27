#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# "Debugging Performance Issues" - a chapter of "The Debugging Book"
# Web site: https://www.debuggingbook.org/html/PerformanceDebugger.html
# Last change: 2021-04-13 11:12:29+02:00
#
# Copyright (c) 2021 CISPA Helmholtz Center for Information Security
# Copyright (c) 2018-2020 Saarland University, authors, and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

r'''
The Debugging Book - Debugging Performance Issues

This file can be _executed_ as a script, running all experiments:

    $ python PerformanceDebugger.py

or _imported_ as a package, providing classes, functions, and constants:

    >>> from debuggingbook.PerformanceDebugger import <identifier>
    
but before you do so, _read_ it and _interact_ with it at:

    https://www.debuggingbook.org/html/PerformanceDebugger.html

_For those only interested in using the code in this chapter (without wanting to know how it works), give an example.  This will be copied to the beginning of the chapter (before the first section) as text with rendered input and output._


For more details, source, and documentation, see
"The Debugging Book - Debugging Performance Issues"
at https://www.debuggingbook.org/html/PerformanceDebugger.html
'''


# Allow to use 'from . import <module>' when run as script (cf. PEP 366)
if __name__ == '__main__' and __package__ is None:
    __package__ = 'debuggingbook'


# Debugging Performance Issues
# ============================

if __name__ == '__main__':
    print('# Debugging Performance Issues')



if __name__ == '__main__':
    from .bookutils import YouTubeVideo
    YouTubeVideo("w4u5gCgPlmg")

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



if __name__ == '__main__':
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)

from . import Intro_Debugging

## Some Long-Running Function
## --------------------------

if __name__ == '__main__':
    print('\n## Some Long-Running Function')



from .ChangeCounter import ChangeCounter, debuggingbook_change_counter

## Simple Profiling
## ----------------

if __name__ == '__main__':
    print('\n## Simple Profiling')



import cProfile

## Alternative: Use a Tracer
## -------------------------

if __name__ == '__main__':
    print('\n## Alternative: Use a Tracer')



from typing import Any, Optional, Type, Union, Dict, Tuple, List

from .Intro_Debugging import remove_html_markup

from .bookutils import print_content

import inspect

if __name__ == '__main__':
    print_content(inspect.getsource(remove_html_markup), '.py',
                  start_line_number=238)

from . import Timer

from types import FrameType

from .Tracer import Tracer

class PerformanceTracer(Tracer):
    def __init__(self) -> None:
        super().__init__()
        self.reset_timer()
        self.hits: Dict[Tuple[str, int], int] = {}
        self.time: Dict[Tuple[str, int], float] = {}

    def reset_timer(self) -> None:
        self.timer = Timer.Timer()

    def __enter__(self) -> Any:
        super().__enter__()
        self.reset_timer()
        return self

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        t = self.timer.elapsed_time()
        key = (frame.f_code.co_name, frame.f_lineno)

        self.hits.setdefault(key, 0)
        self.time.setdefault(key, 0.0)
        self.hits[key] += 1
        self.time[key] += t

        self.reset_timer()

if __name__ == '__main__':
    with PerformanceTracer() as perf_tracer:
        for i in range(10000):
            s = remove_html_markup('<b>foo</b>')

if __name__ == '__main__':
    perf_tracer.hits

if __name__ == '__main__':
    perf_tracer.time

import inspect

from .bookutils import print_content

## Collect
## -------

if __name__ == '__main__':
    print('\n## Collect')



from .StatisticalDebugger import CoverageCollector, SpectrumDebugger

class MetricCollector(CoverageCollector):
    def metric(self, event: Any) -> Optional[float]:
        return None

    def all_metrics(self, func: str) -> List[float]:
        return []

    def total(self, func: str) -> float:
        return sum(self.all_metrics(func))

    def maximum(self, func: str) -> float:
        return max(self.all_metrics(func))

Location = Tuple[str, int]

class TimeCollector(MetricCollector):
    def __init__(self) -> None:
        super().__init__()
        self.reset_timer()
        self.time: Dict[Location, float] = {}
        self.add_items_to_ignore([Timer.Timer, Timer.clock])

    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        t = self.timer.elapsed_time()
        super().collect(frame, event, arg)
        location = (frame.f_code.co_name, frame.f_lineno)

        self.time.setdefault(location, 0.0)
        self.time[location] += t

        self.reset_timer()

    def reset_timer(self) -> None:
        self.timer = Timer.Timer()

    def __enter__(self) -> Any:
        super().__enter__()
        self.reset_timer()
        return self

    def metric(self, location: Any) -> Optional[float]:
        if location in self.time:
            return self.time[location]
        else:
            return None
        
    def all_metrics(self, func: str) -> List[float]:
        return [time
                for (func_name, lineno), time in self.time.items()
                if func_name == func]

if __name__ == '__main__':
    with TimeCollector() as collector:
        for i in range(100):
            s = remove_html_markup('<b>foo</b>')

if __name__ == '__main__':
    for location, time in collector.time.items():
        print(location, time)

if __name__ == '__main__':
    collector.total('remove_html_markup')

## Visualize
## ---------

if __name__ == '__main__':
    print('\n## Visualize')



class MetricDebugger(SpectrumDebugger):
    def metric(self, location: Location) -> float:
        sum = 0.0
        for outcome in self.collectors:
            for collector in self.collectors[outcome]:
                assert isinstance(collector, MetricCollector)
                m = collector.metric(location)
                if m is not None:
                    sum += m  # type: ignore

        return sum
    
    def total(self, func_name: str) -> float:
        total = 0.0
        for outcome in self.collectors:
            for collector in self.collectors[outcome]:
                assert isinstance(collector, MetricCollector)
                total += sum(collector.all_metrics(func_name))

        return total
    
    def maximum(self, func_name: str) -> float:
        maximum = 0.0
        for outcome in self.collectors:
            for collector in self.collectors[outcome]:
                assert isinstance(collector, MetricCollector)
                maximum = max(maximum, 
                              max(collector.all_metrics(func_name)))

        return maximum
    
    def suspiciousness(self, location: Location) -> float:
        func_name, _ = location
        return self.metric(location) / self.total(func_name)
    
    def color(self, location: Location) -> str:
        func_name, _ = location
        hue = 240  # blue
        saturation = 100  # fully saturated
        darkness = self.metric(location) / self.maximum(func_name)
        lightness = 100 - darkness * 25
        return f"hsl({hue}, {saturation}%, {lightness}%)"
    
    def tooltip(self, location: Location) -> str:
        return f"{super().tooltip(location)} {self.metric(location)}"

class PerformanceDebugger(MetricDebugger):
    def __init__(self, collector_class: Type, log: bool = False):
        assert issubclass(collector_class, MetricCollector)
        super().__init__(collector_class, log=log)

if __name__ == '__main__':
    with PerformanceDebugger(TimeCollector) as debugger:
        for i in range(100):
            s = remove_html_markup('<b>foo</b>')

if __name__ == '__main__':
    print(debugger)

if __name__ == '__main__':
    debugger

## Other Metrics
## -------------

if __name__ == '__main__':
    print('\n## Other Metrics')



class HitCollector(MetricCollector):
    def __init__(self) -> None:
        super().__init__()
        self.hits: Dict[Location, int] = {}

    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        super().collect(frame, event, arg)
        location = (frame.f_code.co_name, frame.f_lineno)

        self.hits.setdefault(location, 0)
        self.hits[location] += 1
    
    def metric(self, location: Location) -> Optional[int]:
        if location in self.hits:
            return self.hits[location]
        else:
            return None
        
    def all_metrics(self, func: str) -> List[float]:
        return [hits
                for (func_name, lineno), hits in self.hits.items()
                if func_name == func]

if __name__ == '__main__':
    with PerformanceDebugger(HitCollector) as debugger:
        for i in range(100):
            s = remove_html_markup('<b>foo</b>')

if __name__ == '__main__':
    debugger.total('remove_html_markup')

if __name__ == '__main__':
    print(debugger)

if __name__ == '__main__':
    debugger

## _Section 1_
## -----------

if __name__ == '__main__':
    print('\n## _Section 1_')



## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



## Lessons Learned
## ---------------

if __name__ == '__main__':
    print('\n## Lessons Learned')



## Next Steps
## ----------

if __name__ == '__main__':
    print('\n## Next Steps')



## Background
## ----------

if __name__ == '__main__':
    print('\n## Background')



## Exercises
## ---------

if __name__ == '__main__':
    print('\n## Exercises')



### Exercise 1: _Title_

if __name__ == '__main__':
    print('\n### Exercise 1: _Title_')



if __name__ == '__main__':
    pass

if __name__ == '__main__':
    2 + 2

### Exercise 2: _Title_

if __name__ == '__main__':
    print('\n### Exercise 2: _Title_')


