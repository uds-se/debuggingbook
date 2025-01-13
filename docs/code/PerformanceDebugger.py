#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# "Debugging Performance Issues" - a chapter of "The Debugging Book"
# Web site: https://www.debuggingbook.org/html/PerformanceDebugger.html
# Last change: 2025-01-13 16:01:43+01:00
#
# Copyright (c) 2021-2025 CISPA Helmholtz Center for Information Security
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

This chapter provides a class `PerformanceDebugger` that allows measuring and visualizing the time taken per line in a function.

>>> with PerformanceDebugger(TimeCollector) as debugger:
>>>     for i in range(100):
>>>         s = remove_html_markup('foo')

The distribution of executed time within each function can be obtained by printing out the debugger:

>>> print(debugger)
 238   3% def remove_html_markup(s):  # type: ignore
 239   2%     tag = False
 240   2%     quote = False
 241   2%     out = ""
 242   0%
 243  17%     for c in s:
 244  14%         assert tag or not quote
 245   0%
 246  14%         if c == '<' and not quote:
 247   2%             tag = True
 248  11%         elif c == '>' and not quote:
 249   2%             tag = False
 250   8%         elif (c == '"' or c == "'") and tag:
 251   0%             quote = not quote
 252   8%         elif not tag:
 253   4%             out = out + c
 254   0%
 255   2%     return out



The sum of all percentages in a function should always be 100%.

These percentages can also be visualized, where darker shades represent higher percentage values:

>>> debugger

 238 def remove_html_markup(s):  # type: ignore
 239     tag = False
 240     quote = False
 241     out = ""
 242  
 243     for c in s:
 244         assert tag or not quote
 245  
 246         if c == '<' and not quote:
 247             tag = True
 248         elif c == '>' and not quote:
 249             tag = False
 250         elif (c == '"' or c == "'") and tag:
 251             quote = not quote
 252         elif not tag:
 253             out = out + c
 254  
 255     return out


The abstract `MetricCollector` class allows subclassing to build more collectors, such as `HitCollector`.

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
    YouTubeVideo("0tMeB9G0uUI")

if __name__ == '__main__':
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)

from . import StatisticalDebugger
from . import DeltaDebugger

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



## Measuring Performance
## ---------------------

if __name__ == '__main__':
    print('\n## Measuring Performance')



### Tracing Execution Profiles

if __name__ == '__main__':
    print('\n### Tracing Execution Profiles')



from .ChangeCounter import ChangeCounter, debuggingbook_change_counter  # minor dependency

from . import Timer

if __name__ == '__main__':
    with Timer.Timer() as t:
        change_counter = debuggingbook_change_counter(ChangeCounter)

if __name__ == '__main__':
    t.elapsed_time()

import cProfile

if __name__ == '__main__':
    cProfile.run('debuggingbook_change_counter(ChangeCounter)', sort='cumulative')

### Sampling Execution Profiles

if __name__ == '__main__':
    print('\n### Sampling Execution Profiles')



## Improving Performance
## ---------------------

if __name__ == '__main__':
    print('\n## Improving Performance')



from .bookutils import quiz

if __name__ == '__main__':
    quiz('Donald E. Knuth said: "Premature optimization..."',
        [
            "... is the root of all evil",
            "... requires lots of experience",
            "... should be left to assembly programmers",
            "... is the reason why TeX is so fast",
        ], 'len("METAFONT") - len("TeX") - len("CWEB")')

## Building a Profiler
## -------------------

if __name__ == '__main__':
    print('\n## Building a Profiler')



from .Intro_Debugging import remove_html_markup

from typing import Any, Optional, Type, Dict, Tuple, List

from .bookutils import print_content

import inspect

if __name__ == '__main__':
    print_content(inspect.getsource(remove_html_markup), '.py',
                  start_line_number=238)

from .Tracer import Tracer

Location = Tuple[str, int]

class PerformanceTracer(Tracer):
    """Trace time and #hits for individual program lines"""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        self.reset_timer()
        self.hits: Dict[Location, int] = {}
        self.time: Dict[Location, float] = {}

    def reset_timer(self) -> None:
        self.timer = Timer.Timer()

from types import FrameType

class PerformanceTracer(PerformanceTracer):
    def __enter__(self) -> Any:
        """Enter a `with` block."""
        super().__enter__()
        self.reset_timer()
        return self

class PerformanceTracer(PerformanceTracer):
    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """Tracing function; called for every line."""
        t = self.timer.elapsed_time()
        location = (frame.f_code.co_name, frame.f_lineno)

        self.hits.setdefault(location, 0)
        self.time.setdefault(location, 0.0)
        self.hits[location] += 1
        self.time[location] += t

        self.reset_timer()

if __name__ == '__main__':
    with PerformanceTracer() as perf_tracer:
        for i in range(10000):
            s = remove_html_markup('<b>foo</b>')

if __name__ == '__main__':
    perf_tracer.hits

if __name__ == '__main__':
    perf_tracer.time

## Visualizing Performance Metrics
## -------------------------------

if __name__ == '__main__':
    print('\n## Visualizing Performance Metrics')



### Collecting Time Spent

if __name__ == '__main__':
    print('\n### Collecting Time Spent')



from .StatisticalDebugger import CoverageCollector, SpectrumDebugger

class MetricCollector(CoverageCollector):
    """Abstract superclass for collecting line-specific metrics"""

    def metric(self, event: Any) -> Optional[float]:
        """Return a metric for an event, or none."""
        return None

    def all_metrics(self, func: str) -> List[float]:
        """Return all metric for a function `func`."""
        return []

class MetricCollector(MetricCollector):
    def total(self, func: str) -> float:
        return sum(self.all_metrics(func))

    def maximum(self, func: str) -> float:
        return max(self.all_metrics(func))

class TimeCollector(MetricCollector):
    """Collect time executed for each line"""

    def __init__(self) -> None:
        """Constructor"""
        super().__init__()
        self.reset_timer()
        self.time: Dict[Location, float] = {}
        self.add_items_to_ignore([Timer.Timer, Timer.clock])

    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        """Invoked for every line executed. Accumulate time spent."""
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

class TimeCollector(TimeCollector):
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
    for location, time_spent in collector.time.items():
        print(location, time_spent)

if __name__ == '__main__':
    collector.total('remove_html_markup')

### Visualizing Time Spent

if __name__ == '__main__':
    print('\n### Visualizing Time Spent')



class MetricDebugger(SpectrumDebugger):
    """Visualize a metric"""

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
    """Collect and visualize a metric"""

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

### Other Metrics

if __name__ == '__main__':
    print('\n### Other Metrics')



class HitCollector(MetricCollector):
    """Collect how often a line is executed"""

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

## Integrating with Delta Debugging
## --------------------------------

if __name__ == '__main__':
    print('\n## Integrating with Delta Debugging')



import time

def remove_html_markup_ampersand(s: str) -> str:
    tag = False
    quote = False
    out = ""

    for c in s:
        assert tag or not quote

        if c == '&':
            time.sleep(0.1)  # <-- the obvious performance issue

        if c == '<' and not quote:
            tag = True
        elif c == '>' and not quote:
            tag = False
        elif (c == '"' or c == "'") and tag:
            quote = not quote
        elif not tag:
            out = out + c

    return out

if __name__ == '__main__':
    with Timer.Timer() as t:
        remove_html_markup_ampersand('&&&')
    t.elapsed_time()

def remove_html_test(s: str) -> None:
    with Timer.Timer() as t:
        remove_html_markup_ampersand(s)
    assert t.elapsed_time() < 0.1

if __name__ == '__main__':
    s_fail = '<b>foo&amp;</b>'

if __name__ == '__main__':
    with DeltaDebugger.DeltaDebugger() as dd:
        remove_html_test(s_fail)

if __name__ == '__main__':
    dd.min_args()

if __name__ == '__main__':
    s_pass = dd.max_args()

if __name__ == '__main__':
    s_pass

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



if __name__ == '__main__':
    with PerformanceDebugger(TimeCollector) as debugger:
        for i in range(100):
            s = remove_html_markup('<b>foo</b>')

if __name__ == '__main__':
    print(debugger)

if __name__ == '__main__':
    debugger

from .ClassDiagram import display_class_hierarchy

if __name__ == '__main__':
    display_class_hierarchy([PerformanceDebugger, TimeCollector, HitCollector],
                            public_methods=[
                                PerformanceDebugger.__init__,
                            ],
                            project='debuggingbook')

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



### Exercise 1: Profiling Memory Usage

if __name__ == '__main__':
    print('\n### Exercise 1: Profiling Memory Usage')



import tracemalloc

if __name__ == '__main__':
    tracemalloc.start()

if __name__ == '__main__':
    current_size, peak_size = tracemalloc.get_traced_memory()
    current_size

if __name__ == '__main__':
    tracemalloc.stop()

## Exercise 2: Statistical Performance Debugging
## ---------------------------------------------

if __name__ == '__main__':
    print('\n## Exercise 2: Statistical Performance Debugging')


