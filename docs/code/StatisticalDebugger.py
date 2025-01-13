#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# "Statistical Debugging" - a chapter of "The Debugging Book"
# Web site: https://www.debuggingbook.org/html/StatisticalDebugger.html
# Last change: 2025-01-13 15:54:58+01:00
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
The Debugging Book - Statistical Debugging

This file can be _executed_ as a script, running all experiments:

    $ python StatisticalDebugger.py

or _imported_ as a package, providing classes, functions, and constants:

    >>> from debuggingbook.StatisticalDebugger import <identifier>

but before you do so, _read_ it and _interact_ with it at:

    https://www.debuggingbook.org/html/StatisticalDebugger.html

This chapter introduces classes and techniques for _statistical debugging_ – that is, correlating specific events, such as lines covered, with passing and failing outcomes.

To make use of the code in this chapter, use one of the provided `StatisticalDebugger` subclasses such as `TarantulaDebugger` or `OchiaiDebugger`. 

Both are instantiated with a `Collector` denoting the type of events you want to correlate outcomes with. The default `CoverageCollector`, collecting line coverage.

### Collecting Events from Calls

To collect events from calls that are labeled manually, use

>>> debugger = TarantulaDebugger()
>>> with debugger.collect_pass():
>>>     remove_html_markup("abc")
>>> with debugger.collect_pass():
>>>     remove_html_markup('abc')
>>> with debugger.collect_fail():
>>>     remove_html_markup('"abc"')

Within each `with` block, the _first function call_ is collected and tracked for coverage. (Note that _only_ the first call is tracked.)

### Collecting Events from Tests

To collect events from _tests_ that use exceptions to indicate failure, use the simpler `with` form:

>>> debugger = TarantulaDebugger()
>>> with debugger:
>>>     remove_html_markup("abc")
>>> with debugger:
>>>     remove_html_markup('abc')
>>> with debugger:
>>>     remove_html_markup('"abc"')
>>>     assert False  # raise an exception

`with` blocks that raise an exception will be classified as failing, blocks that do not will be classified as passing. Note that exceptions raised are "swallowed" by the debugger.

### Visualizing Events as a Table

After collecting events, you can print out the observed events – in this case, line numbers – in a table, showing in which runs they occurred (`X`), and with colors highlighting the suspiciousness of the event. A "red" event means that the event predominantly occurs in failing runs.

>>> debugger.event_table(args=True, color=True)

| `remove_html_markup` | `s='abc'` | `s='abc'` | `s='"abc"'` | 
| --------------------- | ---- | ---- | ---- | 
|  remove_html_markup:1 |    X |    X |    X | 
|  remove_html_markup:2 |    X |    X |    X | 
|  remove_html_markup:3 |    X |    X |    X | 
|  remove_html_markup:4 |    X |    X |    X | 
|  remove_html_markup:6 |    X |    X |    X | 
|  remove_html_markup:7 |    X |    X |    X | 
|  remove_html_markup:8 |    - |    X |    - | 
|  remove_html_markup:9 |    X |    X |    X | 
| remove_html_markup:10 |    - |    X |    - | 
| remove_html_markup:11 |    X |    X |    X | 
| remove_html_markup:12 |    - |    - |    X | 
| remove_html_markup:13 |    X |    X |    X | 
| remove_html_markup:14 |    X |    X |    X | 
| remove_html_markup:16 |    X |    X |    X | 


### Visualizing Suspicious Code

If you collected coverage with `CoverageCollector`, you can also visualize the code with similar colors, highlighting suspicious lines:

>>> debugger

   1 def remove_html_markup(s):  # type: ignore
   2     tag = False
   3     quote = False
   4     out = ""
   5  
   6     for c in s:
   7         if c == '<' and not quote:
   8             tag = True
   9         elif c == '>' and not quote:
  10             tag = False
  11         elif c == '"' or c == "'" and tag:
  12             quote = not quote
  13         elif not tag:
  14             out = out + c
  15  
  16     return out


### Ranking Events

The method `rank()` returns a ranked list of events, starting with the most suspicious. This is useful for automated techniques that need potential defect locations.

>>> debugger.rank()
[('remove_html_markup', 12),
 ('remove_html_markup', 3),
 ('remove_html_markup', 9),
 ('remove_html_markup', 6),
 ('remove_html_markup', 4),
 ('remove_html_markup', 1),
 ('remove_html_markup', 7),
 ('remove_html_markup', 16),
 ('remove_html_markup', 13),
 ('remove_html_markup', 2),
 ('remove_html_markup', 11),
 ('remove_html_markup', 14),
 ('remove_html_markup', 10),
 ('remove_html_markup', 8)]

### Classes and Methods

Here are all classes defined in this chapter:
![](PICS/StatisticalDebugger-synopsis-2.svg)


For more details, source, and documentation, see
"The Debugging Book - Statistical Debugging"
at https://www.debuggingbook.org/html/StatisticalDebugger.html
'''


# Allow to use 'from . import <module>' when run as script (cf. PEP 366)
if __name__ == '__main__' and __package__ is None:
    __package__ = 'debuggingbook'


# Statistical Debugging
# =====================

if __name__ == '__main__':
    print('# Statistical Debugging')



if __name__ == '__main__':
    from .bookutils import YouTubeVideo
    YouTubeVideo("UNuso00zYiI")

if __name__ == '__main__':
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



## Introduction
## ------------

if __name__ == '__main__':
    print('\n## Introduction')



## Collecting Events
## -----------------

if __name__ == '__main__':
    print('\n## Collecting Events')



from .Tracer import Tracer

from typing import Any, Callable, Optional, Type, Tuple
from typing import Dict, Set, List, TypeVar, Union

from types import FrameType, TracebackType

class Collector(Tracer):
    """A class to record events during execution."""

    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        """Collecting function. To be overridden in subclasses."""
        pass

    def events(self) -> Set:
        """Return a collection of events. To be overridden in subclasses."""
        return set()

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        self.collect(frame, event, arg)

def remove_html_markup(s):  # type: ignore
    tag = False
    quote = False
    out = ""

    for c in s:
        if c == '<' and not quote:
            tag = True
        elif c == '>' and not quote:
            tag = False
        elif c == '"' or c == "'" and tag:
            quote = not quote
        elif not tag:
            out = out + c

    return out

if __name__ == '__main__':
    with Collector() as c:
        out = remove_html_markup('"abc"')
    out

Coverage = Set[Tuple[Callable, int]]

class Collector(Collector):
    def __init__(self) -> None:
        """Constructor."""
        self._function: Optional[Callable] = None
        self._args: Optional[Dict[str, Any]] = None
        self._argstring: Optional[str] = None
        self._exception: Optional[Type] = None
        self.items_to_ignore: List[Union[Type, Callable]] = [self.__class__]

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """
        Tracing function.
        Saves the first function and calls collect().
        """
        for item in self.items_to_ignore:
            if (isinstance(item, type) and 'self' in frame.f_locals and
                isinstance(frame.f_locals['self'], item)):
                # Ignore this class
                return
            if item.__name__ == frame.f_code.co_name:
                # Ignore this function
                return

        if self._function is None and event == 'call':
            # Save function
            self._function = self.create_function(frame)
            self._args = frame.f_locals.copy()
            self._argstring = ", ".join([f"{var}={repr(self._args[var])}" 
                                         for var in self._args])

        self.collect(frame, event, arg)

    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        """Collector function. To be overloaded in subclasses."""
        pass

    def id(self) -> str:
        """Return an identifier for the collector, 
        created from the first call"""
        return f"{self.function().__name__}({self.argstring()})"

    def function(self) -> Callable:
        """Return the function from the first call, as a function object"""
        if not self._function:
            raise ValueError("No call collected")
        return self._function

    def argstring(self) -> str:
        """
        Return the list of arguments from the first call,
        as a printable string
        """
        if not self._argstring:
            raise ValueError("No call collected")
        return self._argstring

    def args(self) -> Dict[str, Any]:
        """Return a dict of argument names and values from the first call"""
        if not self._args:
            raise ValueError("No call collected")
        return self._args

    def exception(self) -> Optional[Type]:
        """Return the exception class from the first call,
        or None if no exception was raised."""
        return self._exception

    def __repr__(self) -> str:
        """Return a string representation of the collector"""
        # We use the ID as default representation when printed
        return self.id()

    def covered_functions(self) -> Set[Callable]:
        """Set of covered functions. To be overloaded in subclasses."""
        return set()

    def coverage(self) -> Coverage:
        """
        Return a set (function, lineno) with locations covered.
        To be overloaded in subclasses.
        """
        return set()

if __name__ == '__main__':
    with Collector() as c:
        remove_html_markup('abc')

if __name__ == '__main__':
    c.function()

if __name__ == '__main__':
    c.args()

if __name__ == '__main__':
    c.id()

if __name__ == '__main__':
    c.argstring()

### Error Prevention

if __name__ == '__main__':
    print('\n### Error Prevention')



class Collector(Collector):
    def add_items_to_ignore(self,
                            items_to_ignore: List[Union[Type, Callable]]) \
                            -> None:
        """
        Define additional classes and functions to ignore during collection
        (typically `Debugger` classes using these collectors).
        """
        self.items_to_ignore += items_to_ignore

class Collector(Collector):
    def __exit__(self, exc_tp: Type, exc_value: BaseException,
                 exc_traceback: TracebackType) -> Optional[bool]:
        """Exit the `with` block."""
        ret = super().__exit__(exc_tp, exc_value, exc_traceback)

        if not self._function:
            if exc_tp:
                return False  # re-raise exception
            else:
                raise ValueError("No call collected")

        return ret

## Collecting Coverage
## -------------------

if __name__ == '__main__':
    print('\n## Collecting Coverage')



from types import FrameType

from .StackInspector import StackInspector

class CoverageCollector(Collector, StackInspector):
    """A class to record covered locations during execution."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        self._coverage: Coverage = set()

    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        """
        Save coverage for an observed event.
        """
        name = frame.f_code.co_name
        function = self.search_func(name, frame)

        if function is None:
            function = self.create_function(frame)

        location = (function, frame.f_lineno)
        self._coverage.add(location)

class CoverageCollector(CoverageCollector):
    def events(self) -> Set[Tuple[str, int]]:
        """
        Return the set of locations covered.
        Each location comes as a pair (`function_name`, `lineno`).
        """
        return {(func.__name__, lineno) for func, lineno in self._coverage}

class CoverageCollector(CoverageCollector):
    def covered_functions(self) -> Set[Callable]:
        """Return a set with all functions covered."""
        return {func for func, lineno in self._coverage}

    def coverage(self) -> Coverage:
        """Return a set (function, lineno) with all locations covered."""
        return self._coverage

if __name__ == '__main__':
    with CoverageCollector() as c:
        remove_html_markup('abc')
    c.events()

import inspect

from .bookutils import getsourcelines    # like inspect.getsourcelines(), but in color

def code_with_coverage(function: Callable, coverage: Coverage) -> None:
    source_lines, starting_line_number = \
       getsourcelines(function)

    line_number = starting_line_number
    for line in source_lines:
        marker = '*' if (function, line_number) in coverage else ' '
        print(f"{line_number:4} {marker} {line}", end='')
        line_number += 1

if __name__ == '__main__':
    code_with_coverage(remove_html_markup, c.coverage())

from .bookutils import quiz

if __name__ == '__main__':
    quiz('Let the input be `"<b>Don\'t do this!</b>"`. '
         "Which of these lines are executed? Use the code to find out!",
         [
             "`tag = True`",
             "`tag = False`",
             "`quote = not quote`",
             "`out = out + c`"
         ], "[ord(c) - ord('a') - 1 for c in 'cdf']")

if __name__ == '__main__':
    with CoverageCollector() as c:
        remove_html_markup("<b>Don't do this!</b>")
    # code_with_coverage(remove_html_markup, c.coverage)

## Computing Differences
## ---------------------

if __name__ == '__main__':
    print('\n## Computing Differences')



### A Base Class for Statistical Debugging

if __name__ == '__main__':
    print('\n### A Base Class for Statistical Debugging')



class StatisticalDebugger:
    """A class to collect events for multiple outcomes."""

    def __init__(self, collector_class: Type = CoverageCollector, log: bool = False):
        """Constructor. Use instances of `collector_class` to collect events."""
        self.collector_class = collector_class
        self.collectors: Dict[str, List[Collector]] = {}
        self.log = log

class StatisticalDebugger(StatisticalDebugger):
    def collect(self, outcome: str, *args: Any, **kwargs: Any) -> Collector:
        """Return a collector for the given outcome. 
        Additional args are passed to the collector."""
        collector = self.collector_class(*args, **kwargs)
        collector.add_items_to_ignore([self.__class__])
        return self.add_collector(outcome, collector)

    def add_collector(self, outcome: str, collector: Collector) -> Collector:
        if outcome not in self.collectors:
            self.collectors[outcome] = []
        self.collectors[outcome].append(collector)
        return collector

class StatisticalDebugger(StatisticalDebugger):
    def all_events(self, outcome: Optional[str] = None) -> Set[Any]:
        """Return a set of all events observed."""
        all_events = set()

        if outcome:
            if outcome in self.collectors:
                for collector in self.collectors[outcome]:
                    all_events.update(collector.events())
        else:
            for outcome in self.collectors:
                for collector in self.collectors[outcome]:
                    all_events.update(collector.events())

        return all_events

if __name__ == '__main__':
    s = StatisticalDebugger()
    with s.collect('PASS'):
        remove_html_markup("abc")
    with s.collect('PASS'):
        remove_html_markup('<b>abc</b>')
    with s.collect('FAIL'):
        remove_html_markup('"abc"')

if __name__ == '__main__':
    s.all_events()

if __name__ == '__main__':
    s.all_events('FAIL')

if __name__ == '__main__':
    s.collectors

if __name__ == '__main__':
    s.collectors['PASS'][0].id()

if __name__ == '__main__':
    s.collectors['PASS'][0].events()

### Excursion: Printing an Event Table

if __name__ == '__main__':
    print('\n### Excursion: Printing an Event Table')



if __name__ == '__main__':
    from IPython.display import Markdown

import html

class StatisticalDebugger(StatisticalDebugger):
    def function(self) -> Optional[Callable]:
        """
        Return the entry function from the events observed,
        or None if ambiguous.
        """
        names_seen = set()
        functions = []
        for outcome in self.collectors:
            for collector in self.collectors[outcome]:
                # We may have multiple copies of the function,
                # but sharing the same name
                func = collector.function()
                if func.__name__ not in names_seen:
                    functions.append(func)
                    names_seen.add(func.__name__)

        if len(functions) != 1:
            return None  # ambiguous
        return functions[0]

    def covered_functions(self) -> Set[Callable]:
        """Return a set of all functions observed."""
        functions = set()
        for outcome in self.collectors:
            for collector in self.collectors[outcome]:
                functions |= collector.covered_functions()
        return functions

    def coverage(self) -> Coverage:
        """Return a set of all (functions, line_numbers) observed"""
        coverage = set()
        for outcome in self.collectors:
            for collector in self.collectors[outcome]:
                coverage |= collector.coverage()
        return coverage

    def color(self, event: Any) -> Optional[str]:
        """
        Return a color for the given event, or None.
        To be overloaded in subclasses.
        """
        return None

    def tooltip(self, event: Any) -> Optional[str]:
        """
        Return a tooltip string for the given event, or None.
        To be overloaded in subclasses.
        """
        return None

    def event_str(self, event: Any) -> str:
        """Format the given event. To be overloaded in subclasses."""
        if isinstance(event, str):
            return event
        if isinstance(event, tuple):
            return ":".join(self.event_str(elem) for elem in event)
        return str(event)

    def event_table_text(self, *, args: bool = False, color: bool = False) -> str:
        """
        Print out a table of events observed.
        If `args` is True, use arguments as headers.
        If `color` is True, use colors.
        """
        sep = ' | '
        all_events = self.all_events()
        longest_event = max(len(f"{self.event_str(event)}") 
                            for event in all_events)
        out = ""

        # Header
        if args:
            out += '| '
            func = self.function()
            if func:
                out += '`' + func.__name__ + '`'
            out += sep
            for name in self.collectors:
                for collector in self.collectors[name]:
                    out += '`' + collector.argstring() + '`' + sep
            out += '\n'
        else:
            out += '| ' + ' ' * longest_event + sep
            for name in self.collectors:
                for i in range(len(self.collectors[name])):
                    out += name + sep
            out += '\n'

        out += '| ' + '-' * longest_event + sep
        for name in self.collectors:
            for i in range(len(self.collectors[name])):
                out += '-' * len(name) + sep
        out += '\n'

        # Data
        for event in sorted(all_events):
            event_name = self.event_str(event).rjust(longest_event)

            tooltip = self.tooltip(event)
            if tooltip:
                title = f' title="{tooltip}"'
            else:
                title = ''

            if color:
                color_name = self.color(event)
                if color_name:
                    event_name = \
                        f'<samp style="background-color: {color_name}"{title}>' \
                        f'{html.escape(event_name)}' \
                        f'</samp>'

            out += f"| {event_name}" + sep
            for name in self.collectors:
                for collector in self.collectors[name]:
                    out += ' ' * (len(name) - 1)
                    if event in collector.events():
                        out += "X"
                    else:
                        out += "-"
                    out += sep
            out += '\n'

        return out

    def event_table(self, **_args: Any) -> Any:
        """Print out event table in Markdown format."""
        return Markdown(self.event_table_text(**_args))

    def __repr__(self) -> str:
        return self.event_table_text()

    def _repr_markdown_(self) -> str:
        return self.event_table_text(args=True, color=True)

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



if __name__ == '__main__':
    s = StatisticalDebugger()
    with s.collect('PASS'):
        remove_html_markup("abc")
    with s.collect('PASS'):
        remove_html_markup('<b>abc</b>')
    with s.collect('FAIL'):
        remove_html_markup('"abc"')

if __name__ == '__main__':
    s.event_table(args=True)

if __name__ == '__main__':
    quiz("How many lines are executed in the failing run only?",
         [
             "One",
             "Two",
             "Three"
         ], 'len([12])')

### Collecting Passing and Failing Runs

if __name__ == '__main__':
    print('\n### Collecting Passing and Failing Runs')



class DifferenceDebugger(StatisticalDebugger):
    """A class to collect events for passing and failing outcomes."""

    PASS = 'PASS'
    FAIL = 'FAIL'

    def collect_pass(self, *args: Any, **kwargs: Any) -> Collector:
        """Return a collector for passing runs."""
        return self.collect(self.PASS, *args, **kwargs)

    def collect_fail(self, *args: Any, **kwargs: Any) -> Collector:
        """Return a collector for failing runs."""
        return self.collect(self.FAIL, *args, **kwargs)

    def pass_collectors(self) -> List[Collector]:
        return self.collectors[self.PASS]

    def fail_collectors(self) -> List[Collector]:
        return self.collectors[self.FAIL]

    def all_fail_events(self) -> Set[Any]:
        """Return all events observed in failing runs."""
        return self.all_events(self.FAIL)

    def all_pass_events(self) -> Set[Any]:
        """Return all events observed in passing runs."""
        return self.all_events(self.PASS)

    def only_fail_events(self) -> Set[Any]:
        """Return all events observed only in failing runs."""
        return self.all_fail_events() - self.all_pass_events()

    def only_pass_events(self) -> Set[Any]:
        """Return all events observed only in passing runs."""
        return self.all_pass_events() - self.all_fail_events()

T1 = TypeVar('T1', bound='DifferenceDebugger')

def test_debugger_html_simple(debugger: T1) -> T1:
    with debugger.collect_pass():
        remove_html_markup('abc')
    with debugger.collect_pass():
        remove_html_markup('<b>abc</b>')
    with debugger.collect_fail():
        remove_html_markup('"abc"')
    return debugger

class DifferenceDebugger(DifferenceDebugger):
    def __enter__(self) -> Any:
        """Enter a `with` block. Collect coverage and outcome;
        classify as FAIL if the block raises an exception,
        and PASS if it does not.
        """
        self.collector = self.collector_class()
        self.collector.add_items_to_ignore([self.__class__])
        self.collector.__enter__()
        return self

    def __exit__(self, exc_tp: Type, exc_value: BaseException,
                 exc_traceback: TracebackType) -> Optional[bool]:
        """Exit the `with` block."""
        status = self.collector.__exit__(exc_tp, exc_value, exc_traceback)

        if status is None:
            pass
        else:
            return False  # Internal error; re-raise exception

        if exc_tp is None:
            outcome = self.PASS
        else:
            outcome = self.FAIL

        self.add_collector(outcome, self.collector)
        return True  # Ignore exception, if any

T2 = TypeVar('T2', bound='DifferenceDebugger')

def test_debugger_html(debugger: T2) -> T2:
    with debugger:
        remove_html_markup('abc')
    with debugger:
        remove_html_markup('<b>abc</b>')
    with debugger:
        remove_html_markup('"abc"')
        assert False  # Mark test as failing

    return debugger

if __name__ == '__main__':
    test_debugger_html(DifferenceDebugger())

### Analyzing Events

if __name__ == '__main__':
    print('\n### Analyzing Events')



if __name__ == '__main__':
    debugger = test_debugger_html(DifferenceDebugger())

if __name__ == '__main__':
    pass_1_events = debugger.pass_collectors()[0].events()

if __name__ == '__main__':
    pass_2_events = debugger.pass_collectors()[1].events()

if __name__ == '__main__':
    in_any_pass = pass_1_events | pass_2_events
    in_any_pass

if __name__ == '__main__':
    fail_events = debugger.fail_collectors()[0].events()

if __name__ == '__main__':
    only_in_fail = fail_events - in_any_pass
    only_in_fail

if __name__ == '__main__':
    code_with_coverage(remove_html_markup, only_in_fail)

if __name__ == '__main__':
    debugger = test_debugger_html(DifferenceDebugger())

if __name__ == '__main__':
    debugger.all_events()

if __name__ == '__main__':
    debugger.only_fail_events()

if __name__ == '__main__':
    debugger.only_pass_events()

## Visualizing Differences
## -----------------------

if __name__ == '__main__':
    print('\n## Visualizing Differences')



### Discrete Spectrum

if __name__ == '__main__':
    print('\n### Discrete Spectrum')



class SpectrumDebugger(DifferenceDebugger):
    def suspiciousness(self, event: Any) -> Optional[float]:
        """
        Return a suspiciousness value in the range [0, 1.0]
        for the given event, or `None` if unknown.
        To be overloaded in subclasses.
        """
        return None

class SpectrumDebugger(SpectrumDebugger):
    def tooltip(self, event: Any) -> str:
        """
        Return a tooltip for the given event (default: percentage).
        To be overloaded in subclasses.
        """
        return self.percentage(event)

    def percentage(self, event: Any) -> str:
        """
        Return the suspiciousness for the given event as percentage string.
        """
        suspiciousness = self.suspiciousness(event)
        if suspiciousness is not None:
            return str(int(suspiciousness * 100)).rjust(3) + '%'
        else:
            return ' ' * len('100%')

class SpectrumDebugger(SpectrumDebugger):
    def code(self, functions: Optional[Set[Callable]] = None, *, 
             color: bool = False, suspiciousness: bool = False,
             line_numbers: bool = True) -> str:
        """
        Return a listing of `functions` (default: covered functions).
        If `color` is True, render as HTML, using suspiciousness colors.
        If `suspiciousness` is True, include suspiciousness values.
        If `line_numbers` is True (default), include line numbers.
        """

        if not functions:
            functions = self.covered_functions()

        out = ""
        seen = set()
        for function in functions:
            source_lines, starting_line_number = \
               inspect.getsourcelines(function)

            if (function.__name__, starting_line_number) in seen:
                continue
            seen.add((function.__name__, starting_line_number))

            if out:
                out += '\n'
                if color:
                    out += '<p/>'

            line_number = starting_line_number
            for line in source_lines:
                if color:
                    line = html.escape(line)
                    if line.strip() == '':
                        line = '&nbsp;'

                location = (function.__name__, line_number)
                location_suspiciousness = self.suspiciousness(location)
                if location_suspiciousness is not None:
                    tooltip = f"Line {line_number}: {self.tooltip(location)}"
                else:
                    tooltip = f"Line {line_number}: not executed"

                if suspiciousness:
                    line = self.percentage(location) + ' ' + line

                if line_numbers:
                    line = str(line_number).rjust(4) + ' ' + line

                line_color = self.color(location)

                if color and line_color:
                    line = f'''<pre style="background-color:{line_color}"
                    title="{tooltip}">{line.rstrip()}</pre>'''
                elif color:
                    line = f'<pre title="{tooltip}">{line}</pre>'
                else:
                    line = line.rstrip()

                out += line + '\n'
                line_number += 1

        return out

class SpectrumDebugger(SpectrumDebugger):
    def _repr_html_(self) -> str:
        """When output in Jupyter, visualize as HTML"""
        return self.code(color=True)

    def __str__(self) -> str:
        """Show code as string"""
        return self.code(color=False, suspiciousness=True)

    def __repr__(self) -> str:
        """Show code as string"""
        return self.code(color=False, suspiciousness=True)

class DiscreteSpectrumDebugger(SpectrumDebugger):
    """Visualize differences between executions using three discrete colors"""

    def suspiciousness(self, event: Any) -> Optional[float]:
        """
        Return a suspiciousness value [0, 1.0]
        for the given event, or `None` if unknown.
        """
        passing = self.all_pass_events()
        failing = self.all_fail_events()

        if event in passing and event in failing:
            return 0.5
        elif event in failing:
            return 1.0
        elif event in passing:
            return 0.0
        else:
            return None

    def color(self, event: Any) -> Optional[str]:
        """
        Return a HTML color for the given event.
        """
        suspiciousness = self.suspiciousness(event)
        if suspiciousness is None:
            return None

        if suspiciousness > 0.8:
            return 'mistyrose'
        if suspiciousness >= 0.5:
            return 'lightyellow'

        return 'honeydew'

    def tooltip(self, event: Any) -> str:
        """Return a tooltip for the given event."""
        passing = self.all_pass_events()
        failing = self.all_fail_events()

        if event in passing and event in failing:
            return "in passing and failing runs"
        elif event in failing:
            return "only in failing runs"
        elif event in passing:
            return "only in passing runs"
        else:
            return "never"

if __name__ == '__main__':
    debugger = test_debugger_html(DiscreteSpectrumDebugger())

if __name__ == '__main__':
    debugger

if __name__ == '__main__':
    quiz("Does the line `quote = not quote` actually contain the defect?",
        [
            "Yes, it should be fixed",
            "No, the defect is elsewhere"
        ], '164 * 2 % 326')

if __name__ == '__main__':
    print(debugger)

### Continuous Spectrum

if __name__ == '__main__':
    print('\n### Continuous Spectrum')



if __name__ == '__main__':
    remove_html_markup('<b color="blue">text</b>')

if __name__ == '__main__':
    debugger = test_debugger_html(DiscreteSpectrumDebugger())
    with debugger.collect_pass():
        remove_html_markup('<b link="blue"></b>')

if __name__ == '__main__':
    debugger.only_fail_events()

if __name__ == '__main__':
    debugger

class ContinuousSpectrumDebugger(DiscreteSpectrumDebugger):
    """Visualize differences between executions using a color spectrum"""

    def collectors_with_event(self, event: Any, category: str) -> Set[Collector]:
        """
        Return all collectors in a category
        that observed the given event.
        """
        all_runs = self.collectors[category]
        collectors_with_event = set(collector for collector in all_runs 
                                    if event in collector.events())
        return collectors_with_event

    def collectors_without_event(self, event: Any, category: str) -> Set[Collector]:
        """
        Return all collectors in a category
        that did not observe the given event.
        """
        all_runs = self.collectors[category]
        collectors_without_event = set(collector for collector in all_runs 
                              if event not in collector.events())
        return collectors_without_event

    def event_fraction(self, event: Any, category: str) -> float:
        if category not in self.collectors:
            return 0.0

        all_collectors = self.collectors[category]
        collectors_with_event = self.collectors_with_event(event, category)
        fraction = len(collectors_with_event) / len(all_collectors)
        # print(f"%{category}({event}) = {fraction}")
        return fraction

    def passed_fraction(self, event: Any) -> float:
        return self.event_fraction(event, self.PASS)

    def failed_fraction(self, event: Any) -> float:
        return self.event_fraction(event, self.FAIL)

    def hue(self, event: Any) -> Optional[float]:
        """Return a color hue from 0.0 (red) to 1.0 (green)."""
        passed = self.passed_fraction(event)
        failed = self.failed_fraction(event)
        if passed + failed > 0:
            return passed / (passed + failed)
        else:
            return None

class ContinuousSpectrumDebugger(ContinuousSpectrumDebugger):
    def suspiciousness(self, event: Any) -> Optional[float]:
        hue = self.hue(event)
        if hue is None:
            return None
        return 1 - hue

    def tooltip(self, event: Any) -> str:
        return self.percentage(event)

if __name__ == '__main__':
    debugger = test_debugger_html(ContinuousSpectrumDebugger())

if __name__ == '__main__':
    for location in debugger.only_fail_events():
        print(location, debugger.hue(location))

if __name__ == '__main__':
    for location in debugger.only_pass_events():
        print(location, debugger.hue(location))

class ContinuousSpectrumDebugger(ContinuousSpectrumDebugger):
    def brightness(self, event: Any) -> float:
        return max(self.passed_fraction(event), self.failed_fraction(event))

if __name__ == '__main__':
    debugger = test_debugger_html(ContinuousSpectrumDebugger())
    for location in debugger.only_fail_events():
        print(location, debugger.brightness(location))

class ContinuousSpectrumDebugger(ContinuousSpectrumDebugger):
    def color(self, event: Any) -> Optional[str]:
        hue = self.hue(event)
        if hue is None:
            return None
        saturation = self.brightness(event)

        # HSL color values are specified with: 
        # hsl(hue, saturation, lightness).
        return f"hsl({hue * 120}, {saturation * 100}%, 80%)"

if __name__ == '__main__':
    debugger = test_debugger_html(ContinuousSpectrumDebugger())

if __name__ == '__main__':
    for location in debugger.only_fail_events():
        print(location, debugger.color(location))

if __name__ == '__main__':
    for location in debugger.only_pass_events():
        print(location, debugger.color(location))

if __name__ == '__main__':
    debugger

if __name__ == '__main__':
    with debugger.collect_pass():
        out = remove_html_markup('<b link="blue"></b>')

if __name__ == '__main__':
    quiz('In which color will the `quote = not quote` "culprit" line '
         'be shown after executing the above code?',
        [
            '<span style="background-color: hsl(120.0, 50.0%, 80%)">Green</span>',
            '<span style="background-color: hsl(60.0, 100.0%, 80%)">Yellow</span>',
            '<span style="background-color: hsl(30.0, 100.0%, 80%)">Orange</span>',
            '<span style="background-color: hsl(0.0, 100.0%, 80%)">Red</span>'
        ], '999 // 333')

if __name__ == '__main__':
    debugger

def middle(x, y, z):  # type: ignore
    if y < z:
        if x < y:
            return y
        elif x < z:
            return y
    else:
        if x > y:
            return y
        elif x > z:
            return x
    return z

if __name__ == '__main__':
    middle(1, 2, 3)

if __name__ == '__main__':
    middle(2, 1, 3)

T3 = TypeVar('T3', bound='DifferenceDebugger')

def test_debugger_middle(debugger: T3) -> T3:
    with debugger.collect_pass():
        middle(3, 3, 5)
    with debugger.collect_pass():
        middle(1, 2, 3)
    with debugger.collect_pass():
        middle(3, 2, 1)
    with debugger.collect_pass():
        middle(5, 5, 5)
    with debugger.collect_pass():
        middle(5, 3, 4)
    with debugger.collect_fail():
        middle(2, 1, 3)
    return debugger

if __name__ == '__main__':
    debugger = test_debugger_middle(ContinuousSpectrumDebugger())

if __name__ == '__main__':
    debugger.event_table(args=True)

if __name__ == '__main__':
    debugger

if __name__ == '__main__':
    quiz("Which of the above lines should be fixed?",
        [
            '<span style="background-color: hsl(45.0, 100%, 80%)">Line 3: `if x < y`</span>',
            '<span style="background-color: hsl(34.28571428571429, 100.0%, 80%)">Line 5: `elif x < z`</span>',
            '<span style="background-color: hsl(20.000000000000004, 100.0%, 80%)">Line 6: `return y`</span>',
            '<span style="background-color: hsl(120.0, 20.0%, 80%)">Line 9: `return y`</span>',
        ], r'len(" middle  ".strip()[:3])')

def middle_fixed(x, y, z):  # type: ignore
    if y < z:
        if x < y:
            return y
        elif x < z:
            return x
    else:
        if x > y:
            return y
        elif x > z:
            return x
    return z

if __name__ == '__main__':
    middle_fixed(2, 1, 3)

## Ranking Lines by Suspiciousness
## -------------------------------

if __name__ == '__main__':
    print('\n## Ranking Lines by Suspiciousness')



class RankingDebugger(DiscreteSpectrumDebugger):
    """Rank events by their suspiciousness"""

    def rank(self) -> List[Any]:
        """Return a list of events, sorted by suspiciousness, highest first."""

        def susp(event: Any) -> float:
            suspiciousness = self.suspiciousness(event)
            assert suspiciousness is not None
            return suspiciousness

        events = list(self.all_events())
        events.sort(key=susp, reverse=True)
        return events

    def __repr__(self) -> str:
        return repr(self.rank())

### The Tarantula Metric

if __name__ == '__main__':
    print('\n### The Tarantula Metric')



class TarantulaDebugger(ContinuousSpectrumDebugger, RankingDebugger):
    """Spectrum-based Debugger using the Tarantula metric for suspiciousness"""
    pass

if __name__ == '__main__':
    tarantula_html = test_debugger_html(TarantulaDebugger())

if __name__ == '__main__':
    tarantula_html

if __name__ == '__main__':
    tarantula_html.rank()

if __name__ == '__main__':
    tarantula_html.suspiciousness(tarantula_html.rank()[0])

if __name__ == '__main__':
    tarantula_middle = test_debugger_middle(TarantulaDebugger())

if __name__ == '__main__':
    tarantula_middle

if __name__ == '__main__':
    tarantula_middle.rank()

if __name__ == '__main__':
    tarantula_middle.suspiciousness(tarantula_middle.rank()[0])

### The Ochiai Metric

if __name__ == '__main__':
    print('\n### The Ochiai Metric')



import math

class OchiaiDebugger(ContinuousSpectrumDebugger, RankingDebugger):
    """Spectrum-based Debugger using the Ochiai metric for suspiciousness"""

    def suspiciousness(self, event: Any) -> Optional[float]:
        failed = len(self.collectors_with_event(event, self.FAIL))
        not_in_failed = len(self.collectors_without_event(event, self.FAIL))
        passed = len(self.collectors_with_event(event, self.PASS))

        try:
            return failed / math.sqrt((failed + not_in_failed) * (failed + passed))
        except ZeroDivisionError:
            return None

    def hue(self, event: Any) -> Optional[float]:
        suspiciousness = self.suspiciousness(event)
        if suspiciousness is None:
            return None
        return 1 - suspiciousness

if __name__ == '__main__':
    ochiai_html = test_debugger_html(OchiaiDebugger())

if __name__ == '__main__':
    ochiai_html

if __name__ == '__main__':
    ochiai_html.rank()

if __name__ == '__main__':
    ochiai_html.suspiciousness(ochiai_html.rank()[0])

if __name__ == '__main__':
    ochiai_middle = test_debugger_middle(OchiaiDebugger())

if __name__ == '__main__':
    ochiai_middle

if __name__ == '__main__':
    ochiai_middle.rank()

if __name__ == '__main__':
    ochiai_middle.suspiciousness(ochiai_middle.rank()[0])

### How Useful is Ranking?

if __name__ == '__main__':
    print('\n### How Useful is Ranking?')



## Using Large Test Suites
## -----------------------

if __name__ == '__main__':
    print('\n## Using Large Test Suites')



import random

def middle_testcase() -> Tuple[int, int, int]:
    x = random.randrange(10)
    y = random.randrange(10)
    z = random.randrange(10)
    return x, y, z

if __name__ == '__main__':
    [middle_testcase() for i in range(5)]

def middle_test(x: int, y: int, z: int) -> None:
    m = middle(x, y, z)
    assert m == sorted([x, y, z])[1]

if __name__ == '__main__':
    middle_test(4, 5, 6)

from .ExpectError import ExpectError

if __name__ == '__main__':
    with ExpectError():
        middle_test(2, 1, 3)

def middle_passing_testcase() -> Tuple[int, int, int]:
    while True:
        try:
            x, y, z = middle_testcase()
            middle_test(x, y, z)
            return x, y, z
        except AssertionError:
            pass

if __name__ == '__main__':
    (x, y, z) = middle_passing_testcase()
    m = middle(x, y, z)
    print(f"middle({x}, {y}, {z}) = {m}")

def middle_failing_testcase() -> Tuple[int, int, int]:
    while True:
        try:
            x, y, z = middle_testcase()
            middle_test(x, y, z)
        except AssertionError:
            return x, y, z

if __name__ == '__main__':
    (x, y, z) = middle_failing_testcase()
    m = middle(x, y, z)
    print(f"middle({x}, {y}, {z}) = {m}")

MIDDLE_TESTS = 100

MIDDLE_PASSING_TESTCASES = [middle_passing_testcase()
                            for i in range(MIDDLE_TESTS)]

MIDDLE_FAILING_TESTCASES = [middle_failing_testcase()
                            for i in range(MIDDLE_TESTS)]

if __name__ == '__main__':
    ochiai_middle = OchiaiDebugger()

    for x, y, z in MIDDLE_PASSING_TESTCASES:
        with ochiai_middle.collect_pass():
            middle(x, y, z)

    for x, y, z in MIDDLE_FAILING_TESTCASES:
        with ochiai_middle.collect_fail():
            middle(x, y, z)

if __name__ == '__main__':
    ochiai_middle

## Other Events besides Coverage
## -----------------------------

if __name__ == '__main__':
    print('\n## Other Events besides Coverage')



class ValueCollector(Collector):
    """"A class to collect local variables and their values."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        self.vars: Set[str] = set()

    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        local_vars = frame.f_locals
        for var in local_vars:
            value = local_vars[var]
            self.vars.add(f"{var} = {repr(value)}")

    def events(self) -> Set[str]:
        """A set of (variable, value) pairs observed"""
        return self.vars

if __name__ == '__main__':
    debugger = test_debugger_html(ContinuousSpectrumDebugger(ValueCollector))
    for event in debugger.all_events():
        print(event)

if __name__ == '__main__':
    for event in debugger.only_fail_events():
        print(event)

if __name__ == '__main__':
    debugger.event_table(color=True, args=True)

## Training Classifiers
## --------------------

if __name__ == '__main__':
    print('\n## Training Classifiers')



class ClassifyingDebugger(DifferenceDebugger):
    """A debugger implementing a decision tree for events"""

    PASS_VALUE = +1.0
    FAIL_VALUE = -1.0

    def samples(self) -> Dict[str, float]:
        samples = {}
        for collector in self.pass_collectors():
            samples[collector.id()] = self.PASS_VALUE
        for collector in debugger.fail_collectors():
            samples[collector.id()] = self.FAIL_VALUE
        return samples

if __name__ == '__main__':
    debugger = test_debugger_html(ClassifyingDebugger())
    debugger.samples()

class ClassifyingDebugger(ClassifyingDebugger):
    def features(self) -> Dict[str, Any]:
        features = {}
        for collector in debugger.pass_collectors():
            features[collector.id()] = collector.events()
        for collector in debugger.fail_collectors():
            features[collector.id()] = collector.events()
        return features

if __name__ == '__main__':
    debugger = test_debugger_html(ClassifyingDebugger())
    debugger.features()

class ClassifyingDebugger(ClassifyingDebugger):
    def feature_names(self) -> List[str]:
        return [repr(feature) for feature in self.all_events()]

if __name__ == '__main__':
    debugger = test_debugger_html(ClassifyingDebugger())
    debugger.feature_names()

class ClassifyingDebugger(ClassifyingDebugger):
    def shape(self, sample: str) -> List[float]:
        x = []
        features = self.features()
        for f in self.all_events():
            if f in features[sample]:
                x += [+1.0]
            else:
                x += [-1.0]
        return x

if __name__ == '__main__':
    debugger = test_debugger_html(ClassifyingDebugger())
    debugger.shape("remove_html_markup(s='abc')")

class ClassifyingDebugger(ClassifyingDebugger):
    def X(self) -> List[List[float]]:
        X = []
        samples = self.samples()
        for key in samples:
            X += [self.shape(key)]
        return X

if __name__ == '__main__':
    debugger = test_debugger_html(ClassifyingDebugger())
    debugger.X()

class ClassifyingDebugger(ClassifyingDebugger):
    def Y(self) -> List[float]:
        Y = []
        samples = self.samples()
        for key in samples:
            Y += [samples[key]]
        return Y

if __name__ == '__main__':
    debugger = test_debugger_html(ClassifyingDebugger())
    debugger.Y()

from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz

class ClassifyingDebugger(ClassifyingDebugger):
    def classifier(self) -> DecisionTreeClassifier:
        classifier = DecisionTreeClassifier()
        classifier = classifier.fit(self.X(), self.Y())
        return classifier

import graphviz

class ClassifyingDebugger(ClassifyingDebugger):
    def show_classifier(self, classifier: DecisionTreeClassifier) -> Any:
        dot_data = export_graphviz(classifier, out_file=None, 
                                   filled=False, rounded=True,
                                   feature_names=self.feature_names(),
                                   class_names=["FAIL", "PASS"],
                                   label='none',
                                   node_ids=False,
                                   impurity=False,
                                   proportion=True,
                                   special_characters=True)

        return graphviz.Source(dot_data)

if __name__ == '__main__':
    debugger = test_debugger_html(ClassifyingDebugger())
    classifier = debugger.classifier()
    debugger.show_classifier(classifier)

if __name__ == '__main__':
    classifier.predict([[1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1]])

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



### Collecting Events from Calls

if __name__ == '__main__':
    print('\n### Collecting Events from Calls')



if __name__ == '__main__':
    debugger = TarantulaDebugger()
    with debugger.collect_pass():
        remove_html_markup("abc")
    with debugger.collect_pass():
        remove_html_markup('<b>abc</b>')
    with debugger.collect_fail():
        remove_html_markup('"abc"')

### Collecting Events from Tests

if __name__ == '__main__':
    print('\n### Collecting Events from Tests')



if __name__ == '__main__':
    debugger = TarantulaDebugger()
    with debugger:
        remove_html_markup("abc")
    with debugger:
        remove_html_markup('<b>abc</b>')
    with debugger:
        remove_html_markup('"abc"')
        assert False  # raise an exception

### Visualizing Events as a Table

if __name__ == '__main__':
    print('\n### Visualizing Events as a Table')



if __name__ == '__main__':
    debugger.event_table(args=True, color=True)

### Visualizing Suspicious Code

if __name__ == '__main__':
    print('\n### Visualizing Suspicious Code')



if __name__ == '__main__':
    debugger

### Ranking Events

if __name__ == '__main__':
    print('\n### Ranking Events')



if __name__ == '__main__':
    debugger.rank()

### Classes and Methods

if __name__ == '__main__':
    print('\n### Classes and Methods')



from .ClassDiagram import display_class_hierarchy

if __name__ == '__main__':
    display_class_hierarchy([TarantulaDebugger, OchiaiDebugger],
                            abstract_classes=[
                                StatisticalDebugger,
                                DifferenceDebugger,
                                RankingDebugger
                            ],
                            public_methods=[
                                StatisticalDebugger.__init__,
                                StatisticalDebugger.all_events,
                                StatisticalDebugger.event_table,
                                StatisticalDebugger.function,
                                StatisticalDebugger.coverage,
                                StatisticalDebugger.covered_functions,
                                DifferenceDebugger.__enter__,
                                DifferenceDebugger.__exit__,
                                DifferenceDebugger.all_pass_events,
                                DifferenceDebugger.all_fail_events,
                                DifferenceDebugger.collect_pass,
                                DifferenceDebugger.collect_fail,
                                DifferenceDebugger.only_pass_events,
                                DifferenceDebugger.only_fail_events,
                                SpectrumDebugger.code,
                                SpectrumDebugger.__repr__,
                                SpectrumDebugger.__str__,
                                SpectrumDebugger._repr_html_,
                                ContinuousSpectrumDebugger.code,
                                ContinuousSpectrumDebugger.__repr__,
                                RankingDebugger.rank
                            ],
                            project='debuggingbook')

if __name__ == '__main__':
    display_class_hierarchy([CoverageCollector, ValueCollector],
                            public_methods=[
                                Tracer.__init__,
                                Tracer.__enter__,
                                Tracer.__exit__,
                                Tracer.changed_vars,  # type: ignore
                                Collector.__init__,
                                Collector.__repr__,
                                Collector.function,
                                Collector.args,
                                Collector.argstring,
                                Collector.exception,
                                Collector.id,
                                Collector.collect,
                                CoverageCollector.coverage,
                                CoverageCollector.covered_functions,
                                CoverageCollector.events,
                                ValueCollector.__init__,
                                ValueCollector.events
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



### Exercise 1: A Postcondition for Middle

if __name__ == '__main__':
    print('\n### Exercise 1: A Postcondition for Middle')



def middle_checked(x, y, z):  # type: ignore
    m = middle(x, y, z)
    assert m == sorted([x, y, z])[1]
    return m

from .ExpectError import ExpectError

if __name__ == '__main__':
    with ExpectError():
        m = middle_checked(2, 1, 3)

### Exercise 2: Statistical Dependencies

if __name__ == '__main__':
    print('\n### Exercise 2: Statistical Dependencies')



### Exercise 3: Correlating with Conditions

if __name__ == '__main__':
    print('\n### Exercise 3: Correlating with Conditions')



#### Part 1: Experiment

if __name__ == '__main__':
    print('\n#### Part 1: Experiment')



#### Part 2: Collecting Conditions

if __name__ == '__main__':
    print('\n#### Part 2: Collecting Conditions')


