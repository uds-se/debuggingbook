#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Debugging Book".
# Web site: https://www.debuggingbook.org/html/StatisticalDebugger.html
# Last change: 2021-01-23 13:19:38+01:00
#
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


# # Statistical Debugging

if __name__ == "__main__":
    print('# Statistical Debugging')




if __name__ == "__main__":
    from bookutils import YouTubeVideo
    YouTubeVideo("UNuso00zYiI")


if __name__ == "__main__":
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




# ## Introduction

if __name__ == "__main__":
    print('\n## Introduction')




# ## Collecting Events

if __name__ == "__main__":
    print('\n## Collecting Events')




if __package__ is None or __package__ == "":
    from Tracer import Tracer
else:
    from .Tracer import Tracer


class Collector(Tracer):
    """A class to record events during execution."""

    def collect(self, frame, event, arg):
        """Collecting function. To be overridden in subclasses."""
        pass

    def events(self):
        """Return a collection of events. To be overridden in subclasses."""
        return set()

    def traceit(self, frame, event, arg):
        self.collect(frame, event, arg)

def remove_html_markup(s):
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

if __name__ == "__main__":
    with Collector() as c:
        out = remove_html_markup('"abc"')
    out


from types import FunctionType

class Collector(Collector):
    def __init__(self):
        """Constructor."""
        self._function = None
        self._args = None
        self._argstring = None
        self._exception = None

    def traceit(self, frame, event, arg):
        """Tracing function. Saves the first function and calls collect()."""
        if self._function is None and event == 'call':
            # Save function
            self._function = FunctionType(frame.f_code,
                                          globals=frame.f_globals,
                                          name=frame.f_code.co_name)
            self._args = frame.f_locals.copy()
            self._argstring = ", ".join([f"{var}={repr(self._args[var])}" 
                                         for var in self._args])

        self.collect(frame, event, arg)
        
    def collect(self, frame, event, arg):
        """Collector function. To be overloaded in subclasses."""
        pass

    def id(self):
        """Return an identifier for the collector, 
        created from the first call"""
        return f"{self._function.__name__}({self.argstring()})"

    def function(self):
        """Return the function from the first call, 
        as a function object"""
        return self._function

    def argstring(self):
        """Return the list of arguments from the first call, 
        as a printable string"""
        return self._argstring

    def args(self):
        """Return a dict of argument names and values from the first call"""
        return self._args

    def exception(self):
        """Return the exception class from the first call,
        or None if no exception was raised."""
        return self._exception

    def __repr__(self):
        """Return a string representation of the collector"""
        # We use the ID as default representation when printed
        return self.id()

if __name__ == "__main__":
    with Collector() as c:
        remove_html_markup('abc')
    c.function(), c.id()


# ## Collecting Coverage

if __name__ == "__main__":
    print('\n## Collecting Coverage')




from types import FunctionType

class CoverageCollector(Collector):
    """A class to record covered locations during execution."""

    def __init__(self):
        super().__init__()
        self._coverage = set()

    def collect(self, frame, event, arg):
        """Save coverage for an observed event."""
        name = frame.f_code.co_name
        if name in frame.f_globals:
            # Access exactly this function
            function = frame.f_globals[name]
        else:
            # Create new function from given code
            function = FunctionType(frame.f_code,
                                    globals=frame.f_globals,
                                    name=name)

        location = (function, frame.f_lineno)
        self._coverage.add(location)

class CoverageCollector(CoverageCollector):
    def events(self):
        """Return the set of locations covered.
        Each location comes as a pair (`function_name`, `lineno`)."""
        return {(func.__name__, lineno) for func, lineno in self._coverage}

class CoverageCollector(CoverageCollector):
    def covered_functions(self):
        """Return a set with all functions covered."""
        return {func for func, lineno in self._coverage}

    def coverage(self):
        """Return a set (function, lineno) with all locations covered."""
        return self._coverage

if __name__ == "__main__":
    with CoverageCollector() as c:
        remove_html_markup('abc')
    c.events()


import inspect

if __package__ is None or __package__ == "":
    from bookutils import getsourcelines    # like inspect.getsourcelines(), but in color
else:
    from .bookutils import getsourcelines    # like inspect.getsourcelines(), but in color


def code_with_coverage(function, coverage):
    source_lines, starting_line_number = \
       getsourcelines(function)

    line_number = starting_line_number
    for line in source_lines:
        marker = '*' if (function, line_number) in coverage else ' '
        print(f"{line_number:4} {marker} {line}", end='')
        line_number += 1

if __name__ == "__main__":
    code_with_coverage(remove_html_markup, c.coverage())


if __package__ is None or __package__ == "":
    from bookutils import quiz
else:
    from .bookutils import quiz


if __name__ == "__main__":
    quiz('Let the input be `"<b>Don\'t do this!</b>"`. '
         "Which of these lines are executed? Use the code to find out!",
         [
             "`tag = True`",
             "`tag = False`",
             "`quote = not quote`",
             "`out = out + c`"
         ], [ord(c) - ord('a') - 1 for c in 'cdf'])


if __name__ == "__main__":
    with CoverageCollector() as c:
        remove_html_markup("<b>Don't do this!</b>")
    # code_with_coverage(remove_html_markup, c.coverage)


# ## Computing Differences

if __name__ == "__main__":
    print('\n## Computing Differences')




# ### A Base Class for Statistical Debugging

if __name__ == "__main__":
    print('\n### A Base Class for Statistical Debugging')




class StatisticalDebugger():
    """A class to collect events for multiple outcomes."""

    def __init__(self, collector_class=CoverageCollector, log=False):
        """Constructor. Use instances of `collector_class` to collect events."""
        self.collector_class = collector_class
        self.collectors = {}
        self.log = log

class StatisticalDebugger(StatisticalDebugger):
    def collect(self, outcome, *args, **kwargs):
        """Return a collector for the given outcome. 
        Additional args are passed to the collector."""
        collector = self.collector_class(*args, **kwargs)
        return self.add_collector(outcome, collector)

    def add_collector(self, outcome, collector):
        if outcome not in self.collectors:
            self.collectors[outcome] = []
        self.collectors[outcome].append(collector)
        return collector

class StatisticalDebugger(StatisticalDebugger):
    def all_events(self, outcome=None):
        """Return a set of all events observed."""
        all_events = set()
        if outcome:
            for collector in self.collectors[outcome]:
                all_events.update(collector.events())
        else:
            for outcome in self.collectors:
                for collector in self.collectors[outcome]:
                    all_events.update(collector.events())
        return all_events

if __name__ == "__main__":
    s = StatisticalDebugger()
    with s.collect('PASS'):
        remove_html_markup("abc")
    with s.collect('PASS'):
        remove_html_markup('<b>abc</b>')
    with s.collect('FAIL'):
        remove_html_markup('"abc"')


if __name__ == "__main__":
    s.all_events()


if __name__ == "__main__":
    s.all_events('FAIL')


if __name__ == "__main__":
    s.collectors


if __name__ == "__main__":
    s.collectors['PASS'][0].id()


if __name__ == "__main__":
    s.collectors['PASS'][0].events()


# ### Excursion: Printing an Event Table

if __name__ == "__main__":
    print('\n### Excursion: Printing an Event Table')




if __name__ == "__main__":
    from IPython.display import display, Markdown, HTML


import html

class StatisticalDebugger(StatisticalDebugger):
    def function(self):
        """Return the entry function from the events observed,
           or None if ambiguous"""
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

    def covered_functions(self):
        """Return a set of all functions observed"""
        functions = set()
        for outcome in self.collectors:
            for collector in self.collectors[outcome]:
                functions |= collector.covered_functions()
        return functions
    
    def coverage(self):
        """Return a set of all (functions, line_numbers) observed"""
        coverage = set()
        for outcome in self.collectors:
            for collector in self.collectors[outcome]:
                coverage |= collector.coverage()
        return coverage

    def color(self, event):
        """Return a color for the given event, or None.
           To be overloaded in subclasses."""
        return None

    def tooltip(self, event):
        """Return a tooltip string for the given event, or None.
           To be overloaded in subclasses."""
        return None

    def event_str(self, event):
        """Format the given event. To be overloaded in subclasses."""
        if isinstance(event, str):
            return event
        if isinstance(event, tuple):
            return ":".join(self.event_str(elem) for elem in event)
        return str(event)

    def event_table_text(self, args=False, color=False):
        """Print out a table of events observed.
           If args is set, use arguments as headers.
           If color is set, use colors."""
        sep = ' | '

        all_events = self.all_events()
        longest_event = max(len(f"{self.event_str(event)}") 
                            for event in all_events)

        out = ""

        # Header
        if args:
            out += '| ' + '`' + self.function().__name__ + '`' + sep
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

    def event_table(self, **_args):
        """Print out event table in Markdown format."""
        return Markdown(self.event_table_text(**_args))

    def __repr__(self):
        return self._event_table_text()

# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




if __name__ == "__main__":
    s = StatisticalDebugger()
    with s.collect('PASS'):
        remove_html_markup("abc")
    with s.collect('PASS'):
        remove_html_markup('<b>abc</b>')
    with s.collect('FAIL'):
        remove_html_markup('"abc"')


if __name__ == "__main__":
    s.event_table(args=True)


if __name__ == "__main__":
    quiz("How many lines are executed in the failing run only?",
        ["One", "Two", "Three"], int(chr(50)))


# ### Collecting Passing and Failing Runs

if __name__ == "__main__":
    print('\n### Collecting Passing and Failing Runs')




class DifferenceDebugger(StatisticalDebugger):
    """A class to collect events for passing and failing outcomes."""
    PASS = 'PASS'
    FAIL = 'FAIL'

    def collect_pass(self, *args, **kwargs):
        """Return a collector for passing runs."""
        return self.collect(self.PASS, *args, **kwargs)

    def collect_fail(self, *args, **kwargs):
        """Return a collector for failing runs."""
        return self.collect(self.FAIL, *args, **kwargs)

    def pass_collectors(self):
        return self.collectors[self.PASS]

    def fail_collectors(self):
        return self.collectors[self.FAIL]

def test_debugger_html(debugger):
    with debugger.collect_pass():
        remove_html_markup('abc')
    with debugger.collect_pass():
        remove_html_markup('<b>abc</b>')
    with debugger.collect_fail():
        remove_html_markup('"abc"')
    return debugger

class DifferenceDebugger(DifferenceDebugger):
    def __enter__(self):
        """Enter a `with` block. Collect coverage and outcome;
        classify as FAIL if the block raises an exception,
        and PASS if it does not.
        """
        self.collector = self.collector_class()
        self.collector.__enter__()
        return self

    def __exit__(self, exc_tp, value, traceback):
        """Exit the `with` block."""
        self.collector.__exit__(exc_tp, value, traceback)
        if exc_tp is None:
            outcome = self.PASS
        else:
            outcome = self.FAIL
        self.add_collector(outcome, self.collector)
        return True  # Ignore exception

def test_debugger_html(debugger):
    with debugger:
        remove_html_markup('abc')
    with debugger:
        remove_html_markup('<b>abc</b>')
    with debugger:
        remove_html_markup('"abc"')
        assert False  # Mark test as failing

    return debugger

# ### Analyzing Events

if __name__ == "__main__":
    print('\n### Analyzing Events')




if __name__ == "__main__":
    debugger = test_debugger_html(DifferenceDebugger())


if __name__ == "__main__":
    pass_1_events = debugger.pass_collectors()[0].events()


if __name__ == "__main__":
    pass_2_events = debugger.pass_collectors()[1].events()


if __name__ == "__main__":
    in_any_pass = pass_1_events | pass_2_events
    in_any_pass


if __name__ == "__main__":
    fail_events = debugger.fail_collectors()[0].events()


if __name__ == "__main__":
    only_in_fail = fail_events - in_any_pass
    only_in_fail


if __name__ == "__main__":
    code_with_coverage(remove_html_markup, only_in_fail)


class DifferenceDebugger(DifferenceDebugger):
    def all_fail_events(self):
        """Return all events observed in failing runs."""
        return self.all_events(self.FAIL)

    def all_pass_events(self):
        """Return all events observed in passing runs."""
        return self.all_events(self.PASS)

class DifferenceDebugger(DifferenceDebugger):
    def only_fail_events(self):
        """Return all events observed only in failing runs."""
        return self.all_fail_events() - self.all_pass_events()

    def only_pass_events(self):
        """Return all events observed only in passing runs."""
        return self.all_pass_events() - self.all_fail_events()

if __name__ == "__main__":
    debugger = test_debugger_html(DifferenceDebugger())


if __name__ == "__main__":
    debugger.all_events()


if __name__ == "__main__":
    debugger.only_fail_events()


if __name__ == "__main__":
    debugger.only_pass_events()


# ## Visualizing Differences

if __name__ == "__main__":
    print('\n## Visualizing Differences')




# ### Discrete Spectrum

if __name__ == "__main__":
    print('\n### Discrete Spectrum')




class DiscreteSpectrumDebugger(DifferenceDebugger):
    def suspiciousness(self, event):
        """Return a suspiciousness value [0, 1.0]
        for the given event, or `None` if unknown"""
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

    def color(self, event):
        """Return a color for the given event."""
        suspiciousness = self.suspiciousness(event)
        if suspiciousness is None:
            return None

        if suspiciousness > 0.8:
            return 'mistyrose'
        if suspiciousness >= 0.5:
            return 'lightyellow'

        return 'honeydew'

    def tooltip(self, event):
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

    def percentage(self, event):
        """Return the suspiciousness for the given event as percentage string"""
        suspiciousness = self.suspiciousness(event)
        if suspiciousness is not None:
            return str(int(suspiciousness * 100)).rjust(3) + '%'
        else:
            return ' ' * len('100%')

class DiscreteSpectrumDebugger(DiscreteSpectrumDebugger):
    def code(self, functions=None, color=False, suspiciousness=False,
             line_numbers=True):
        """Print a listing of `functions` (default: covered functions).
           If `color` is set, render as HTML, using suspiciousness colors.
           If `suspiciousness` is set, include suspiciousness values.
           If `line_numbers` is set, include line numbers.
           """

        if not functions:
            functions = self.covered_functions()

        out = ""
        seen = set()
        for function in functions:
            if out:
                out += '\n'
                if color:
                    out += '<p/>'

            source_lines, starting_line_number = \
               inspect.getsourcelines(function)

            if (function.__name__, starting_line_number) in seen:
                continue
            seen.add((function.__name__, starting_line_number))

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

    def _repr_html_(self):
        """When output in Jupyter, visualize as HTML"""
        return self.code(color=True)

    def __str__(self):
        """Show code as string"""
        return self.code(color=False, suspiciousness=True)

    def __repr__(self):
        """Show code as string"""
        return self.code(color=False, suspiciousness=True)

if __name__ == "__main__":
    debugger = test_debugger_html(DiscreteSpectrumDebugger())


if __name__ == "__main__":
    debugger


if __name__ == "__main__":
    quiz("Does the line `quote = not quote` actually contain the defect?",
        [
            "Yes, it should be fixed",
            "No, the defect is elsewhere"
        ],
         164 * 2 % 326
        )


if __name__ == "__main__":
    print(debugger)


# ### Continuous Spectrum

if __name__ == "__main__":
    print('\n### Continuous Spectrum')




if __name__ == "__main__":
    remove_html_markup('<b color="blue">text</b>')


if __name__ == "__main__":
    debugger = test_debugger_html(DiscreteSpectrumDebugger())
    with debugger.collect_pass():
        remove_html_markup('<b link="blue"></b>')


if __name__ == "__main__":
    debugger.only_fail_events()


if __name__ == "__main__":
    debugger


class ContinuousSpectrumDebugger(DiscreteSpectrumDebugger):
    def collectors_with_event(self, event, category):
        """Return all collectors in a category
        that observed the given event."""
        all_runs = self.collectors[category]
        collectors_with_event = set(collector for collector in all_runs 
                              if event in collector.events())
        return collectors_with_event

    def collectors_without_event(self, event, category):
        """Return all collectors in a category
        that did not observe the given event."""
        all_runs = self.collectors[category]
        collectors_without_event = set(collector for collector in all_runs 
                              if event not in collector.events())
        return collectors_without_event

    def event_fraction(self, event, category):
        all_collectors = self.collectors[category]
        collectors_with_event = self.collectors_with_event(event, category)
        fraction = len(collectors_with_event) / len(all_collectors)
        # print(f"%{category}({event}) = {fraction}")
        return fraction

    def passed_fraction(self, line_number):
        return self.event_fraction(line_number, self.PASS)

    def failed_fraction(self, line_number):
        return self.event_fraction(line_number, self.FAIL)

    def hue(self, line_number):
        """Return a color hue from 0.0 (red) to 1.0 (green)."""
        passed = self.passed_fraction(line_number)
        failed = self.failed_fraction(line_number)
        if passed + failed > 0:
            return passed / (passed + failed)
        else:
            return None

class ContinuousSpectrumDebugger(ContinuousSpectrumDebugger):
    def suspiciousness(self, event):
        hue = self.hue(event)
        if hue is None:
            return None
        return 1 - hue
    
    def tooltip(self, event):
        return self.percentage(event)

if __name__ == "__main__":
    debugger = test_debugger_html(ContinuousSpectrumDebugger())


if __name__ == "__main__":
    for line in debugger.only_fail_events():
        print(line, debugger.hue(line))


if __name__ == "__main__":
    for line in debugger.only_pass_events():
        print(line, debugger.hue(line))


class ContinuousSpectrumDebugger(ContinuousSpectrumDebugger):
    def brightness(self, line):
        return max(self.passed_fraction(line), self.failed_fraction(line))

if __name__ == "__main__":
    debugger = test_debugger_html(ContinuousSpectrumDebugger())
    for line in debugger.only_fail_events():
        print(line, debugger.brightness(line))


class ContinuousSpectrumDebugger(ContinuousSpectrumDebugger):
    def color(self, event):
        hue = self.hue(event)
        if hue is None:
            return None
        saturation = self.brightness(event)

        # HSL color values are specified with: 
        # hsl(hue, saturation, lightness).
        return f"hsl({hue * 120}, {saturation * 100}%, 80%)"

if __name__ == "__main__":
    debugger = test_debugger_html(ContinuousSpectrumDebugger())


if __name__ == "__main__":
    for location in debugger.only_fail_events():
        print(location, debugger.color(line))


if __name__ == "__main__":
    for location in debugger.only_pass_events():
        print(location, debugger.color(line))


if __name__ == "__main__":
    debugger


if __name__ == "__main__":
    with debugger.collect_pass():
        out = remove_html_markup('<b link="blue"></b>')


if __name__ == "__main__":
    quiz('In which color will the `quote = not quote` "culprit" line '
         'be shown after executing the above code?',
        [
            '<span style="background-color: hsl(120.0, 50.0%, 80%)">Green</span>',
            '<span style="background-color: hsl(60.0, 100.0%, 80%)">Yellow</span>',
            '<span style="background-color: hsl(30.0, 100.0%, 80%)">Orange</span>',
            '<span style="background-color: hsl(0.0, 100.0%, 80%)">Red</span>'
        ],
         999 / 333
        )


if __name__ == "__main__":
    debugger


def middle(x, y, z):
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

if __name__ == "__main__":
    middle(1, 2, 3)


if __name__ == "__main__":
    middle(2, 1, 3)


def test_debugger_middle(debugger):
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

if __name__ == "__main__":
    debugger = test_debugger_middle(ContinuousSpectrumDebugger())


if __name__ == "__main__":
    debugger.event_table(args=True)


if __name__ == "__main__":
    debugger


if __name__ == "__main__":
    quiz("Which of the above lines should be fixed?",
        [
            '<span style="background-color: hsl(45.0, 100%, 80%)">Line 3: `elif x < y`</span>',
            '<span style="background-color: hsl(34.28571428571429, 100.0%, 80%)">Line 5: `elif x < z`</span>',
            '<span style="background-color: hsl(20.000000000000004, 100.0%, 80%)">Line 6: `return y`</span>',
            '<span style="background-color: hsl(120.0, 20.0%, 80%)">Line 9: `return y`</span>',
        ],
         len(" middle \n".strip()[:3])
        )


def middle_fixed(x, y, z):
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

if __name__ == "__main__":
    middle_fixed(2, 1, 3)


# ## Ranking Lines by Suspiciousness

if __name__ == "__main__":
    print('\n## Ranking Lines by Suspiciousness')




class RankingDebugger(DifferenceDebugger):
    def rank(self):
        """Return a list of events, sorted by suspiciousness, highest first."""
        events = list(self.all_events())
        events.sort(key=self.suspiciousness, reverse=True)
        return events

    def __repr__(self):
        return repr(self.rank())

# ### The Tarantula Metric

if __name__ == "__main__":
    print('\n### The Tarantula Metric')




class TarantulaDebugger(ContinuousSpectrumDebugger, RankingDebugger):
    pass

if __name__ == "__main__":
    tarantula_html = test_debugger_html(TarantulaDebugger())


if __name__ == "__main__":
    tarantula_html


if __name__ == "__main__":
    tarantula_html.rank()


if __name__ == "__main__":
    tarantula_html.suspiciousness(tarantula_html.rank()[0])


if __name__ == "__main__":
    tarantula_middle = test_debugger_middle(TarantulaDebugger())


if __name__ == "__main__":
    tarantula_middle


if __name__ == "__main__":
    tarantula_middle.rank()


if __name__ == "__main__":
    tarantula_middle.suspiciousness(tarantula_middle.rank()[0])


# ### The Ochiai Metric

if __name__ == "__main__":
    print('\n### The Ochiai Metric')




import math

class OchiaiDebugger(ContinuousSpectrumDebugger, RankingDebugger):
    def suspiciousness(self, event):
        failed = len(self.collectors_with_event(event, self.FAIL))
        not_in_failed = len(self.collectors_without_event(event, self.FAIL))
        passed = len(self.collectors_with_event(event, self.PASS))

        try:
            return failed / math.sqrt((failed + not_in_failed) * (failed + passed))
        except ZeroDivisionError:
            return None

    def hue(self, event):
        suspiciousness = self.suspiciousness(event)
        if suspiciousness is None:
            return None
        return 1 - suspiciousness

if __name__ == "__main__":
    ochiai_html = test_debugger_html(OchiaiDebugger())


if __name__ == "__main__":
    ochiai_html


if __name__ == "__main__":
    ochiai_html.rank()


if __name__ == "__main__":
    ochiai_html.suspiciousness(ochiai_html.rank()[0])


if __name__ == "__main__":
    ochiai_middle = test_debugger_middle(OchiaiDebugger())


if __name__ == "__main__":
    ochiai_middle


if __name__ == "__main__":
    ochiai_middle.rank()


if __name__ == "__main__":
    ochiai_middle.suspiciousness(ochiai_middle.rank()[0])


# ### How Useful is Ranking?

if __name__ == "__main__":
    print('\n### How Useful is Ranking?')




# ## Other Events besides Coverage

if __name__ == "__main__":
    print('\n## Other Events besides Coverage')




class ValueCollector(Collector):
    """"A class to collect local variables and their values."""

    def __init__(self):
        """Constructor."""
        super().__init__()
        self.vars = set()

    def collect(self, frame, event, arg):
        local_vars = frame.f_locals
        for var in local_vars:
            value = local_vars[var]
            self.vars.add(f"{var} = {repr(value)}")

    def events(self):
        """A set of (variable, value) pairs observed"""
        return self.vars

if __name__ == "__main__":
    debugger = test_debugger_html(ContinuousSpectrumDebugger(ValueCollector))
    for event in debugger.all_events():
        print(event)


if __name__ == "__main__":
    for event in debugger.only_fail_events():
        print(event)


if __name__ == "__main__":
    debugger.event_table(color=True, args=True)


# ## Training Classifiers

if __name__ == "__main__":
    print('\n## Training Classifiers')




class ClassifyingDebugger(DifferenceDebugger):
    """A debugger implementing a decision tree for events"""

    PASS_VALUE = +1
    FAIL_VALUE = -1

    def samples(self):
        samples = {}
        for collector in self.pass_collectors():
            samples[collector.id()] = self.PASS_VALUE
        for collector in debugger.fail_collectors():
            samples[collector.id()] = self.FAIL_VALUE
        return samples

if __name__ == "__main__":
    debugger = test_debugger_html(ClassifyingDebugger())
    debugger.samples()


class ClassifyingDebugger(ClassifyingDebugger):
    def features(self):
        features = {}
        for collector in debugger.pass_collectors():
            features[collector.id()] = collector.events()
        for collector in debugger.fail_collectors():
            features[collector.id()] = collector.events()
        return features

if __name__ == "__main__":
    debugger = test_debugger_html(ClassifyingDebugger())
    debugger.features()


class ClassifyingDebugger(ClassifyingDebugger):
    def feature_names(self):
        return [repr(feature) for feature in self.all_events()]

if __name__ == "__main__":
    debugger = test_debugger_html(ClassifyingDebugger())
    debugger.feature_names()


class ClassifyingDebugger(ClassifyingDebugger):
    def shape(self, sample):
        x = []
        features = self.features()
        for f in self.all_events():
            if f in features[sample]:
                x += [+1]
            else:
                x += [-1]
        return x

if __name__ == "__main__":
    debugger = test_debugger_html(ClassifyingDebugger())
    debugger.shape("remove_html_markup(s='abc')")


class ClassifyingDebugger(ClassifyingDebugger):
    def X(self):
        X = []
        samples = self.samples()
        for key in samples:
            X += [self.shape(key)]
        return X

if __name__ == "__main__":
    debugger = test_debugger_html(ClassifyingDebugger())
    debugger.X()


class ClassifyingDebugger(ClassifyingDebugger):
    def Y(self):
        Y = []
        samples = self.samples()
        for key in samples:
            Y += [samples[key]]
        return Y

if __name__ == "__main__":
    debugger = test_debugger_html(ClassifyingDebugger())
    debugger.Y()


from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz

class ClassifyingDebugger(ClassifyingDebugger):
    def classifier(self):
        classifier = DecisionTreeClassifier()
        classifier = classifier.fit(self.X(), self.Y())
        return classifier

import graphviz

class ClassifyingDebugger(ClassifyingDebugger):
    def show_classifier(self, classifier):
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

if __name__ == "__main__":
    debugger = test_debugger_html(ClassifyingDebugger())
    classifier = debugger.classifier()
    debugger.show_classifier(classifier)


if __name__ == "__main__":
    classifier.predict([[1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1]])


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




# ### Collecting Events from Calls

if __name__ == "__main__":
    print('\n### Collecting Events from Calls')




if __name__ == "__main__":
    debugger = TarantulaDebugger()
    with debugger.collect_pass():
        remove_html_markup("abc")
    with debugger.collect_pass():
        remove_html_markup('<b>abc</b>')
    with debugger.collect_fail():
        remove_html_markup('"abc"')


# ### Collecting Events from Tests

if __name__ == "__main__":
    print('\n### Collecting Events from Tests')




if __name__ == "__main__":
    debugger = TarantulaDebugger()
    with debugger:
        remove_html_markup("abc")
    with debugger:
        remove_html_markup('<b>abc</b>')
    with debugger:
        remove_html_markup('"abc"')
        assert False  # raise an exception


# ### Visualizing Events as a Table

if __name__ == "__main__":
    print('\n### Visualizing Events as a Table')




if __name__ == "__main__":
    debugger.event_table(args=True, color=True)


# ### Visualizing Suspicious Code

if __name__ == "__main__":
    print('\n### Visualizing Suspicious Code')




if __name__ == "__main__":
    debugger


# ### Ranking Events

if __name__ == "__main__":
    print('\n### Ranking Events')




if __name__ == "__main__":
    debugger.rank()


# ### Classes and Methods

if __name__ == "__main__":
    print('\n### Classes and Methods')




if __package__ is None or __package__ == "":
    from ClassDiagram import display_class_hierarchy
else:
    from .ClassDiagram import display_class_hierarchy


if __name__ == "__main__":
    display_class_hierarchy([TarantulaDebugger, OchiaiDebugger],
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
                                DiscreteSpectrumDebugger.code,
                                DiscreteSpectrumDebugger.__repr__,
                                DiscreteSpectrumDebugger._repr_html_,
                                ContinuousSpectrumDebugger.code,
                                ContinuousSpectrumDebugger.__repr__,
                                RankingDebugger.rank
                            ],
                            project='debuggingbook')


if __name__ == "__main__":
    display_class_hierarchy([CoverageCollector, ValueCollector],
                            public_methods=[
                                Tracer.__init__,
                                Tracer.__enter__,
                                Tracer.__exit__,
                                Tracer.changed_vars,
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


# ## Lessons Learned

if __name__ == "__main__":
    print('\n## Lessons Learned')




# ## Next Steps

if __name__ == "__main__":
    print('\n## Next Steps')




# ## Background

if __name__ == "__main__":
    print('\n## Background')




# ## Exercises

if __name__ == "__main__":
    print('\n## Exercises')




# ### Exercise 1: A Postcondition for Middle

if __name__ == "__main__":
    print('\n### Exercise 1: A Postcondition for Middle')




def middle_checked(x, y, z):
    m = middle(x, y, z)
    assert m == sorted([x, y, z])[1]
    return m

if __package__ is None or __package__ == "":
    from ExpectError import ExpectError
else:
    from .ExpectError import ExpectError


if __name__ == "__main__":
    with ExpectError():
        m = middle_checked(2, 1, 3)


# ### Exercise 2: Statistical Dependencies

if __name__ == "__main__":
    print('\n### Exercise 2: Statistical Dependencies')



