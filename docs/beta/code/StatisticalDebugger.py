#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Fuzzing Book".
# Web site: https://www.fuzzingbook.org/html/StatisticalDebugger.html
# Last change: 2020-11-16 20:07:52+01:00
#
#!/
# Copyright (c) 2018-2020 CISPA, Saarland University, authors, and contributors
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
    c = Collector()
    with c:
        out = remove_html_markup('"abc"')
    out


class Collector(Collector):
    def __init__(self):
        self._id = None

    def traceit(self, frame, event, arg):
        if self._id is None and event == 'call':
            # Save ID
            function = frame.f_code.co_name
            locals = frame.f_locals
            args = ", ".join([f"{var}={repr(locals[var])}" for var in locals])
            self._id = f"{function}({args})"

        self.collect(frame, event, arg)

    def id(self):
        """Return an identifier for the collector, created from the first call"""
        return self._id

if __name__ == "__main__":
    c = Collector()
    with c:
        remove_html_markup('abc')
    c.id()


# ## Collecting Coverage

if __name__ == "__main__":
    print('\n## Collecting Coverage')




class CoverageCollector(Collector):
    """A class to record covered lines during execution."""

    def __init__(self):
        super().__init__()
        self.coverage = set()

    def collect(self, frame, event, arg):
        self.coverage.add(frame.f_lineno)

class CoverageCollector(CoverageCollector):
    def events(self):
        """Return a set of predicates holding for the execution"""
        return self.coverage

if __name__ == "__main__":
    c = CoverageCollector()
    with c:
        remove_html_markup('abc')
    print(c.events())


import inspect

if __package__ is None or __package__ == "":
    from bookutils import getsourcelines    # like inspect.getsourcelines(), but in color
else:
    from .bookutils import getsourcelines    # like inspect.getsourcelines(), but in color


def list_with_coverage(function, coverage):
    source_lines, starting_line_number = \
       getsourcelines(function)

    line_number = starting_line_number
    for line in source_lines:
        marker = '*' if line_number in coverage else ' '
        print(f"{line_number:4} {marker} {line}", end='')
        line_number += 1

if __name__ == "__main__":
    list_with_coverage(remove_html_markup, c.coverage)


if __package__ is None or __package__ == "":
    from bookutils import quiz
else:
    from .bookutils import quiz


if __name__ == "__main__":
    quiz("Let the input be <code>&quot;&lt;b&gt;Don't do this!&lt;/b&gt;&quot;</code>. "
         "Which of these lines are executed? Use the code to find out!",
         [
             "<code>tag = True</code>",
             "<code>tag = False</code>",
             "<code>quote = not quote</code>",
             "<code>out = out + c</code>"
         ], [ord(c) - ord('a') - 1 for c in 'cdf'])


if __name__ == "__main__":
    c = CoverageCollector()
    with c:
        remove_html_markup("<b>Don't do this!</b>")
    # list_with_coverage(remove_html_markup, c.coverage)


# ## Computing Differences

if __name__ == "__main__":
    print('\n## Computing Differences')




# ### A Base Class for Statistical Debugging

if __name__ == "__main__":
    print('\n### A Base Class for Statistical Debugging')




class StatisticalDebugger():
    """A class to collect events for multiple outcomes."""

    def __init__(self, collector_class):
        self.collector_class = collector_class
        self.collectors = {}

class StatisticalDebugger(StatisticalDebugger):
    def collect(self, outcome, *args):
        """Return a collector for the given outcome. 
        Additional args are passed to the collector."""
        collector = self.collector_class(*args)
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
    s = StatisticalDebugger(CoverageCollector)
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
    def color(self, event):
        """Return a color for the given event, or None. To be overloaded in subclasses."""
        return None

    def event_table(self, ids=False, color=False):
        """Print out a table of events observed."""
        sep = ' | '

        all_events = self.all_events()
        longest_event = max(len(f"{event}") for event in all_events)

        out = ""

        # Header
        if ids:
            out += '| ' + ' ' * longest_event + sep
            for name in self.collectors:
                for collector in self.collectors[name]:
                    out += '`' + collector.id() + '`' + sep
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
        for event in all_events:
            event_name = str(event).rjust(longest_event)  # could also use repr(event)

            if color:
                color_name = self.color(event)
                if color_name:
                    event_name = \
                        f'<samp style="background-color: {color_name}">{html.escape(event_name)}</samp>'

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

        return Markdown(out)

# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




if __name__ == "__main__":
    s = StatisticalDebugger(CoverageCollector)
    with s.collect('PASS'):
        remove_html_markup("abc")
    with s.collect('PASS'):
        remove_html_markup('<b>abc</b>')
    with s.collect('FAIL'):
        remove_html_markup('"abc"')
    s.event_table(ids=True)


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

    def collect_pass(self, *args):
        """Return a collector for passing runs."""
        return self.collect(self.PASS, *args)
    
    def collect_fail(self, *args):
        """Return a collector for failing runs."""
        return self.collect(self.FAIL, *args)

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

if __name__ == "__main__":
    debugger = test_debugger_html(DifferenceDebugger(CoverageCollector))


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
    list_with_coverage(remove_html_markup, only_in_fail)


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
    debugger = test_debugger_html(DifferenceDebugger(CoverageCollector))


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
    def color(self, event):
        """Return a color for the given event."""
        passing = self.all_pass_events()
        failing = self.all_fail_events()

        if event in passing and event in failing:
            return 'lightyellow'
        elif event in failing:
            return 'mistyrose'
        elif event in passing:
            return 'honeydew'
        else:
            return None

class DiscreteSpectrumDebugger(DiscreteSpectrumDebugger):
    def list_with_spectrum(self, function, show_color_names=False):
        """Print a listing of the given function, using suspiciousness colors."""
        source_lines, starting_line_number = \
           inspect.getsourcelines(function)

        line_number = starting_line_number
        out = ""
        for line in source_lines:
            line = html.escape(line)
            if line.strip() == '':
                line = '&nbsp;'

            line = str(line_number).rjust(4) + ' ' + line
            color = self.color(line_number)

            if show_color_names:
                line = f'{repr(color):20} {line}'

            if color:
                line = f'<pre style="background-color:{color}">' \
                        f'{line.rstrip()}</pre>'
            else:
                line = f'<pre>{line}</pre>'

            out += line + '\n'
            line_number += 1

        return HTML(out)

if __name__ == "__main__":
    debugger = test_debugger_html(DiscreteSpectrumDebugger(CoverageCollector))


if __name__ == "__main__":
    debugger.list_with_spectrum(remove_html_markup)


if __name__ == "__main__":
    quiz("Does the line <code>quote = not quote</code> actually contain the defect?",
        [
            "Yes, it should be fixed",
            "No, the defect is elsewhere"
        ],
         164 * 2 % 326
        )


# ### Continuous Spectrum

if __name__ == "__main__":
    print('\n### Continuous Spectrum')




if __name__ == "__main__":
    remove_html_markup('<b color="blue">text</b>')


if __name__ == "__main__":
    debugger = test_debugger_html(DiscreteSpectrumDebugger(CoverageCollector))
    with debugger.collect_pass():
        remove_html_markup('<b link="blue"></b>')


if __name__ == "__main__":
    debugger.only_fail_events()


if __name__ == "__main__":
    debugger.list_with_spectrum(remove_html_markup)


class ContinuousSpectrumDebugger(DiscreteSpectrumDebugger):
    def collectors_with_event(self, event, category):
        """Return all collectors in a category that observed the given event."""
        all_runs = self.collectors[category]
        collectors_with_event = set(collector for collector in all_runs 
                              if event in collector.events())
        return collectors_with_event

    def collectors_without_event(self, event, category):
        """Return all collectors in a category that did not observe the given event."""
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

if __name__ == "__main__":
    debugger = test_debugger_html(ContinuousSpectrumDebugger(CoverageCollector))


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
    debugger = test_debugger_html(ContinuousSpectrumDebugger(CoverageCollector))
    for line in debugger.only_fail_events():
        print(line, debugger.brightness(line))


class ContinuousSpectrumDebugger(ContinuousSpectrumDebugger):
    def color(self, line):
        hue = debugger.hue(line)
        if hue is None:
            return None
        saturation = debugger.brightness(line)

        # HSL color values are specified with: 
        # hsl(hue, saturation, lightness).
        return f"hsl({hue * 120}, {saturation * 100}%, 80%)"

if __name__ == "__main__":
    debugger = test_debugger_html(ContinuousSpectrumDebugger(CoverageCollector))


if __name__ == "__main__":
    for line in debugger.only_fail_events():
        print(line, debugger.color(line))


if __name__ == "__main__":
    for line in debugger.only_pass_events():
        print(line, debugger.color(line))


if __name__ == "__main__":
    debugger.list_with_spectrum(remove_html_markup)


if __name__ == "__main__":
    with debugger.collect_pass():
        out = remove_html_markup('<b link="blue"></b>')


if __name__ == "__main__":
    quiz('In which color will the <code>quote = not quote</code> "culprit" line be shown after executing the above code?',
        [
            '<span style="background-color: hsl(120.0, 50.0%, 80%)">Green</span>',
            '<span style="background-color: hsl(60.0, 100.0%, 80%)">Yellow</span>',
            '<span style="background-color: hsl(30.0, 100.0%, 80%)">Orange</span>',
            '<span style="background-color: hsl(0.0, 100.0%, 80%)">Red</span>'
        ],
         999 / 333
        )


if __name__ == "__main__":
    debugger.list_with_spectrum(remove_html_markup)


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
    debugger = test_debugger_middle(ContinuousSpectrumDebugger(CoverageCollector))


if __name__ == "__main__":
    debugger.event_table()


if __name__ == "__main__":
    debugger.list_with_spectrum(middle)


if __name__ == "__main__":
    quiz("Which of the above lines should be fixed?",
        [
            '<span style="background-color: hsl(45.0, 100%, 80%)">Line 3: <code>elif x &lt; y</code></span>',
            '<span style="background-color: hsl(34.28571428571429, 100.0%, 80%)">Line 5: <code>elif x &lt; z</code></span>',
            '<span style="background-color: hsl(20.000000000000004, 100.0%, 80%)">Line 6: <code>return y</code></span>',
            '<span style="background-color: hsl(120.0, 20.0%, 80%)">Line 9: <code>return y</code></span>',
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
    def suspiciousness(self, event):
        """Return the suspiciousness (>=0) of an event. 0 stands for not suspicious."""
        return 0

    def rank_by_suspiciousness(self):
        """Return a list of events, sorted by suspiciousness, highest first."""
        events = list(self.all_events())
        events.sort(key=self.suspiciousness, reverse=True)
        return events

# ### The Tarantula Metric

if __name__ == "__main__":
    print('\n### The Tarantula Metric')




class TarantulaDebugger(ContinuousSpectrumDebugger, RankingDebugger):
    def suspiciousness(self, event):
        hue = self.hue(event)
        if hue is None:
            return None
        return 1 - hue

if __name__ == "__main__":
    debugger = test_debugger_html(TarantulaDebugger(CoverageCollector))


if __name__ == "__main__":
    debugger.list_with_spectrum(remove_html_markup)


if __name__ == "__main__":
    debugger.rank_by_suspiciousness()


if __name__ == "__main__":
    debugger.suspiciousness(2)


if __name__ == "__main__":
    debugger = test_debugger_middle(TarantulaDebugger(CoverageCollector))


if __name__ == "__main__":
    debugger.list_with_spectrum(middle)


if __name__ == "__main__":
    debugger.rank_by_suspiciousness()


if __name__ == "__main__":
    debugger.suspiciousness(5)


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
    debugger = test_debugger_html(OchiaiDebugger(CoverageCollector))


if __name__ == "__main__":
    debugger.list_with_spectrum(remove_html_markup)


if __name__ == "__main__":
    debugger.rank_by_suspiciousness()


if __name__ == "__main__":
    debugger.suspiciousness(2)


if __name__ == "__main__":
    debugger = test_debugger_middle(OchiaiDebugger(CoverageCollector))


if __name__ == "__main__":
    debugger.list_with_spectrum(middle)


if __name__ == "__main__":
    debugger.rank_by_suspiciousness()


if __name__ == "__main__":
    debugger.suspiciousness(5)


# ### How Useful is Ranking?

if __name__ == "__main__":
    print('\n### How Useful is Ranking?')




# ## Other Events besides Coverage

if __name__ == "__main__":
    print('\n## Other Events besides Coverage')




class ValueCollector(Collector):
    """"A class to collect local variables and their values."""

    def __init__(self):
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
    debugger.event_table(color=True)


# ### Training Classifiers

if __name__ == "__main__":
    print('\n### Training Classifiers')




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
    debugger = test_debugger_html(ClassifyingDebugger(CoverageCollector))
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
    debugger = test_debugger_html(ClassifyingDebugger(CoverageCollector))
    debugger.features()


class ClassifyingDebugger(ClassifyingDebugger):
    def feature_names(self):
        return [repr(feature) for feature in self.all_events()]

if __name__ == "__main__":
    debugger = test_debugger_html(ClassifyingDebugger(CoverageCollector))
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
    debugger = test_debugger_html(ClassifyingDebugger(CoverageCollector))
    debugger.shape("remove_html_markup(s='abc')")


class ClassifyingDebugger(ClassifyingDebugger):
    def X(self):
        X = []
        samples = self.samples()
        for key in samples:
            X += [self.shape(key)]
        return X

if __name__ == "__main__":
    debugger = test_debugger_html(ClassifyingDebugger(CoverageCollector))
    debugger.X()


class ClassifyingDebugger(ClassifyingDebugger):
    def Y(self):
        Y = []
        samples = self.samples()
        for key in samples:
            Y += [samples[key]]
        return Y

if __name__ == "__main__":
    debugger = test_debugger_html(ClassifyingDebugger(CoverageCollector))
    debugger.Y()


from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz

class ClassifyingDebugger(ClassifyingDebugger):
    def classifier(self):
        classifier = DecisionTreeClassifier()
        classifier = classifier.fit(self.X(), self.Y())
        return classifier

if __name__ == "__main__":
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
    debugger = test_debugger_html(ClassifyingDebugger(CoverageCollector))
    classifier = debugger.classifier()
    debugger.show_classifier(classifier)


if __name__ == "__main__":
    classifier.predict([[1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1]])


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




if __name__ == "__main__":
    debugger = TarantulaDebugger(CoverageCollector)
    with debugger.collect_pass():
        remove_html_markup("abc")
    with debugger.collect_pass():
        remove_html_markup('<b>abc</b>')
    with debugger.collect_fail():
        remove_html_markup('"abc"')


if __name__ == "__main__":
    debugger.event_table(color=True)


if __name__ == "__main__":
    debugger.list_with_spectrum(remove_html_markup)


if __name__ == "__main__":
    debugger.rank_by_suspiciousness()


if __name__ == "__main__":
    # ignore
    from ClassDiagram import display_class_hierarchy


if __name__ == "__main__":
    # ignore
    display_class_hierarchy([TarantulaDebugger, OchiaiDebugger])


if __name__ == "__main__":
    # ignore
    display_class_hierarchy([CoverageCollector, ValueCollector])


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



