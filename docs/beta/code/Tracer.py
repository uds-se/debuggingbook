#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# "Tracing Executions" - a chapter of "The Debugging Book"
# Web site: https://www.debuggingbook.org/html/Tracer.html
# Last change: 2025-10-26 18:59:11+01:00
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
The Debugging Book - Tracing Executions

This file can be _executed_ as a script, running all experiments:

    $ python Tracer.py

or _imported_ as a package, providing classes, functions, and constants:

    >>> from debuggingbook.Tracer import <identifier>

but before you do so, _read_ it and _interact_ with it at:

    https://www.debuggingbook.org/html/Tracer.html

**Note**: The examples in this section only work after the rest of the cells have been executed.

For more details, source, and documentation, see
"The Debugging Book - Tracing Executions"
at https://www.debuggingbook.org/html/Tracer.html
'''


# Allow to use 'from . import <module>' when run as script (cf. PEP 366)
if __name__ == '__main__' and __package__ is None:
    __package__ = 'debuggingbook'


# Tracing Executions
# ==================

if __name__ == '__main__':
    print('# Tracing Executions')



if __name__ == '__main__':
    from .bookutils import YouTubeVideo
    YouTubeVideo("UYAvCl-5NGY")

if __name__ == '__main__':
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)

from .bookutils import quiz

from . import Intro_Debugging

## Tracing Python Programs
## -----------------------

if __name__ == '__main__':
    print('\n## Tracing Python Programs')



from types import FrameType, TracebackType

from typing import Any, Optional, Callable, Dict, List, Type, TextIO, cast

def traceit(frame: FrameType, event: str, arg: Any) -> Optional[Callable]:
    ...

def traceit(frame: FrameType, event: str, arg: Any) -> Optional[Callable]:  # type: ignore
    print(event, frame.f_lineno, frame.f_code.co_name, frame.f_locals)

def traceit(frame: FrameType, event: str, arg: Any) -> Optional[Callable]:  # type: ignore
    print(event, frame.f_lineno, frame.f_code.co_name, frame.f_locals)
    return traceit

from .Intro_Debugging import remove_html_markup

import inspect

from .bookutils import print_content

if __name__ == '__main__':
    content, start_line_number = inspect.getsourcelines(remove_html_markup)
    print_content(content="".join(content).strip(), filename='.py', start_line_number=start_line_number)

import sys

def remove_html_markup_traced(s):  # type: ignore
    sys.settrace(traceit)
    ret = remove_html_markup(s)
    sys.settrace(None)
    return ret

if __name__ == '__main__':
    remove_html_markup_traced('xyz')

def traceit(frame: FrameType, event: str, arg: Any) -> Optional[Callable]:  # type: ignore
    if 'c' in frame.f_locals:
        value_of_c = frame.f_locals['c']
        print(f"{frame.f_lineno:} c = {repr(value_of_c)}")
    else:
        print(f"{frame.f_lineno:} c is undefined")

    return traceit

if __name__ == '__main__':
    remove_html_markup_traced('xyz')

import math

if __name__ == '__main__':
    quiz("What happens if the tracing function returns `None` while tracing function `f()`?"
         " (You can also try this out yourself.)",
         [
             'Tracing stops for all functions;'
             ' the tracing function is no longer called',
             'Tracing stops for `f()`: the tracing function is called when `f()` returns',
             'Tracing stops for `f()` the rest of the execution: the tracing function'
             ' is no longer called for calls to `f()`',
             'Nothing changes'
         ], "int(math.log(7.38905609893065))", globals())

## A Tracer Class
## --------------

if __name__ == '__main__':
    print('\n## A Tracer Class')



from .StackInspector import StackInspector

class Tracer(StackInspector):
    """A class for tracing a piece of code. Use as `with Tracer(): block()`"""

    def __init__(self, *, file: TextIO = sys.stdout) -> None:
        """Trace a block of code, sending logs to `file` (default: stdout)"""
        self.original_trace_function: Optional[Callable] = None
        self.file = file

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """Tracing function. To be overridden in subclasses."""
        self.log(event, frame.f_lineno, frame.f_code.co_name, frame.f_locals)

    def _traceit(self, frame: FrameType, event: str, arg: Any) -> Optional[Callable]:
        """Internal tracing function."""
        if self.our_frame(frame):
            # Do not trace our own methods
            pass
        else:
            self.traceit(frame, event, arg)
        return self._traceit

    def log(self, *objects: Any, 
            sep: str = ' ', end: str = '\n', 
            flush: bool = True) -> None:
        """
        Like `print()`, but always sending to `file` given at initialization,
        and flushing by default.
        """
        print(*objects, sep=sep, end=end, file=self.file, flush=flush)

    def __enter__(self) -> Any:
        """Called at begin of `with` block. Turn tracing on."""
        self.original_trace_function = sys.gettrace()
        sys.settrace(self._traceit)

        # This extra line also enables tracing for the current block
        # inspect.currentframe().f_back.f_trace = self._traceit
        return self

    def __exit__(self, exc_tp: Type, exc_value: BaseException, 
                 exc_traceback: TracebackType) -> Optional[bool]:
        """
        Called at end of `with` block. Turn tracing off.
        Return `None` if ok, not `None` if internal error.
        """
        sys.settrace(self.original_trace_function)

        # Note: we must return a non-True value here,
        # such that we re-raise all exceptions
        if self.is_internal_error(exc_tp, exc_value, exc_traceback):
            return False  # internal error
        else:
            return None  # all ok

if __name__ == '__main__':
    with Tracer():
        remove_html_markup("abc")

## Accessing Source Code
## ---------------------

if __name__ == '__main__':
    print('\n## Accessing Source Code')



import inspect

class Tracer(Tracer):
    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """Tracing function; called at every line. To be overloaded in subclasses."""

        if event == 'line':
            module = inspect.getmodule(frame.f_code)
            if module is None:
                source = inspect.getsource(frame.f_code)
            else:
                source = inspect.getsource(module)
            current_line = source.split('\n')[frame.f_lineno - 1]
            self.log(frame.f_lineno, current_line)

if __name__ == '__main__':
    with Tracer():
        remove_html_markup("abc")

## Tracing Calls and Returns
## -------------------------

if __name__ == '__main__':
    print('\n## Tracing Calls and Returns')



class Tracer(Tracer):
    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """Tracing function. To be overridden in subclasses."""

        if event == 'call':
            self.log(f"Calling {frame.f_code.co_name}()")

        if event == 'line':
            module = inspect.getmodule(frame.f_code)
            if module is None:
                source = inspect.getsource(frame.f_code)
            else:
                source = inspect.getsource(module)
            if source:
                current_line = source.split('\n')[frame.f_lineno - 1]
                self.log(frame.f_lineno, current_line)

        if event == 'return':
            self.log(f"{frame.f_code.co_name}() returns {repr(arg)}")

if __name__ == '__main__':
    with Tracer():
        remove_html_markup("abc")

## Tracing Variable Changes
## ------------------------

if __name__ == '__main__':
    print('\n## Tracing Variable Changes')



class Tracer(Tracer):
    def __init__(self, file: TextIO = sys.stdout) -> None:
        """
        Create a new tracer.
        If `file` is given, output to `file` instead of stdout.
        """

        self.last_vars: Dict[str, Any] = {}
        super().__init__(file=file)

    def changed_vars(self, new_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Track changed variables, based on `new_vars` observed."""
        changed = {}
        for var_name, var_value in new_vars.items():
            if (var_name not in self.last_vars or
                    self.last_vars[var_name] != var_value):
                changed[var_name] = var_value
        self.last_vars = new_vars.copy()
        return changed

if __name__ == '__main__':
    tracer = Tracer()

if __name__ == '__main__':
    tracer.changed_vars({'a': 10})

if __name__ == '__main__':
    tracer.changed_vars({'a': 10, 'b': 25})

if __name__ == '__main__':
    tracer.changed_vars({'a': 10, 'b': 25})

if __name__ == '__main__':
    changes = tracer.changed_vars({'c': 10, 'd': 25})
    changes

if __name__ == '__main__':
    ", ".join([var + " = " + repr(changes[var]) for var in changes])

class Tracer(Tracer):
    def print_debugger_status(self, frame: FrameType, event: str, arg: Any) -> None:
        """Show current source line and changed vars"""
        changes = self.changed_vars(frame.f_locals)
        changes_s = ", ".join([var + " = " + repr(changes[var])
                               for var in changes])

        if event == 'call':
            self.log("Calling " + frame.f_code.co_name + '(' + changes_s + ')')
        elif changes:
            self.log(' ' * 40, '#', changes_s)

        if event == 'line':
            try:
                module = inspect.getmodule(frame.f_code)
                if module is None:
                    source = inspect.getsource(frame.f_code)
                else:
                    source = inspect.getsource(module)
                current_line = source.split('\n')[frame.f_lineno - 1]

            except OSError as err:
                self.log(f"{err.__class__.__name__}: {err}")
                current_line = ""

            self.log(repr(frame.f_lineno) + ' ' + current_line)

        if event == 'return':
            self.log(frame.f_code.co_name + '()' + " returns " + repr(arg))
            self.last_vars = {}  # Delete 'last' variables

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """Tracing function; called at every line. To be overloaded in subclasses."""
        self.print_debugger_status(frame, event, arg)

if __name__ == '__main__':
    with Tracer():
        remove_html_markup('<b>x</b>')

## Conditional Tracing
## -------------------

if __name__ == '__main__':
    print('\n## Conditional Tracing')



class ConditionalTracer(Tracer):
    def __init__(self, *, condition: Optional[str] = None, file: TextIO = sys.stdout) -> None:
        """Constructor. Trace all events for which `condition` (a Python expr) holds."""

        if condition is None:
            condition = 'False'
        assert isinstance(condition, str)

        self.condition: str = condition
        self.last_report: Optional[bool] = None
        super().__init__(file=file)

class ConditionalTracer(ConditionalTracer):
    def eval_in_context(self, expr: str, frame: FrameType) -> Optional[bool]:
        try:
            cond = eval(expr, None, frame.f_locals)
        except NameError:  # (yet) undefined variable
            cond = None
        return cond

class ConditionalTracer(ConditionalTracer):
    def do_report(self, frame: FrameType, event: str, arg: Any) -> Optional[bool]:
        return self.eval_in_context(self.condition, frame)

class ConditionalTracer(ConditionalTracer):
    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        report = self.do_report(frame, event, arg)
        if report != self.last_report:
            if report:
                self.log("...")
            self.last_report = report

        if report:
            self.print_debugger_status(frame, event, arg)

if __name__ == '__main__':
    with ConditionalTracer(condition='quote'):
        remove_html_markup('<b title="bar">"foo"</b>')

if __name__ == '__main__':
    quiz("What happens if the condition contains a syntax error?",
         [
             "The tracer stops, raising an exception",
             "The tracer continues as if the condition were `True`",
             "The tracer continues as if the condition were `False`",
         ], '393 % 7')

from .ExpectError import ExpectError

if __name__ == '__main__':
    with ExpectError(SyntaxError):
        with ConditionalTracer(condition='2 +'):
            remove_html_markup('<b title="bar">"foo"</b>')

if __name__ == '__main__':
    with ExpectError():
        with ConditionalTracer(condition='undefined_variable'):
            remove_html_markup('<b title="bar">"foo"</b>')

class ConditionalTracer(ConditionalTracer):
    def eval_in_context(self, expr: str, frame: FrameType) -> Any:
        frame.f_locals['function'] = frame.f_code.co_name
        frame.f_locals['line'] = frame.f_lineno

        return super().eval_in_context(expr, frame)

if __name__ == '__main__':
    with ConditionalTracer(condition="function == 'remove_html_markup' and line >= 237"):
        remove_html_markup('xyz')

if __name__ == '__main__':
    quiz("If the program under test contains a variable named `line`, "
         "which `line` does the condition refer to?",
         [
            "`line` as in the debugger",
            "`line` as in the program"
         ], '(326 * 27 == 8888) + 1')

## Watching Events
## ---------------

if __name__ == '__main__':
    print('\n## Watching Events')



class EventTracer(ConditionalTracer):
    """Log when a given event expression changes its value"""

    def __init__(self, *, condition: Optional[str] = None,
                 events: List[str] = [], file: TextIO = sys.stdout) -> None:
        """Constructor. `events` is a list of expressions to watch."""
        assert isinstance(events, List)  # avoid type errors
        self.events = events
        self.last_event_values: Dict[str, Any] = {}
        super().__init__(file=file, condition=condition)

class EventTracer(EventTracer):
    def events_changed(self, events: List[str], frame: FrameType) -> bool:
        """Return True if any of the observed `events` has changed"""
        change = False
        for event in events:
            value = self.eval_in_context(event, frame)

            if (event not in self.last_event_values or
                    value != self.last_event_values[event]):
                self.last_event_values[event] = value
                change = True

        return change

class EventTracer(EventTracer):
    def do_report(self, frame: FrameType, event: str, arg: Any) -> bool:
        """Return True if a line should be shown"""
        return (self.eval_in_context(self.condition, frame) or
                self.events_changed(self.events, frame))

if __name__ == '__main__':
    with EventTracer(events=['quote', 'tag']):
        remove_html_markup('<b title="bar">"foo"</b>')

## Efficient Tracing
## -----------------

if __name__ == '__main__':
    print('\n## Efficient Tracing')



from .Timer import Timer

if __name__ == '__main__':
    runs = 1000

if __name__ == '__main__':
    with Timer() as t:
        for i in range(runs):
            remove_html_markup('<b title="bar">"foo"</b>')
    untraced_execution_time = t.elapsed_time()
    untraced_execution_time

if __name__ == '__main__':
    with Timer() as t:
        for i in range(runs):
            with EventTracer():
                remove_html_markup('<b title="bar">"foo"</b>')
    traced_execution_time = t.elapsed_time()
    traced_execution_time

if __name__ == '__main__':
    traced_execution_time / untraced_execution_time

TRACER_CODE = \
    "TRACER.print_debugger_status(inspect.currentframe(), 'line', None); "

TRACER = Tracer()

def insert_tracer(function: Callable, breakpoints: List[int] = []) -> Callable:
    """Return a variant of `function` with tracing code `TRACER_CODE` inserted
       at each line given by `breakpoints`."""

    source_lines, starting_line_number = inspect.getsourcelines(function)

    breakpoints.sort(reverse=True)
    for given_line in breakpoints:
        # Set new source line
        relative_line = given_line - starting_line_number + 1
        inject_line = source_lines[relative_line - 1]
        indent = len(inject_line) - len(inject_line.lstrip())
        source_lines[relative_line - 1] = ' ' * indent + TRACER_CODE + inject_line.lstrip()

    # Rename function
    new_function_name = function.__name__ + "_traced"
    source_lines[0] = source_lines[0].replace(function.__name__, new_function_name)
    new_def = "".join(source_lines)

    # For debugging
    print_content(new_def, '.py', start_line_number=starting_line_number)

    # We keep original source and filename to ease debugging
    prefix = '\n' * starting_line_number    # Get line number right
    new_function_code = compile(prefix + new_def, function.__code__.co_filename, 'exec')
    exec(new_function_code)
    new_function = eval(new_function_name)
    return new_function

if __name__ == '__main__':
    _, remove_html_markup_starting_line_number = inspect.getsourcelines(remove_html_markup)
    breakpoints = [(remove_html_markup_starting_line_number - 1) + 7, 
                   (remove_html_markup_starting_line_number - 1) + 18]

if __name__ == '__main__':
    remove_html_markup_traced = insert_tracer(remove_html_markup, breakpoints)

if __name__ == '__main__':
    with Timer() as t:
        remove_html_markup_traced('<b title="bar">"foo"</b>')
    static_tracer_execution_time = t.elapsed_time()

if __name__ == '__main__':
    static_tracer_execution_time

if __name__ == '__main__':
    line7 = (remove_html_markup_starting_line_number - 1) + 7
    line18 = (remove_html_markup_starting_line_number - 1) + 18

    with Timer() as t:
        with EventTracer(condition=f'line == {line7} or line == {line18}'):
            remove_html_markup('<b title="bar">"foo"</b>')

    dynamic_tracer_execution_time = t.elapsed_time()
    dynamic_tracer_execution_time

if __name__ == '__main__':
    dynamic_tracer_execution_time / static_tracer_execution_time

def some_extreme_function(s: str) -> None:
    ...  # Long-running function
    remove_html_markup(s)

if __name__ == '__main__':
    with EventTracer(condition=f"function=='remove_html_markup' and line == {line18}"):
        some_extreme_function("foo")

if __name__ == '__main__':
    quiz("In the above example, "
         "where is the `EventTracer.traceit()` function called?",
         [
             "When `some_extreme_function()` returns",
             "For each line of `some_extreme_function()`",
             "When `remove_html_markup()` returns",
             "For each line of `remove_html_markup()`"
         ], "[ord(c) - 100 for c in 'efgh']")

## Tracing Binary Executables
## --------------------------

if __name__ == '__main__':
    print('\n## Tracing Binary Executables')



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



### Low-Level Debugging Interfaces

if __name__ == '__main__':
    print('\n### Low-Level Debugging Interfaces')



### High-Level Debugging Interfaces

if __name__ == '__main__':
    print('\n### High-Level Debugging Interfaces')



## Exercises
## ---------

if __name__ == '__main__':
    print('\n## Exercises')



### Exercise 1: Exception Handling

if __name__ == '__main__':
    print('\n### Exercise 1: Exception Handling')



def fail() -> float:
    return 2 / 0

if __name__ == '__main__':
    with Tracer():
        try:
            fail()
        except Exception:
            pass

class Tracer(Tracer):
    def print_debugger_status(self, frame: FrameType, event: str, arg: Any) -> None:
        if event == 'exception':
            exception, value, tb = arg
            self.log(f"{frame.f_code.co_name}() "
                     f"raises {exception.__name__}: {value}")
        else:
            super().print_debugger_status(frame, event, arg)

if __name__ == '__main__':
    with Tracer():
        try:
            fail()
        except Exception:
            pass

### Exercise 2: Syntax-Based Instrumentation

if __name__ == '__main__':
    print('\n### Exercise 2: Syntax-Based Instrumentation')



def foo():  # type: ignore
    ret = 2 * 2
    return ret

if __name__ == '__main__':
    source = inspect.getsource(foo)
    print_content(source, '.py')

import ast

from .bookutils import show_ast

if __name__ == '__main__':
    tree = ast.parse(source)

if __name__ == '__main__':
    show_ast(tree)

from ast import NodeTransformer, FunctionDef, fix_missing_locations, AST, Module

if __name__ == '__main__':
    subtree_to_be_injected: AST = ast.parse("print('entering function')")

if __name__ == '__main__':
    show_ast(subtree_to_be_injected)

if __name__ == '__main__':
    subtree_to_be_injected = cast(Module, subtree_to_be_injected).body[0]

class InjectPass(NodeTransformer):
    def visit_FunctionDef(self, node: FunctionDef) -> AST:
        return FunctionDef(
            name=node.name,
            args=node.args,
            body=[subtree_to_be_injected] + node.body,
            decorator_list=node.decorator_list,
            returns=node.returns
        )  # type: ignore

if __name__ == '__main__':
    new_tree = fix_missing_locations(InjectPass().visit(tree))

if __name__ == '__main__':
    show_ast(new_tree)

if __name__ == '__main__':
    new_source = ast.unparse(new_tree)
    print_content(new_source, '.py')

if __name__ == '__main__':
    exec(new_source)

if __name__ == '__main__':
    foo()

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



if __name__ == '__main__':
    with EventTracer(condition='line == 223 or len(out) >= 6'):
        remove_html_markup('<b>foo</b>bar')

if __name__ == '__main__':
    with EventTracer(events=["c == '/'"]):
        remove_html_markup('<b>foo</b>bar')

from .ClassDiagram import display_class_hierarchy

if __name__ == '__main__':
    display_class_hierarchy(EventTracer,
                            public_methods=[
                                Tracer.__init__,
                                Tracer.__enter__,
                                Tracer.__exit__,
                                Tracer.changed_vars,
                                Tracer.print_debugger_status,
                                ConditionalTracer.__init__,
                                EventTracer.__init__,
                            ],
                            project='debuggingbook')
