#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Fuzzing Book".
# Web site: https://www.fuzzingbook.org/html/DeltaDebugger.html
# Last change: 2020-11-29 16:19:00+01:00
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


# # Reducing Failure-Inducing Inputs

if __name__ == "__main__":
    print('# Reducing Failure-Inducing Inputs')




# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




# ## Why Reducing?

if __name__ == "__main__":
    print('\n## Why Reducing?')




if __name__ == "__main__":
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)


if __package__ is None or __package__ == "":
    import Tracer
else:
    from . import Tracer


if __package__ is None or __package__ == "":
    from bookutils import quiz
else:
    from .bookutils import quiz


def mystery(inp):
    x = inp.find(chr(0o17 + 0o31))
    y = inp.find(chr(0o27 + 0o22))
    if x >= 0 and y >= 0 and x < y:
        raise ValueError("Invalid input")
    else:
        pass

import random

if __name__ == "__main__":
    random.randrange(32, 128)


def fuzz():
    length = random.randrange(10, 70)
    fuzz = ""
    for i in range(length):
        fuzz += chr(random.randrange(32, 127))
    return fuzz

if __name__ == "__main__":
    for i in range(6):
        print(repr(fuzz()))


if __name__ == "__main__":
    while True:
        inp = fuzz()
        try:
            mystery(inp)
        except ValueError:
            break


if __name__ == "__main__":
    failing_input = inp
    failing_input


if __package__ is None or __package__ == "":
    from ExpectError import ExpectError
else:
    from .ExpectError import ExpectError


if __name__ == "__main__":
    with ExpectError():
        mystery(failing_input)


# ## Manual Input Reduction

if __name__ == "__main__":
    print('\n## Manual Input Reduction')




if __name__ == "__main__":
    failing_input


if __name__ == "__main__":
    half_length = len(failing_input) // 2   # // is integer division
    first_half = failing_input[:half_length]
    first_half


if __name__ == "__main__":
    with ExpectError():
        mystery(first_half)


if __name__ == "__main__":
    second_half = failing_input[half_length:]
    assert first_half + second_half == failing_input
    second_half


if __name__ == "__main__":
    with ExpectError():
        mystery(second_half)


# ## Delta Debugging

if __name__ == "__main__":
    print('\n## Delta Debugging')




if __name__ == "__main__":
    quarter_length = len(failing_input) // 4
    input_without_first_quarter = failing_input[quarter_length:]
    input_without_first_quarter


if __name__ == "__main__":
    with ExpectError():
        mystery(input_without_first_quarter)


if __name__ == "__main__":
    input_without_first_and_second_quarter = failing_input[quarter_length * 2:]
    input_without_first_and_second_quarter


if __name__ == "__main__":
    with ExpectError():
        mystery(input_without_first_and_second_quarter)


if __name__ == "__main__":
    second_half


if __name__ == "__main__":
    input_without_first_and_second_quarter


if __name__ == "__main__":
    input_without_first_and_third_quarter = failing_input[quarter_length:
                                                          quarter_length * 2] + failing_input[quarter_length * 3:]
    input_without_first_and_third_quarter


if __name__ == "__main__":
    with ExpectError():
        mystery(input_without_first_and_third_quarter)


PASS = 'PASS'
FAIL = 'FAIL'
UNRESOLVED = 'UNRESOLVED'

def ddmin(test, inp, *test_args):
    """Reduce the input inp, using the outcome of test(fun, inp)."""
    assert test(inp, *test_args) != PASS

    n = 2     # Initial granularity
    while len(inp) >= 2:
        start = 0
        subset_length = len(inp) / n
        some_complement_is_failing = False

        while start < len(inp):
            complement = inp[:int(start)] + \
                inp[int(start + subset_length):]

            if test(complement, *test_args) == FAIL:
                inp = complement
                n = max(n - 1, 2)
                some_complement_is_failing = True
                break

            start += subset_length

        if not some_complement_is_failing:
            if n == len(inp):
                break
            n = min(n * 2, len(inp))

    return inp

def generic_test(inp, fun, expected_exc=None):
    result = None
    detail = ""
    try:
        result = fun(inp)
        outcome = PASS
    except Exception as exc:
        detail = f" ({type(exc).__name__}: {str(exc)})"
        if expected_exc is None:
            outcome = FAIL
        elif type(exc) == type(expected_exc) and str(exc) == str(expected_exc):
            outcome = FAIL
        else:
            outcome = UNRESOLVED

    print(f"{fun.__name__}({repr(inp)}): {outcome}{detail}")
    return outcome

if __name__ == "__main__":
    ddmin(generic_test, failing_input, mystery, ValueError('Invalid input'))


# ## A Simple DeltaDebugger Interface

if __name__ == "__main__":
    print('\n## A Simple DeltaDebugger Interface')




# ### Excursion: Implementing DeltaDebugger

if __name__ == "__main__":
    print('\n### Excursion: Implementing DeltaDebugger')




# #### Collecting a Call

if __name__ == "__main__":
    print('\n#### Collecting a Call')




import sys

from types import FunctionType

class CallCollector(object):
    """Collect an exception-raising function call f().
    Use as `with CallCollector(): f()`"""

    def __init__(self):
        """Initialize collector"""
        self._function = None
        self._args = None
        self._exception = None

    def traceit(self, frame, event, arg):
        """Tracing function. Collect first call, then turn tracing off."""
        if event == 'call':
            name = frame.f_code.co_name
            if name.startswith('__'):
                # Internal function
                return

            if self._function is None:
                self._function = FunctionType(frame.f_code,
                                              globals=globals(),
                                              name=name)
                self._args = {}  # Create a local copy
                for var in frame.f_locals:
                    self._args[var] = frame.f_locals[var]

            # Turn tracing off
            sys.settrace(self.original_trace_function)

    def diagnosis(self):
        """Produce a diagnosis. To be defined in subclasses."""
        pass

    def args(self):
        """Return the dictionary of collected arguments."""
        return self._args

    def function(self):
        """Return the function called."""
        return self._function

    def __enter__(self):
        """Called at begin of `with` block. Turn tracing on."""
        self.original_trace_function = sys.gettrace()
        sys.settrace(self.traceit)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Called at end of `with` block. Turn tracing off."""
        sys.settrace(self.original_trace_function)
        if self._function is None:
            return False  # Re-raise exception, if any

        self._exception = exc_value
        self.diagnosis()
        return True  # Ignore exception

if __name__ == "__main__":
    with CallCollector() as call_collector:
        mystery(failing_input)


if __name__ == "__main__":
    call_collector._function


if __name__ == "__main__":
    call_collector._args


if __name__ == "__main__":
    call_collector._exception


if __name__ == "__main__":
    with ExpectError():
        with CallCollector() as c:
            some_error()


# #### Repeating a Call

if __name__ == "__main__":
    print('\n#### Repeating a Call')




if __name__ == "__main__":
    with ExpectError():
        call_collector._function("foo")


if __name__ == "__main__":
    with ExpectError():
        call_collector._function(failing_input)


if __name__ == "__main__":
    with ExpectError():
        call_collector._function(**call_collector._args)


class CallCollector(CallCollector):
    def call(self, new_args={}):
        """Call collected function. If new_args is given,
        override arguments from its {var: value} entries."""
        args = {}  # Create local copy
        for var in self._args:
            args[var] = self._args[var]
        for var in new_args:
            args[var] = new_args[var]

        return self._function(**new_args)

if __name__ == "__main__":
    with CallCollector() as call_collector:
        mystery(failing_input)
    with ExpectError():
        call_collector.call()


if __name__ == "__main__":
    with ExpectError():
        call_collector.call({'inp': 'foo'})


# #### Testing, Logging, and Caching

if __name__ == "__main__":
    print('\n#### Testing, Logging, and Caching')




class CallReducer(CallCollector):
    def __init__(self, log=False):
        """Initialize. If log is True, enable logging."""
        super().__init__()
        self.log = log
        self.reset()

    def reset(self):
        """Reset the number of tests."""
        self.tests = 0

    def run(self, args):
        """Run collected function with args. Return
        * PASS if no exception occurred
        * FAIL if the collected exception occurred
        * UNRESOLVED if some other exception occurred.
        Not to be used directly; can be overloaded in subclasses.
        """
        try:
            result = self.call(args)
        except Exception as exc:
            self.last_exception = exc
            if (type(exc) == type(self._exception) and
                    str(exc) == str(self._exception)):
                return FAIL
            else:
                return UNRESOLVED  # Some other failure

        self.last_result = result
        return PASS

class CallReducer(CallReducer):
    def format_call(self, args=None):
        """Return a string representing a call of the function with given args."""
        if args is None:
            args = self._args
        return self._function.__name__ + "(" + \
            ", ".join(f"{arg}={repr(args[arg])}" for arg in args) + ")"

    def format_exception(self, exc):
        """Return a string representing the given exception."""
        s = type(exc).__name__
        if str(exc):
            s += ": " + str(exc)
        return s

    def test(self, args):
        """Like run(), but also log detail and keep statistics."""
        outcome = self.run(args)
        if outcome == PASS:
            detail = ""
        else:
            detail = f" ({self.format_exception(self.last_exception)})"

        self.tests += 1
        if self.log:
            print(f"Test #{self.tests} {self.format_call(args)}: {outcome}{detail}")

        return outcome

    def reduce_arg(self, var_to_be_reduced, args):
        """Determine and return a minimal value for var_to_be_reduced.
        To be overloaded in subclasses."""
        return args[var_to_be_reduced]

if __name__ == "__main__":
    with CallReducer(log=True) as reducer:
        mystery(failing_input)

    reducer.test({'inp': failing_input})
    reducer.test({'inp': '123'})
    reducer.test({'inp': '123'})


class CachingCallReducer(CallReducer):
    """Like CallReducer, but cache test outcomes."""
    def reset(self):
        super().reset()
        self._cache = {}

    def test(self, args):
        # Create a hashable index
        try:
            index = frozenset((k, v) for k, v in args.items())
        except TypeError:
            # Non-hashable value – do not use cache
            return super().test(args)

        if index in self._cache:
            return self._cache[index]

        outcome = super().test(args)
        self._cache[index] = outcome

        return outcome

if __name__ == "__main__":
    with CachingCallReducer(log=True) as reducer:
        mystery(failing_input)

    reducer.test({'inp': failing_input})
    reducer.test({'inp': '123'})
    reducer.test({'inp': '123'})


# #### Reducing Arguments

if __name__ == "__main__":
    print('\n#### Reducing Arguments')




class DeltaDebugger(CachingCallReducer):
    def __init__(self, **args):
        super().__init__(**args)
        self._reduced_args = None

    def reduce_arg(self, var_to_be_reduced, args):
        inp = args[var_to_be_reduced]

        n = 2     # Initial granularity
        while len(inp) >= 2:
            start = 0
            subset_length = len(inp) / n
            some_complement_is_failing = False

            while start < len(inp):
                complement = inp[:int(start)] + \
                    inp[int(start + subset_length):]

                new_args = {}  # Create copy
                for var in args:
                    new_args[var] = args[var]
                new_args[var_to_be_reduced] = complement
                if self.test(new_args) == FAIL:
                    inp = complement
                    n = max(n - 1, 2)
                    some_complement_is_failing = True
                    break

                start += subset_length

            if not some_complement_is_failing:
                if n == len(inp):
                    break
                n = min(n * 2, len(inp))

        return inp

class DeltaDebugger(DeltaDebugger):
    def reducible(self, arg):
        """Return true if arg supports len() and indexing."""
        try:
            x = len(arg)
        except TypeError:
            return False

        try:
            x = arg[0]
        except TypeError:
            return False
        except IndexError:
            return False

        return True

class FailureNotReproducedError(ValueError):
    pass

class NotFailingError(ValueError):
    pass

class NoCallError(ValueError):
    pass

class DeltaDebugger(DeltaDebugger):
    def check_reproducibility(self):
        """Check whether running the function again fails"""
        self.reset()
        outcome = self.test(self._args)
        if outcome == UNRESOLVED:
            raise FailureNotReproducedError(
                "When called again, " +
                self.format_call(self._args) + 
                " raised " +
                self.format_exception(self.last_exception) +
                " instead of " +
                self.format_exception(self._exception))

        if outcome == PASS:
            raise NotFailingError("When called again, " +
                                  self.format_call(self._args) + 
                                  " did not fail")
        assert outcome == FAIL

class DeltaDebugger(DeltaDebugger):
    def reduce_args(self):
        """Reduce all reducible arguments, using reduce_arg(). Can be overloaded in subclasses"""
        args = {}  # Local copy
        for var in self._args:
            args[var] = self._args[var]
        vars_to_be_reduced = set(args.keys())

        self.check_reproducibility()

        # We take turns in reducing variables until all are processed
        while len(vars_to_be_reduced) > 0:
            for var in vars_to_be_reduced:
                value = args[var]
                if not self.reducible(value):
                    vars_to_be_reduced.remove(var)
                    break

                if self.log:
                    print(f"Reducing {var}...")

                reduced_value = self.reduce_arg(var, args)
                if len(reduced_value) < len(value):
                    args[var] = reduced_value
                    if self.log:
                        print(f"Reduced {var} to {repr(reduced_value)}")
                    vars_to_be_reduced = set(args.keys())

                vars_to_be_reduced.remove(var)
                break

        assert self.test(args) == FAIL, f"{self.format_call(args)} does not fail"
        if self.log:
            print(f"Reduced call to {self.format_call(args)}")

        self._reduced_args = args

class DeltaDebugger(DeltaDebugger):
    def diagnosis(self):
        if self._function is None:
            raise NoCallError("No function call observed")
        if self._exception is None:
            raise NotFailingError(f"{self.format_call()} did not raise an exception")

        if self.log:
            print(f"Observed {self.format_call()} raising {self.format_exception(self._exception)}")

        self.reduce_args()

class DeltaDebugger(DeltaDebugger):
    def reduced_args(self):
        """Return the dictionary {var: value} of reduced arguments."""
        return self._reduced_args

    def __repr__(self):
        return self.format_call(self.reduced_args())

# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




if __name__ == "__main__":
    with DeltaDebugger() as dd:
        mystery(failing_input)
    dd


if __name__ == "__main__":
    with DeltaDebugger(log=True) as dd:
        mystery(failing_input)
    dd


if __name__ == "__main__":
    with DeltaDebugger() as dd:
        mystery(failing_input)


if __name__ == "__main__":
    dd.args()


if __name__ == "__main__":
    dd.reduced_args()


# ## Usage Examples

if __name__ == "__main__":
    print('\n## Usage Examples')




# ### Reducing remove_html_markup()

if __name__ == "__main__":
    print('\n### Reducing remove_html_markup()')




if __package__ is None or __package__ == "":
    from Assertions import remove_html_markup
else:
    from .Assertions import remove_html_markup


if __name__ == "__main__":
    with DeltaDebugger(log=True):
        remove_html_markup('"x > y"')


# ### Reducing Multiple Arguments

if __name__ == "__main__":
    print('\n### Reducing Multiple Arguments')




def string_error(s1, s2):
    assert s1 not in s2, "no substrings"

if __name__ == "__main__":
    with DeltaDebugger(log=True) as dd:
        string_error("foo", "foobar")


if __name__ == "__main__":
    args = dd.reduced_args()
    args


if __name__ == "__main__":
    with ExpectError():
        string_error(args['s1'], args['s2'])


if __package__ is None or __package__ == "":
    from Debugger import Debugger
else:
    from .Debugger import Debugger


if __package__ is None or __package__ == "":
    from bookutils import next_inputs
else:
    from .bookutils import next_inputs


if __name__ == "__main__":
    # ignore
    next_inputs(['print', 'quit'])


if __name__ == "__main__":
    with ExpectError():
        with Debugger():
            string_error(**args)


# ### Reducing other Collections

if __name__ == "__main__":
    print('\n### Reducing other Collections')




def list_error(l1, l2, maxlen):
    assert len(l1) < len(l2) < maxlen, "invalid string length"

if __name__ == "__main__":
    with DeltaDebugger() as dd:
        list_error(l1=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], l2=[1, 2, 3], maxlen=5)
    dd


# ## Reducing Program Code

if __name__ == "__main__":
    print('\n## Reducing Program Code')




if __name__ == "__main__":
    # ignore
    try:
        del remove_html_markup
    except NameError:
        pass


if __package__ is None or __package__ == "":
    import Assertions  # minor dependency
else:
    from . import Assertions  # minor dependency


import inspect

if __name__ == "__main__":
    assertions_source_lines, _ = inspect.getsourcelines(Assertions)
    # print_content("".join(assertions_source_lines), ".py")
    assertions_source_lines[:10]


if __name__ == "__main__":
    len(assertions_source_lines)


def compile_and_run(lines):
    exec("".join(lines), {}, {})

if __name__ == "__main__":
    compile_and_run(assertions_source_lines)


def compile_and_test_html_markup(lines):
    compile_and_run(lines + 
                    ['''\nassert remove_html_markup('"foo"') == '"foo"', "My Test"\n'''])

if __name__ == "__main__":
    with ExpectError():
        compile_and_test_html_markup(assertions_source_lines)


# ### Reducing Code Lines

if __name__ == "__main__":
    print('\n### Reducing Code Lines')




if __name__ == "__main__":
    quiz("What will the reduced set of lines contain?",
         [
             "All of the source code in the assertions chapter.",
             "Only the source code of <samp>remove_html_markup()</samp>",
             "Only a subset of <samp>remove_html_markup()</samp>",
             "No lines at all."
         ], [x for x in range((1 + 1) ** (1 + 1)) if x % (1 + 1) == 1][1])


if __name__ == "__main__":
    with DeltaDebugger(log=False) as dd:
        compile_and_test_html_markup(assertions_source_lines)


if __name__ == "__main__":
    reduced_lines = dd.reduced_args()['lines']
    len(reduced_lines)


if __package__ is None or __package__ == "":
    from bookutils import print_content
else:
    from .bookutils import print_content


if __name__ == "__main__":
    print_content("".join(reduced_lines), ".py")


def compile_and_test_html_markup(lines):
    compile_and_run(lines +
        [
            '''if remove_html_markup('<foo>bar</foo>') != 'bar':\n''',
            '''    raise RuntimeError("Missing functionality")\n''',
            '''assert remove_html_markup('"foo"') == '"foo"', "My Test"\n'''
        ])

if __name__ == "__main__":
    with ExpectError():
        compile_and_test_html_markup(dd.reduced_args()['lines'])


if __name__ == "__main__":
    with DeltaDebugger(log=False) as dd:
        compile_and_test_html_markup(assertions_source_lines)
    reduced_assertions_source_lines = dd.reduced_args()['lines']


if __name__ == "__main__":
    print_content("".join(reduced_assertions_source_lines), ".py")


if __name__ == "__main__":
    len(reduced_assertions_source_lines) / len(assertions_source_lines)


if __name__ == "__main__":
    remove_html_markup_source_lines, _ = inspect.getsourcelines(Assertions.remove_html_markup)
    print_content("".join(remove_html_markup_source_lines), ".py")


if __name__ == "__main__":
    quiz("In the reduced version, what has changed?",
        [
            "Comments are deleted",
            "Blank lines are deleted",
            "Initializations are deleted",
            "The assertion is deleted",
        ], [(1 ** 0 - -1 ** 0) ** n for n in range(0, 3)]
        )


# ### Reducing Code Characters

if __name__ == "__main__":
    print('\n### Reducing Code Characters')




if __name__ == "__main__":
    reduced_assertions_source_characters = list("".join(reduced_assertions_source_lines))
    print(reduced_assertions_source_characters[:30])


if __name__ == "__main__":
    with ExpectError():
        compile_and_test_html_markup(reduced_assertions_source_characters)


if __package__ is None or __package__ == "":
    from Timer import Timer
else:
    from .Timer import Timer


if __name__ == "__main__":
    with Timer() as t:
        with DeltaDebugger(log=False) as dd:
            compile_and_test_html_markup(reduced_assertions_source_characters)


if __name__ == "__main__":
    further_reduced_assertions_source_characters = dd.reduced_args()['lines']
    print_content("".join(further_reduced_assertions_source_characters), ".py")


if __name__ == "__main__":
    dd.tests


if __name__ == "__main__":
    t.elapsed_time()


# ### Reducing Syntax Trees

if __name__ == "__main__":
    print('\n### Reducing Syntax Trees')




if __package__ is None or __package__ == "":
    from Assertions import remove_html_markup
else:
    from .Assertions import remove_html_markup


if __name__ == "__main__":
    fun_source = inspect.getsource(remove_html_markup)


if __name__ == "__main__":
    print_content(fun_source, '.py')


# #### From Code to Syntax Trees

if __name__ == "__main__":
    print('\n#### From Code to Syntax Trees')




import ast
import astor

if __name__ == "__main__":
    fun_tree = ast.parse(fun_source)


if __package__ is None or __package__ == "":
    from bookutils import rich_output
else:
    from .bookutils import rich_output


if __name__ == "__main__":
    if rich_output():
        from showast import show_ast
    else:
        def show_ast(tree):
            ast.dump(tree)


if __name__ == "__main__":
    show_ast(fun_tree)


if __name__ == "__main__":
    test_source = (
        '''if remove_html_markup('<foo>bar</foo>') != 'bar':\n''' +
        '''    raise RuntimeError("Missing functionality")\n''' +
        '''assert remove_html_markup('"foo"') == '"foo"', "My Test"''')


if __name__ == "__main__":
    test_tree = ast.parse(test_source)


if __name__ == "__main__":
    print_content(astor.to_source(test_tree), '.py')


import copy

if __name__ == "__main__":
    fun_test_tree = copy.deepcopy(fun_tree)
    fun_test_tree.body += test_tree.body


if __name__ == "__main__":
    fun_test_code = compile(fun_test_tree, '<string>', 'exec')


if __name__ == "__main__":
    with ExpectError():
        exec(fun_test_code, {}, {})


# #### Traversing Syntax Trees

if __name__ == "__main__":
    print('\n#### Traversing Syntax Trees')




from ast import NodeTransformer, NodeVisitor, fix_missing_locations

class NodeCollector(NodeVisitor):
    """Collect all nodes in an AST."""
    def __init__(self):
        super().__init__()
        self._all_nodes = []

    def generic_visit(self, node):
        self._all_nodes.append(node)
        return super().generic_visit(node)

    def collect(self, tree):
        """Return a list of all nodes in tree."""
        self._all_nodes = []
        self.visit(tree)
        return self._all_nodes

if __name__ == "__main__":
    fun_nodes = NodeCollector().collect(fun_tree)
    len(fun_nodes)


if __name__ == "__main__":
    fun_nodes[:30]


# #### Deleting Nodes

if __name__ == "__main__":
    print('\n#### Deleting Nodes')




class NodeMarker(NodeVisitor):
    def visit(self, node):
        node.marked = True
        return super().generic_visit(node)

class NodeReducer(NodeTransformer):
    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.visit_Node)
        return visitor(node)

    def visit_Module(self, node):
        # Can't remove modules
        return super().generic_visit(node)

    def visit_Node(self, node):
        """Default visitor for all nodes"""
        if node.marked:
            return None  # delete it
        return super().generic_visit(node)

def copy_and_reduce(tree, keep_list):
    """Copy tree, reducing all nodes that are not in keep_list."""

    # Mark all nodes except those in keep_list
    NodeMarker().visit(tree)
    for node in keep_list:
        # print("Clearing", node)
        node.marked = False

    # Copy tree and delete marked nodes
    new_tree = copy.deepcopy(tree)
    NodeReducer().visit(new_tree)
    return new_tree

if __name__ == "__main__":
    fun_nodes[4]


if __name__ == "__main__":
    astor.to_source(fun_nodes[4])


if __name__ == "__main__":
    keep_list = fun_nodes.copy()
    del keep_list[4]


if __name__ == "__main__":
    new_fun_tree = copy_and_reduce(fun_tree, keep_list)
    show_ast(new_fun_tree)


if __name__ == "__main__":
    print_content(astor.to_source(new_fun_tree), '.py')


if __name__ == "__main__":
    new_fun_tree.body += test_tree.body


if __name__ == "__main__":
    fun_code = compile(new_fun_tree, "<string>", 'exec')


if __name__ == "__main__":
    with ExpectError():
        exec(fun_code, {}, {})


if __name__ == "__main__":
    empty_tree = copy_and_reduce(fun_tree, [])


if __name__ == "__main__":
    astor.to_source(empty_tree)


# #### Reducing Trees

if __name__ == "__main__":
    print('\n#### Reducing Trees')




def compile_and_test_ast(tree, keep_list, test_tree=None):
    new_tree = copy_and_reduce(tree, keep_list)
    # print(astor.to_source(new_tree))

    if test_tree is not None:
        new_tree.body += test_tree.body

    try:
        code_object = compile(new_tree, '<string>', 'exec')
    except Exception:
        raise SyntaxError("Cannot compile")

    exec(code_object, {}, {})

if __name__ == "__main__":
    with ExpectError():
        compile_and_test_ast(fun_tree, fun_nodes, test_tree)


if __name__ == "__main__":
    with DeltaDebugger() as dd:
        compile_and_test_ast(fun_tree, fun_nodes, test_tree)


if __name__ == "__main__":
    reduced_nodes = dd.reduced_args()['keep_list']
    len(reduced_nodes)


if __name__ == "__main__":
    reduced_fun_tree = copy_and_reduce(fun_tree, reduced_nodes)
    show_ast(reduced_fun_tree)


if __name__ == "__main__":
    print_content(astor.to_source(reduced_fun_tree), '.py')


if __name__ == "__main__":
    dd.tests


# #### Transforming Nodes

if __name__ == "__main__":
    print('\n#### Transforming Nodes')




class NodeReducer(NodeReducer):
    PASS_TREE = ast.parse("pass").body[0]
    def visit_Assign(self, node):
        if node.marked:
            # Replace by pass
            return self.PASS_TREE
        return super().generic_visit(node)

class NodeReducer(NodeReducer):
    FALSE_TREE = ast.parse("False").body[0].value
    def visit_Compare(self, node):
        if node.marked:
            # Replace by False
            return self.FALSE_TREE
        return super().generic_visit(node)

class NodeReducer(NodeReducer):
    def visit_BoolOp(self, node):
        if node.marked:
            # Replace by left operator
            return node.values[0]
        return super().generic_visit(node)

class NodeReducer(NodeReducer):
    def visit_If(self, node):
        if node.marked:
            # Replace by body
            return node.body
        return super().generic_visit(node)

if __name__ == "__main__":
    with DeltaDebugger() as dd:
        compile_and_test_ast(fun_tree, fun_nodes, test_tree)


if __name__ == "__main__":
    reduced_nodes = dd.reduced_args()['keep_list']
    reduced_fun_tree = copy_and_reduce(fun_tree, reduced_nodes)
    print_content(astor.to_source(reduced_fun_tree), '.py')


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




def myeval(inp):
    return eval(inp)

if __name__ == "__main__":
    with ExpectError():
        myeval('1 + 2 * 3 / 0')


if __name__ == "__main__":
    with DeltaDebugger() as dd:
        myeval('1 + 2 * 3 / 0')
    dd


if __name__ == "__main__":
    dd.reduced_args()


if __name__ == "__main__":
    dd.function().__name__, dd.args()


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




# ### Exercise 1: Advanced Syntactic Code Reduction

if __name__ == "__main__":
    print('\n### Exercise 1: Advanced Syntactic Code Reduction')



