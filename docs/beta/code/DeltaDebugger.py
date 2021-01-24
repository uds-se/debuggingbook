#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Debugging Book".
# Web site: https://www.debuggingbook.org/html/DeltaDebugger.html
# Last change: 2021-01-23 13:12:57+01:00
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


# # Reducing Failure-Inducing Inputs

if __name__ == "__main__":
    print('# Reducing Failure-Inducing Inputs')




if __name__ == "__main__":
    from bookutils import YouTubeVideo
    YouTubeVideo("6fmJ5l257bM")


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
        fuzz_input = fuzz()
        try:
            mystery(fuzz_input)
        except ValueError:
            break


if __name__ == "__main__":
    failing_input = fuzz_input
    failing_input


if __name__ == "__main__":
    len(failing_input)


if __package__ is None or __package__ == "":
    from ExpectError import ExpectError
else:
    from .ExpectError import ExpectError


if __name__ == "__main__":
    with ExpectError(ValueError):
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
    with ExpectError(ValueError):
        mystery(first_half)


if __name__ == "__main__":
    second_half = failing_input[half_length:]
    assert first_half + second_half == failing_input
    second_half


if __name__ == "__main__":
    with ExpectError(ValueError):
        mystery(second_half)


# ## Delta Debugging

if __name__ == "__main__":
    print('\n## Delta Debugging')




if __name__ == "__main__":
    quarter_length = len(failing_input) // 4
    input_without_first_quarter = failing_input[quarter_length:]
    input_without_first_quarter


if __name__ == "__main__":
    with ExpectError(ValueError):
        mystery(input_without_first_quarter)


if __name__ == "__main__":
    input_without_first_and_second_quarter = failing_input[quarter_length * 2:]
    input_without_first_and_second_quarter


if __name__ == "__main__":
    with ExpectError(ValueError):
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
    with ExpectError(ValueError):
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

import inspect

class CallCollector(object):
    """Collect an exception-raising function call f().
    Use as `with CallCollector(): f()`"""

    def __init__(self):
        """Initialize collector"""
        self.init()

    def init(self):
        """Reset for new collection."""
        self._function = None
        self._args = {}
        self._exception = None

    def traceit(self, frame, event, arg):
        """Tracing function. Collect first call, then turn tracing off."""
        if event == 'call':
            name = frame.f_code.co_name
            if name.startswith('__'):
                # Internal function
                return
            if self._function is not None:
                # Already set
                return

            if name in frame.f_globals:
                # Access exactly this function
                self._function = frame.f_globals[name]
            elif name in frame.f_locals:
                self._function = frame.f_locals[name]
            else:
                # Create new function from given code
                self._function = FunctionType(frame.f_code,
                                              globals=frame.f_globals,
                                              name=name)

            self._args = {}  # Create a local copy of args
            for var in frame.f_locals:
                self._args[var] = frame.f_locals[var]

            # Turn tracing off
            sys.settrace(self.original_trace_function)

    def after_collection(self):
        """Called after collection. To be defined in subclasses."""
        pass

    def args(self):
        """Return the dictionary of collected arguments."""
        return self._args

    def function(self):
        """Return the function called."""
        return self._function

    def exception(self):
        """Return the exception produced."""
        return self._exception

    def __enter__(self):
        """Called at begin of `with` block. Turn tracing on."""
        self.init()
        self.original_trace_function = sys.gettrace()
        sys.settrace(self.traceit)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Called at end of `with` block. Turn tracing off."""
        sys.settrace(self.original_trace_function)
        if self._function is None:
            return False  # Re-raise exception, if any

        self._exception = exc_value
        self.after_collection()
        return True  # Ignore exception

if __name__ == "__main__":
    with CallCollector() as call_collector:
        mystery(failing_input)


if __name__ == "__main__":
    call_collector.function()


if __name__ == "__main__":
    call_collector.args()


if __name__ == "__main__":
    call_collector.exception()


if __name__ == "__main__":
    with ExpectError(NameError):
        with CallCollector() as c:
            some_error()


# #### Repeating a Call

if __name__ == "__main__":
    print('\n#### Repeating a Call')




if __name__ == "__main__":
    call_collector.function()("foo")


if __name__ == "__main__":
    with ExpectError(ValueError):
        call_collector.function()(failing_input)


if __name__ == "__main__":
    with ExpectError(ValueError):
        call_collector.function()(**call_collector.args())


class CallCollector(CallCollector):
    def call(self, new_args={}):
        """Call collected function. If new_args is given,
        override arguments from its {var: value} entries."""
        args = {}  # Create local copy
        for var in self.args():
            args[var] = self.args()[var]
        for var in new_args:
            args[var] = new_args[var]

        return self.function()(**args)

if __name__ == "__main__":
    with CallCollector() as call_collector:
        mystery(failing_input)
    with ExpectError(ValueError):
        call_collector.call()


if __name__ == "__main__":
    call_collector.call({'inp': 'foo'})


class CallCollector(CallCollector):
    def format_call(self, args=None):
        """Return a string representing a call of the function with given args."""
        if args is None:
            args = self.args()
        return self.function().__name__ + "(" + \
            ", ".join(f"{arg}={repr(args[arg])}" for arg in args) + ")"

    def format_exception(self, exc=None):
        """Return a string representing the given exception."""
        if exc is None:
            exc = self.exception()
        s = type(exc).__name__
        if str(exc):
            s += ": " + str(exc)
        return s

if __name__ == "__main__":
    with CallCollector() as call_collector:
        mystery(failing_input)


if __name__ == "__main__":
    call_collector.format_call()


if __name__ == "__main__":
    call_collector.format_exception()


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
            if (type(exc) == type(self.exception()) and
                    str(exc) == str(self.exception())):
                return FAIL
            else:
                return UNRESOLVED  # Some other failure

        self.last_result = result
        return PASS

class CallReducer(CallReducer):
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

    def init(self):
        super().init()
        self._cache = {}

    def test(self, args):
        # Create a hashable index
        try:
            index = frozenset((k, v) for k, v in args.items())
        except TypeError:
            index = None

        if index is None:
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


# #### General Delta Debugging

if __name__ == "__main__":
    print('\n#### General Delta Debugging')




def to_set(inp):
    """Convert inp into a set of indices"""
    return set(range(len(inp)))

if __name__ == "__main__":
    to_set("abcd")


def empty(inp):
    """Return an "empty" element of the same type as inp"""
    return type(inp)()

if __name__ == "__main__":
    empty("abc"), empty([1, 2, 3]), empty({0, -1, -2})


def add_to(collection, elem):
    """Add element to collection; return new collection."""
    if isinstance(collection, str):
        return collection + elem  # Strings

    try:  # Lists and other collections
        return collection + type(collection)([elem])
    except TypeError:
        pass

    try:  # Sets
        return collection | type(collection)([elem])
    except TypeError:
        pass

    raise ValueError("Cannot add element to collection")

if __name__ == "__main__":
    add_to("abc", "d"), add_to([1, 2, 3], 4), add_to(set([1, 2, 3]), 4)


def from_set(the_set, inp):
    """Convert a set of indices into inp back into a collection."""
    ret = empty(inp)
    for i, c in enumerate(inp):
        if i in the_set:
            ret = add_to(ret, c)
    return ret

if __name__ == "__main__":
    from_set({1, 2}, "abcd")


def split(elems, n):
    assert 1 <= n <= len(elems)

    k, m = divmod(len(elems), n)
    try:
        subsets = list(elems[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
                       for i in range(n))
    except TypeError:
        # Convert to list and back
        subsets = list(type(elems)(
                    list(elems)[i * k + min(i, m):(i + 1) * k + min(i + 1, m)])
                       for i in range(n))

    assert len(subsets) == n
    assert sum(len(subset) for subset in subsets) == len(elems)
    assert all(len(subset) > 0 for subset in subsets)

    return subsets

if __name__ == "__main__":
    for n in range(1, 8):
        print(split([1, 2, 3, 4, 5, 6, 7], n))


if __name__ == "__main__":
    split("abcd", 3)


if __name__ == "__main__":
    split({1, 2, 3, 4, 5, 6, 7}, 3)


class DeltaDebugger(CachingCallReducer):
    def dd(self, var_to_be_reduced, fail_args, mode='-'):
        """General Delta Debugging.
        fail_args - a dict of (failure-inducing) function arguments;
        fail_args[var_to_be_reduced] - the input to apply dd on.
        mode tells how the algorithm should operate:
            '-': minimize input (ddmin),
            '+': maximizing input (ddmax),
            '+-': minimizing pass/fail difference (dd)
        Returns a triple (pass, fail, diff) 
        with maximized passing input, minimized failing input,
        and their difference (elems in fail, but not in pass).
        """
        def test(c):
            # Set up args
            test_args = {}
            for var in fail_args:
                test_args[var] = fail_args[var]
            test_args[var_to_be_reduced] = from_set(c, fail_inp)
            return self.test(test_args)

        def ret(c_pass, c_fail):
            return (from_set(c_pass, fail_inp),
                    from_set(c_fail, fail_inp),
                    from_set(c_fail - c_pass, fail_inp))

        n = 2  # Initial granularity

        fail_inp = fail_args[var_to_be_reduced]

        c_pass = to_set([])
        c_fail = to_set(fail_inp)
        offset = 0

        minimize_fail = '-' in mode
        maximize_pass = '+' in mode

        while True:
            if self.log > 1:
                print("Passing input:", repr(from_set(c_pass, fail_inp)))
                print("Failing input:", repr(from_set(c_fail, fail_inp)))

            delta = c_fail - c_pass
            if len(delta) < n:
                return ret(c_pass, c_fail)

            deltas = split(delta, n)

            j = 0
            while j < n:
                i = (j + offset) % n
                next_c_pass = c_pass | deltas[i]
                next_c_fail = c_fail - deltas[i]

                if minimize_fail and n == 2 and test(next_c_pass) == FAIL:
                    if self.log > 1:
                        print("Reduce to subset")
                    c_fail = next_c_pass
                    # n = 2
                    offset = 0
                    break
                elif maximize_pass and n == 2 and test(next_c_fail) == PASS:
                    if self.log > 1:
                        print("Increase to subset")
                    c_pass = next_c_fail
                    # n = 2
                    offset = 0
                    break
                elif minimize_fail and test(next_c_fail) == FAIL:
                    if self.log > 1:
                        print("Reduce to complement")
                    c_fail = next_c_fail
                    n = max(n - 1, 2)
                    offset = i
                    break
                elif maximize_pass and test(next_c_pass) == PASS:
                    if self.log > 1:
                        print("Increase to complement")
                    c_pass = next_c_pass
                    n = max(n - 1, 2)
                    offset = i
                    break
                else:
                    j += 1  # choose next subset

            if j >= n:  # no reduction found
                if n >= len(delta):
                    return ret(c_pass, c_fail)

                n = min(n + 2, len(delta))

if __name__ == "__main__":
    with DeltaDebugger() as dd:
        mystery(failing_input)


if __name__ == "__main__":
    dd.dd('inp', {'inp': failing_input})


if __name__ == "__main__":
    with DeltaDebugger(log=2) as dd:
        mystery(failing_input)


if __name__ == "__main__":
    dd.dd('inp', {'inp': failing_input})


# #### Processing Multiple Arguments

if __name__ == "__main__":
    print('\n#### Processing Multiple Arguments')




class DeltaDebugger(DeltaDebugger):
    def reducible(self, arg):
        # Return True if `arg` supports len() and indexing.
        try:
            _ = len(arg)
        except TypeError:
            return False

        try:
            _ = arg[0]
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
        # Check whether running the function again fails
        assert self.function(), \
            "No call collected. Use `with dd: func()` first."
        assert self.args(), \
            "No arguments collected. Use `with dd: func(args)` first."

        self.reset()
        outcome = self.test(self.args())
        if outcome == UNRESOLVED:
            raise FailureNotReproducedError(
                "When called again, " +
                self.format_call(self.args()) + 
                " raised " +
                self.format_exception(self.last_exception) +
                " instead of " +
                self.format_exception(self.exception()))

        if outcome == PASS:
            raise NotFailingError("When called again, " +
                                  self.format_call(self.args()) + 
                                  " did not fail")
        assert outcome == FAIL

class DeltaDebugger(DeltaDebugger):
    def process_args(self, strategy, **strategy_args):
        """Reduce all reducible arguments, using strategy(var, args).
        Can be overloaded in subclasses."""

        pass_args = {}  # Local copy
        fail_args = {}  # Local copy
        diff_args = {}
        for var in self.args():
            fail_args[var] = self.args()[var]
            diff_args[var] = self.args()[var]
            pass_args[var] = self.args()[var]

            if self.reducible(pass_args[var]):
                pass_args[var] = empty(pass_args[var])

        vars_to_be_processed = set(fail_args.keys())

        pass_processed = 0
        fail_processed = 0

        self.check_reproducibility()

        # We take turns in processing variables until all are processed
        while len(vars_to_be_processed) > 0:
            for var in vars_to_be_processed:
                if not self.reducible(fail_args[var]):
                    vars_to_be_processed.remove(var)
                    break

                if self.log:
                    print(f"Processing {var}...")

                maximized_pass_value, minimized_fail_value, diff = \
                    strategy(var, fail_args, **strategy_args)

                if (maximized_pass_value is not None and 
                    len(maximized_pass_value) > len(pass_args[var])):
                    pass_args[var] = maximized_pass_value
                    # FIXME: diff_args may not be correct for multiple args
                    diff_args[var] = diff
                    if self.log:
                        print(f"Maximized {var} to",
                              repr(maximized_pass_value))
                    vars_to_be_processed = set(fail_args.keys())
                    pass_processed += 1

                if (minimized_fail_value is not None and 
                    len(minimized_fail_value) < len(fail_args[var])):
                    fail_args[var] = minimized_fail_value
                    diff_args[var] = diff
                    if self.log:
                        print(f"Minimized {var} to",
                              repr(minimized_fail_value))
                    vars_to_be_processed = set(fail_args.keys())
                    fail_processed += 1

                vars_to_be_processed.remove(var)
                break

        assert pass_processed == 0 or self.test(pass_args) == PASS, \
            f"{self.format_call(pass_args)} does not pass"
        assert fail_processed == 0 or self.test(fail_args) == FAIL, \
            f"{self.format_call(fail_args)} does not fail"

        if self.log and pass_processed > 0:
            print("Maximized passing call to",
                  self.format_call(pass_args))
        if self.log and fail_processed > 0:
            print("Minimized failing call to",
                  self.format_call(fail_args))

        return pass_args, fail_args, diff_args

class DeltaDebugger(DeltaDebugger):
    def after_collection(self):
        # Some post-collection checks
        if self.function() is None:
            raise NoCallError("No function call observed")
        if self.exception() is None:
            raise NotFailingError(f"{self.format_call()} did not raise an exception")

        if self.log:
            print(f"Observed {self.format_call()}" +
                  f" raising {self.format_exception(self.exception())}")

# #### Public API

if __name__ == "__main__":
    print('\n#### Public API')




class DeltaDebugger(DeltaDebugger):
    def min_args(self):
        """Return 1-minimal arguments."""
        pass_args, fail_args, diff = self.process_args(self.dd, mode='-')
        return fail_args

class DeltaDebugger(DeltaDebugger):
    def max_args(self):
        """Return 1-maximal arguments."""
        pass_args, fail_args, diff = self.process_args(self.dd, mode='+')
        return pass_args

class DeltaDebugger(DeltaDebugger):
    def min_arg_diff(self):
        """Return 1-minimal difference between arguments."""
        return self.process_args(self.dd, mode='+-')

class DeltaDebugger(DeltaDebugger):
    def __repr__(self):
        """Return a string representation of the minimized call."""
        return self.format_call(self.min_args())

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
    dd.min_args()


if __name__ == "__main__":
    quiz("What happens if the function under test does not raise an exception?",
        [
            "Delta debugging searches for the minimal input"
            " that produces the same result",
            "Delta debugging starts a fuzzer to find an exception",
            "Delta debugging raises an exception"
            "Delta debugging runs forever in a loop",
        ], 0 ** 0 + 1 ** 0 + 0 ** 1 + 1 ** 1)


if __name__ == "__main__":
    with ExpectError(NotFailingError):
        with DeltaDebugger() as dd:
            mystery("An input that does not fail")


# ## Usage Examples

if __name__ == "__main__":
    print('\n## Usage Examples')




# ### Reducing remove_html_markup()

if __name__ == "__main__":
    print('\n### Reducing remove_html_markup()')




if __package__ is None or __package__ == "":
    from Assertions import remove_html_markup  # minor dependency
else:
    from .Assertions import remove_html_markup  # minor dependency


if __name__ == "__main__":
    with DeltaDebugger(log=True) as dd:
        remove_html_markup('"x > y"')
    dd.min_args()


# ### Reducing Multiple Arguments

if __name__ == "__main__":
    print('\n### Reducing Multiple Arguments')




def string_error(s1, s2):
    assert s1 not in s2, "no substrings"

if __name__ == "__main__":
    with DeltaDebugger(log=True) as dd:
        string_error("foo", "foobar")
    dd.min_args()


if __name__ == "__main__":
    string_error_args = dd.min_args()
    string_error_args


if __name__ == "__main__":
    with ExpectError(AssertionError):
        string_error(string_error_args['s1'], string_error_args['s2'])


if __package__ is None or __package__ == "":
    from Debugger import Debugger  # minor dependency
else:
    from .Debugger import Debugger  # minor dependency


if __package__ is None or __package__ == "":
    from bookutils import next_inputs
else:
    from .bookutils import next_inputs


if __name__ == "__main__":
    next_inputs(['print', 'quit'])


if __name__ == "__main__":
    with ExpectError(AssertionError):
        with Debugger():
            string_error(**string_error_args)


# ### Reducing other Collections

if __name__ == "__main__":
    print('\n### Reducing other Collections')




def list_error(l1, l2, maxlen):
    assert len(l1) < len(l2) < maxlen, "invalid string length"

if __name__ == "__main__":
    with DeltaDebugger() as dd:
        list_error(l1=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], l2=[1, 2, 3], maxlen=5)
    dd


# ## Debugging Inputs

if __name__ == "__main__":
    print('\n## Debugging Inputs')




if __name__ == "__main__":
    with DeltaDebugger(log=True) as dd:
        mystery(failing_input)
    max_passing_input = dd.max_args()['inp']
    max_passing_input


# ## Failure-Inducing Differences

if __name__ == "__main__":
    print('\n## Failure-Inducing Differences')




if __name__ == "__main__":
    with DeltaDebugger(log=True) as dd:
        mystery(failing_input)
    max_passing_args, min_failing_args, diff = dd.min_arg_diff()
    max_passing_args['inp'], min_failing_args['inp'], diff['inp']


if __name__ == "__main__":
    diff['inp']


# ## Reducing Program Code

if __name__ == "__main__":
    print('\n## Reducing Program Code')




if __name__ == "__main__":
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


if __package__ is None or __package__ == "":
    from Assertions import remove_html_markup  # minor dependency
else:
    from .Assertions import remove_html_markup  # minor dependency


def compile_and_test_html_markup(lines):
    compile_and_run(lines + 
                    ['''\nassert remove_html_markup('"foo"') == '"foo"', "My Test"\n'''])

if __name__ == "__main__":
    with ExpectError(AssertionError):
        compile_and_test_html_markup(assertions_source_lines)


# ### Reducing Code Lines

if __name__ == "__main__":
    print('\n### Reducing Code Lines')




if __name__ == "__main__":
    quiz("What will the reduced set of lines contain?",
         [
             "All of the source code in the assertions chapter.",
             "Only the source code of `remove_html_markup()`",
             "Only a subset of `remove_html_markup()`",
             "No lines at all."
         ], [x for x in range((1 + 1) ** (1 + 1)) if x % (1 + 1) == 1][1])


if __name__ == "__main__":
    with DeltaDebugger(log=False) as dd:
        compile_and_test_html_markup(assertions_source_lines)


if __name__ == "__main__":
    reduced_lines = dd.min_args()['lines']
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
    with ExpectError(RuntimeError):
        compile_and_test_html_markup(dd.min_args()['lines'])


if __name__ == "__main__":
    with DeltaDebugger(log=False) as dd:
        compile_and_test_html_markup(assertions_source_lines)
    reduced_assertions_source_lines = dd.min_args()['lines']


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
    with ExpectError(AssertionError):
        compile_and_test_html_markup(reduced_assertions_source_characters)


if __package__ is None or __package__ == "":
    from Timer import Timer
else:
    from .Timer import Timer


if __name__ == "__main__":
    with DeltaDebugger(log=False) as dd:
        compile_and_test_html_markup(reduced_assertions_source_characters)


if __name__ == "__main__":
    with Timer() as t:
        further_reduced_assertions_source_characters = dd.min_args()['lines']
    print_content("".join(further_reduced_assertions_source_characters), ".py")


if __name__ == "__main__":
    dd.tests


if __name__ == "__main__":
    t.elapsed_time()


# ### Reducing Syntax Trees

if __name__ == "__main__":
    print('\n### Reducing Syntax Trees')




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
    from bookutils import show_ast
else:
    from .bookutils import show_ast


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
    with ExpectError(AssertionError):
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
    with ExpectError(UnboundLocalError):
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
    with ExpectError(AssertionError):
        compile_and_test_ast(fun_tree, fun_nodes, test_tree)


if __name__ == "__main__":
    with DeltaDebugger() as dd:
        compile_and_test_ast(fun_tree, fun_nodes, test_tree)


if __name__ == "__main__":
    reduced_nodes = dd.min_args()['keep_list']
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
    reduced_nodes = dd.min_args()['keep_list']
    reduced_fun_tree = copy_and_reduce(fun_tree, reduced_nodes)
    print_content(astor.to_source(reduced_fun_tree), '.py')


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




def myeval(inp):
    return eval(inp)

if __name__ == "__main__":
    with ExpectError(ZeroDivisionError):
        myeval('1 + 2 * 3 / 0')


if __name__ == "__main__":
    with DeltaDebugger() as dd:
        myeval('1 + 2 * 3 / 0')
    dd


if __name__ == "__main__":
    dd.min_args()


if __name__ == "__main__":
    dd.max_args()


if __name__ == "__main__":
    dd.min_arg_diff()


if __name__ == "__main__":
    dd.function().__name__, dd.args()


if __package__ is None or __package__ == "":
    from ClassDiagram import display_class_hierarchy
else:
    from .ClassDiagram import display_class_hierarchy


if __name__ == "__main__":
    display_class_hierarchy([DeltaDebugger],
                            public_methods=[
                                CallCollector.__init__,
                                CallCollector.__enter__,
                                CallCollector.__exit__,
                                CallCollector.function,
                                CallCollector.args,
                                CallCollector.exception,
                                CallCollector.call,
                                CallReducer.__init__,
                                CallReducer.reduce_arg,
                                DeltaDebugger.dd,
                                DeltaDebugger.min_args,
                                DeltaDebugger.max_args,
                                DeltaDebugger.min_arg_diff,
                                DeltaDebugger.__repr__
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




# ### Exercise 1: Advanced Syntactic Code Reduction

if __name__ == "__main__":
    print('\n### Exercise 1: Advanced Syntactic Code Reduction')



