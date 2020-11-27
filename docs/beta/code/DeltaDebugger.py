#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Fuzzing Book".
# Web site: https://www.fuzzingbook.org/html/DeltaDebugger.html
# Last change: 2020-11-27 20:18:59+01:00
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
    from bookutils import quiz
else:
    from .bookutils import quiz


import re

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


# ## A Simple Interface

if __name__ == "__main__":
    print('\n## A Simple Interface')




# ### Collecting a Call

if __name__ == "__main__":
    print('\n### Collecting a Call')




import sys

from types import FunctionType

class CallCollector(object):
    def __init__(self):
        """Reduce a function call."""
        self._function = None
        self._args = None
        self._exception = None

    def traceit(self, frame, event, arg):
        """Tracing function. Collect first call."""
        if event == 'call':
            name = frame.f_code.co_name
            if name.startswith('__'):
                # Internal function
                return

            self._function = FunctionType(frame.f_code,
                                          globals=globals(),
                                          name=name)
            self._args = frame.f_locals

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

    def __exit__(self, exc_type, exc_value, traceback):
        """Called at end of `with` block. Turn tracing off."""
        sys.settrace(self.original_trace_function)
        if exc_type is not None and self._function is None:
            raise exc_value

        self._exception = exc_value
        self.diagnosis()
        return True  # Ignore exception

if __name__ == "__main__":
    call_collector = CallCollector()
    with call_collector:
        mystery(failing_input)


if __name__ == "__main__":
    call_collector._function


if __name__ == "__main__":
    call_collector._args


if __name__ == "__main__":
    call_collector._exception


if __name__ == "__main__":
    with ExpectError():
        c = CallCollector()
        with c:
            some_error()


# ### Repeating a Call

if __name__ == "__main__":
    print('\n### Repeating a Call')




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
        args = {}
        for var in self._args:
            args[var] = self._args[var]
        for var in new_args:
            args[var] = new_args[var]

        return self._function(**new_args)

if __name__ == "__main__":
    call_collector = CallCollector()
    with call_collector:
        mystery(failing_input)


if __name__ == "__main__":
    with ExpectError():
        call_collector.call({'inp': 'foo'})


# ### Reducing Inputs

if __name__ == "__main__":
    print('\n### Reducing Inputs')




class Reducer(CallCollector):
    def __init__(self, log=False):
        super().__init__()
        self.log = log
        self.reset()

    def reset(self):
        self.tests = 0

    def run(self, args):
        try:
            result = self.call(args)
        except Exception as exc:
            self.last_exception = exc
            if type(exc) == type(self._exception) and str(exc) == str(self._exception):
                return FAIL
            else:
                return UNRESOLVED  # Some other failure

        self.last_result = result
        return PASS

    def format_call(self, args=None):
        if args is None:
            args = self._args
        return self._function.__name__ + "(" + \
            ", ".join(f"{arg}={repr(args[arg])}" for arg in args) + ")"

    def test(self, args):
        outcome = self.run(args)
        if outcome == PASS:
            detail = ""
        else:
            detail = type(self.last_exception).__name__
            if str(self.last_exception):
                detail += ": " + str(self.last_exception)
            detail = f" ({detail})"

        self.tests += 1
        if self.log:
            print(f"Test #{self.tests} {self.format_call(args)}: {outcome}{detail}")

        return outcome

class CachingReducer(Reducer):
    def reset(self):
        super().reset()
        self.cache = {}

    def test(self, args):
        index = ((k, v) for k, v in args.items())
        if index in self.cache:
            return self.cache[index]

        outcome = super().test(args)
        self.cache[index] = outcome
        return outcome

class DeltaDebugger(CachingReducer):
    def __init__(self, show=True, **args):
        super().__init__(**args)
        self.show_diagnosis = show
        self._reduced_args = None

    def reduce(self, var_to_be_reduced, args):
        self.reset()
        assert self.test(args) != PASS, f"{self.format_call(args)} did not pass"
        inp = args[var_to_be_reduced]

        n = 2     # Initial granularity
        while len(inp) >= 2:
            start = 0
            subset_length = len(inp) / n
            some_complement_is_failing = False

            while start < len(inp):
                complement = inp[:int(start)] + \
                    inp[int(start + subset_length):]

                new_args = {}
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

class DeltaDebugger(DeltaDebugger):
    def reduced_args(self):
        if self._reduced_args is not None:
            return self._reduced_args

        args = {}
        for var in self._args:
            args[var] = self._args[var]
        vars_to_be_reduced = set(args.keys())
        
        while len(vars_to_be_reduced) > 0:
            for var in vars_to_be_reduced:
                value = args[var]
                if not self.reducible(value):
                    vars_to_be_reduced.remove(var)
                    break
                if self.log:
                    print(f"Reducing {var}...")
                reduced_value = self.reduce(var, args)
                if len(reduced_value) < len(value):
                    args[var] = reduced_value
                    vars_to_be_reduced = set(args.keys())
                vars_to_be_reduced.remove(var)
                break

        assert self.test(args) == FAIL, f"{self.format_call(args)} does not fail"
        self._reduced_args = args
        return args

class DeltaDebugger(DeltaDebugger):
    def diagnosis(self):
        if self._function is None:
            raise ValueError("No function call observed")
        if self._exception is None:
            raise ValueError(f"{self.format_call()} did not raise an exception")

        reduced_args = self.reduced_args()
        if self.show_diagnosis:
            print(self.format_call(reduced_args))
        return reduced_args

if __name__ == "__main__":
    with DeltaDebugger():
        mystery(failing_input)


if __name__ == "__main__":
    with DeltaDebugger(log=True):
        mystery(failing_input)


if __name__ == "__main__":
    dd = DeltaDebugger(show=False)
    with dd:
        mystery(failing_input)


if __name__ == "__main__":
    dd.args()


if __name__ == "__main__":
    dd.reduced_args()


if __package__ is None or __package__ == "":
    from Assertions import remove_html_markup
else:
    from .Assertions import remove_html_markup


if __name__ == "__main__":
    with DeltaDebugger():
        remove_html_markup('"x > y"')


def string_error(s1, s2):
    assert s1 not in s2

if __name__ == "__main__":
    with DeltaDebugger(log=True):
        string_error("foo", "foobar")


def list_error(l1, l2, maxlen):
    assert len(l1) < len(l2) < maxlen

if __name__ == "__main__":
    with DeltaDebugger():
        list_error(l1=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], l2=[1, 2, 3], maxlen=5)


# ## Reducing Program Code

if __name__ == "__main__":
    print('\n## Reducing Program Code')




import inspect

if __name__ == "__main__":
    try:
        del remove_html_markup
    except NameError:
        pass


if __package__ is None or __package__ == "":
    import Assertions
else:
    from . import Assertions


if __name__ == "__main__":
    assertions_source_code, _ = inspect.getsourcelines(Assertions)
    assertions_source_code[:5]


if __name__ == "__main__":
    len(assertions_source_code)


def compile_and_run(lines):
    exec("".join(lines))

def compile_and_test_html_markup(lines):
    compile_and_run(lines + 
                    ['''\nassert remove_html_markup('"foo"') == '"foo"', "My Test"\n'''])

if __name__ == "__main__":
    with ExpectError():
        compile_and_test_html_markup(assertions_source_code)


if __name__ == "__main__":
    dd = DeltaDebugger(log=False, show=False)
    with dd:
        compile_and_test_html_markup(assertions_source_code)


if __name__ == "__main__":
    len(dd.reduced_args()['lines'])


if __package__ is None or __package__ == "":
    from bookutils import print_content
else:
    from .bookutils import print_content


if __name__ == "__main__":
    print_content("".join(dd.reduced_args()['lines']), ".py")


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
    dd = DeltaDebugger(log=False, show=False)
    with dd:
        compile_and_test_html_markup(assertions_source_code)


if __name__ == "__main__":
    print_content("".join(dd.reduced_args()['lines']), ".py")


if __name__ == "__main__":
    remove_html_markup_source_code, _ = inspect.getsourcelines(Assertions.remove_html_markup)
    print_content("".join(remove_html_markup_source_code), ".py")


if __name__ == "__main__":
    quiz("In the reduced version, what has changed?",
        [
            "Comments are deleted",
            "Blank lines are deleted",
            "Initializations are deleted",
            "The assertion is deleted",
        ], [2 ** n for n in range(0, 3)]
        )


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




def myeval(inp):
    return eval(inp)

if __name__ == "__main__":
    with ExpectError():
        myeval('1 + 2 * 3 / 0')


if __name__ == "__main__":
    with DeltaDebugger():
        myeval('1 + 2 * 3 / 0')


if __name__ == "__main__":
    dd = DeltaDebugger(show=False)
    with dd:
        myeval('1 + 2 * 3 / 0')
    dd.reduced_args()


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




# ### Exercise 1: Syntactic Code Reduction

if __name__ == "__main__":
    print('\n### Exercise 1: Syntactic Code Reduction')



