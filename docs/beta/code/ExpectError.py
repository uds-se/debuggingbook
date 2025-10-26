#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# "Error Handling" - a chapter of "The Debugging Book"
# Web site: https://www.debuggingbook.org/html/ExpectError.html
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
The Debugging Book - Error Handling

This file can be _executed_ as a script, running all experiments:

    $ python ExpectError.py

or _imported_ as a package, providing classes, functions, and constants:

    >>> from debuggingbook.ExpectError import <identifier>

but before you do so, _read_ it and _interact_ with it at:

    https://www.debuggingbook.org/html/ExpectError.html

**Note**: The examples in this section only work after the rest of the cells have been executed.

For more details, source, and documentation, see
"The Debugging Book - Error Handling"
at https://www.debuggingbook.org/html/ExpectError.html
'''


# Allow to use 'from . import <module>' when run as script (cf. PEP 366)
if __name__ == '__main__' and __package__ is None:
    __package__ = 'debuggingbook'


# Error Handling
# ==============

if __name__ == '__main__':
    print('# Error Handling')



## Catching Errors
## ---------------

if __name__ == '__main__':
    print('\n## Catching Errors')



if __name__ == '__main__':
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)

import traceback
import sys

from types import FrameType, TracebackType

from typing import Union, Optional, Callable, Any

class ExpectError:
    """Execute a code block expecting (and catching) an error."""

    def __init__(self, exc_type: Optional[type] = None, 
                 print_traceback: bool = True, mute: bool = False):
        """
        Constructor. Expect an exception of type `exc_type` (`None`: any exception).
        If `print_traceback` is set (default), print a traceback to stderr.
        If `mute` is set (default: False), do not print anything.
        """
        self.print_traceback = print_traceback
        self.mute = mute
        self.expected_exc_type = exc_type

    def __enter__(self) -> Any:
        """Begin of `with` block"""
        return self

    def __exit__(self, exc_type: type, 
                 exc_value: BaseException, tb: TracebackType) -> Optional[bool]:
        """End of `with` block"""
        if exc_type is None:
            # No exception
            return

        if (self.expected_exc_type is not None
            and exc_type != self.expected_exc_type):
            raise  # Unexpected exception

        # An exception occurred
        if self.print_traceback:
            lines = ''.join(
                traceback.format_exception(
                    exc_type,
                    exc_value,
                    tb)).strip()
        else:
            lines = traceback.format_exception_only(
                exc_type, exc_value)[-1].strip()

        if not self.mute:
            print(lines, "(expected)", file=sys.stderr)
        return True  # Ignore it

def fail_test() -> None:
    # Trigger an exception
    x = 1 / 0

if __name__ == '__main__':
    with ExpectError():
        fail_test()

if __name__ == '__main__':
    with ExpectError(print_traceback=False):
        fail_test()

if __name__ == '__main__':
    with ExpectError(ZeroDivisionError):
        fail_test()

if __name__ == '__main__':
    with ExpectError():
        with ExpectError(ZeroDivisionError):
            some_nonexisting_function()  # type: ignore

## Catching Timeouts
## -----------------

if __name__ == '__main__':
    print('\n## Catching Timeouts')



import sys
import time

from .Timeout import Timeout

class ExpectTimeout(Timeout):  # type: ignore
    """Execute a code block expecting (and catching) a timeout."""

    def __init__(self, timeout: Union[int, float],
                 print_traceback: bool = True, mute: bool = False):
        """
        Constructor. Interrupt execution after `seconds` seconds.
        If `print_traceback` is set (default), print a traceback to stderr.
        If `mute` is set (default: False), do not print anything.
        """
        super().__init__(timeout)

        self.print_traceback = print_traceback
        self.mute = mute

    def __exit__(self, exc_type: type,
                 exc_value: BaseException, tb: TracebackType) -> Optional[bool]:
        """End of `with` block"""

        super().__exit__(exc_type, exc_value, tb)

        if exc_type is None:
            return

        # An exception occurred
        if self.print_traceback:
            lines = ''.join(
                traceback.format_exception(
                    exc_type,
                    exc_value,
                    tb)).strip()
        else:
            lines = traceback.format_exception_only(
                exc_type, exc_value)[-1].strip()

        if not self.mute:
            print(lines, "(expected)", file=sys.stderr)

        return True  # Ignore exception

def long_running_test() -> None:
    print("Start")
    for i in range(10):
        time.sleep(1)
        print(i, "seconds have passed")
    print("End")

if __name__ == '__main__':
    with ExpectTimeout(5, print_traceback=False):
        long_running_test()

if __name__ == '__main__':
    with ExpectTimeout(5, print_traceback=False):
        with ExpectTimeout(3, print_traceback=False):
            long_running_test()
        long_running_test()

## Lessons Learned
## ---------------

if __name__ == '__main__':
    print('\n## Lessons Learned')



## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



if __name__ == '__main__':
    with ExpectError():
        x = 1 / 0

if __name__ == '__main__':
    with ExpectTimeout(5):
        long_running_test()
