#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# "Inspecting Call Stacks" - a chapter of "The Debugging Book"
# Web site: https://www.debuggingbook.org/html/StackInspector.html
# Last change: 2025-01-13 16:13:03+01:00
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
The Debugging Book - Inspecting Call Stacks

This file can be _executed_ as a script, running all experiments:

    $ python StackInspector.py

or _imported_ as a package, providing classes, functions, and constants:

    >>> from debuggingbook.StackInspector import <identifier>

but before you do so, _read_ it and _interact_ with it at:

    https://www.debuggingbook.org/html/StackInspector.html

`StackInspector` is typically used as superclass, providing its functionality to subclasses. 

Here is an example of how to use `caller_function()`. The `test()` function invokes an internal method `caller()` of `StackInspectorDemo`, which in turn invokes `callee()`:

| Function | Class | |
| --- | --- | --- |
| `callee()` | `StackInspectorDemo` | |
| `caller()` | `StackInspectorDemo` | invokes $\uparrow$ |
| `test()` | (main) | invokes $\uparrow$ |
| -/- | (main) | invokes $\uparrow$ |

Using `caller_function()`, `callee()` determines the first caller outside a `StackInspector` class and prints it out – i.e., ``.

>>> class StackInspectorDemo(StackInspector):
>>>     def callee(self) -> None:
>>>         func = self.caller_function()
>>>         assert func.__name__ == 'test'
>>>         print(func)
>>> 
>>>     def caller(self) -> None:
>>>         self.callee()
>>> def test() -> None:
>>>     demo = StackInspectorDemo()
>>>     demo.caller()
>>> test()



Here are all methods defined in this chapter:

For more details, source, and documentation, see
"The Debugging Book - Inspecting Call Stacks"
at https://www.debuggingbook.org/html/StackInspector.html
'''


# Allow to use 'from . import <module>' when run as script (cf. PEP 366)
if __name__ == '__main__' and __package__ is None:
    __package__ = 'debuggingbook'


# Inspecting Call Stacks
# ======================

if __name__ == '__main__':
    print('# Inspecting Call Stacks')



## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



## Inspecting Call Stacks
## ----------------------

if __name__ == '__main__':
    print('\n## Inspecting Call Stacks')



if __name__ == '__main__':
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)

import inspect
import warnings

from types import FunctionType, FrameType, TracebackType

from typing import cast, Dict, Any, Tuple, Callable, Optional, Type

class StackInspector:
    """Provide functions to inspect the stack"""

    def caller_frame(self) -> FrameType:
        """Return the frame of the caller."""

        # Walk up the call tree until we leave the current class
        frame = cast(FrameType, inspect.currentframe())

        while self.our_frame(frame):
            frame = cast(FrameType, frame.f_back)

        return frame

    def our_frame(self, frame: FrameType) -> bool:
        """Return true if `frame` is in the current (inspecting) class."""
        return isinstance(frame.f_locals.get('self'), self.__class__)

class StackInspector(StackInspector):
    def caller_globals(self) -> Dict[str, Any]:
        """Return the globals() environment of the caller."""
        return self.caller_frame().f_globals

    def caller_locals(self) -> Dict[str, Any]:
        """Return the locals() environment of the caller."""
        return self.caller_frame().f_locals

Location = Tuple[Callable, int]

class StackInspector(StackInspector):
    def caller_location(self) -> Location:
        """Return the location (func, lineno) of the caller."""
        return self.caller_function(), self.caller_frame().f_lineno

class StackInspector(StackInspector):
    def search_frame(self, name: str, frame: Optional[FrameType] = None) -> \
        Tuple[Optional[FrameType], Optional[Callable]]:
        """
        Return a pair (`frame`, `item`) 
        in which the function `name` is defined as `item`.
        """
        if frame is None:
            frame = self.caller_frame()

        while frame:
            item = None
            if name in frame.f_globals:
                item = frame.f_globals[name]
            if name in frame.f_locals:
                item = frame.f_locals[name]
            if item and callable(item):
                return frame, item

            frame = cast(FrameType, frame.f_back)

        return None, None

    def search_func(self, name: str, frame: Optional[FrameType] = None) -> \
        Optional[Callable]:
        """Search in callers for a definition of the function `name`"""
        frame, func = self.search_frame(name, frame)
        return func

class StackInspector(StackInspector):
    # Avoid generating functions more than once
    _generated_function_cache: Dict[Tuple[str, int], Callable] = {}

    def create_function(self, frame: FrameType) -> Callable:
        """Create function for given frame"""
        name = frame.f_code.co_name
        cache_key = (name, frame.f_lineno)
        if cache_key in self._generated_function_cache:
            return self._generated_function_cache[cache_key]

        try:
            # Create new function from given code
            generated_function = cast(Callable,
                                      FunctionType(frame.f_code,
                                                   globals=frame.f_globals,
                                                   name=name))
        except TypeError:
            # Unsuitable code for creating a function
            # Last resort: Return some function
            generated_function = self.unknown

        except Exception as exc:
            # Any other exception
            warnings.warn(f"Couldn't create function for {name} "
                          f" ({type(exc).__name__}: {exc})")
            generated_function = self.unknown

        self._generated_function_cache[cache_key] = generated_function
        return generated_function

class StackInspector(StackInspector):
    def caller_function(self) -> Callable:
        """Return the calling function"""
        frame = self.caller_frame()
        name = frame.f_code.co_name
        func = self.search_func(name)
        if func:
            return func

        if not name.startswith('<'):
            warnings.warn(f"Couldn't find {name} in caller")

        return self.create_function(frame)

    def unknown(self) -> None:  # Placeholder for unknown functions
        pass

import traceback

class StackInspector(StackInspector):
    def is_internal_error(self, exc_tp: Type, 
                          exc_value: BaseException, 
                          exc_traceback: TracebackType) -> bool:
        """Return True if exception was raised from `StackInspector` or a subclass."""
        if not exc_tp:
            return False

        for frame, lineno in traceback.walk_tb(exc_traceback):
            if self.our_frame(frame):
                return True

        return False

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



class StackInspectorDemo(StackInspector):
    def callee(self) -> None:
        func = self.caller_function()
        assert func.__name__ == 'test'
        print(func)

    def caller(self) -> None:
        self.callee()

def test() -> None:
    demo = StackInspectorDemo()
    demo.caller()

if __name__ == '__main__':
    test()

from .ClassDiagram import display_class_hierarchy, class_tree

if __name__ == '__main__':
    display_class_hierarchy([StackInspector],
                            abstract_classes=[
                                StackInspector,
                            ],
                            public_methods=[
                                StackInspector.caller_frame,
                                StackInspector.caller_function,
                                StackInspector.caller_globals,
                                StackInspector.caller_locals,
                                StackInspector.caller_location,
                                StackInspector.search_frame,
                                StackInspector.search_func,
                                StackInspector.is_internal_error,
                                StackInspector.our_frame,
                            ],
                            project='debuggingbook')

## Lessons Learned
## ---------------

if __name__ == '__main__':
    print('\n## Lessons Learned')


