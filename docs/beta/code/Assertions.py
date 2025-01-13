#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# "Asserting Expectations" - a chapter of "The Debugging Book"
# Web site: https://www.debuggingbook.org/html/Assertions.html
# Last change: 2025-01-13 15:53:54+01:00
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
The Debugging Book - Asserting Expectations

This file can be _executed_ as a script, running all experiments:

    $ python Assertions.py

or _imported_ as a package, providing classes, functions, and constants:

    >>> from debuggingbook.Assertions import <identifier>

but before you do so, _read_ it and _interact_ with it at:

    https://www.debuggingbook.org/html/Assertions.html

This chapter discusses _assertions_ to define _assumptions_ on function inputs and results:

>>> def my_square_root(x):  # type: ignore
>>>     assert x >= 0
>>>     y = square_root(x)
>>>     assert math.isclose(y * y, x)
>>>     return y

Notably, assertions detect _violations_ of these assumptions at runtime:

>>> with ExpectError():
>>>     y = my_square_root(-1)
Traceback (most recent call last):
  File "/var/folders/n2/xd9445p97rb3xh7m1dfx8_4h0006ts/T/ipykernel_2156/76616918.py", line 2, in 
    y = my_square_root(-1)
        ^^^^^^^^^^^^^^^^^^
  File "/var/folders/n2/xd9445p97rb3xh7m1dfx8_4h0006ts/T/ipykernel_2156/2617682038.py", line 2, in my_square_root
    assert x >= 0
           ^^^^^^
AssertionError (expected)


_System assertions_ help to detect invalid memory operations.

>>> managed_mem = ManagedMemory()
>>> managed_mem

|Address|0|1|2|3|4|5|6|7|8|9|
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|Allocated| | | | | | | | | | |
|Initialized| | | | | | | | | | |
|Content|-1|0| | | | | | | | |


>>> with ExpectError():
>>>     x = managed_mem[2]
Traceback (most recent call last):
  File "/var/folders/n2/xd9445p97rb3xh7m1dfx8_4h0006ts/T/ipykernel_2156/1296110967.py", line 2, in 
    x = managed_mem[2]
        ~~~~~~~~~~~^^^
  File "/var/folders/n2/xd9445p97rb3xh7m1dfx8_4h0006ts/T/ipykernel_2156/2465984283.py", line 3, in __getitem__
    return self.read(address)
           ^^^^^^^^^^^^^^^^^^
  File "/var/folders/n2/xd9445p97rb3xh7m1dfx8_4h0006ts/T/ipykernel_2156/2898840933.py", line 9, in read
    assert self.allocated[address], \
           ~~~~~~~~~~~~~~^^^^^^^^^
AssertionError: Reading from unallocated memory (expected)



For more details, source, and documentation, see
"The Debugging Book - Asserting Expectations"
at https://www.debuggingbook.org/html/Assertions.html
'''


# Allow to use 'from . import <module>' when run as script (cf. PEP 366)
if __name__ == '__main__' and __package__ is None:
    __package__ = 'debuggingbook'


# Asserting Expectations
# ======================

if __name__ == '__main__':
    print('# Asserting Expectations')



if __name__ == '__main__':
    from .bookutils import YouTubeVideo
    YouTubeVideo("9mI9sbKFkwU")

if __name__ == '__main__':
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)

from .bookutils import quiz

from . import Tracer

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



## Introducing Assertions
## ----------------------

if __name__ == '__main__':
    print('\n## Introducing Assertions')



### Assertions

if __name__ == '__main__':
    print('\n### Assertions')



if __name__ == '__main__':
    assert True

from .ExpectError import ExpectError

if __name__ == '__main__':
    with ExpectError():
        assert False

def test_square_root() -> None:
    assert square_root(4) == 2
    assert square_root(9) == 3
    ...

def my_own_assert(cond: bool) -> None:
    if not cond:
        raise AssertionError

if __name__ == '__main__':
    with ExpectError():
        my_own_assert(2 + 2 == 5)

### Assertion Diagnostics

if __name__ == '__main__':
    print('\n### Assertion Diagnostics')



if __name__ == '__main__':
    open('testassert.c', 'w').write(r'''
#include <stdio.h>
#include "assert.h"

int main(int argc, char *argv[]) {
    assert(2 + 2 == 5);
    printf("Foo\n");
}
''');

from .bookutils import print_content

if __name__ == '__main__':
    print_content(open('testassert.c').read(), '.h')

if __name__ == '__main__':
    import os
    os.system(f'cc -g -o testassert testassert.c')

if __name__ == '__main__':
    import os
    os.system(f'./testassert')

if __name__ == '__main__':
    open('assert.h', 'w').write(r'''
#include <stdio.h>
#include <stdlib.h>

#ifndef NDEBUG
#define assert(cond) \
    if (!(cond)) { \
        fprintf(stderr, "Assertion failed: %s, function %s, file %s, line %d", \
            #cond, __func__, __FILE__, __LINE__); \
        exit(1); \
    }
#else
#define assert(cond) ((void) 0)
#endif
''');

if __name__ == '__main__':
    print_content(open('assert.h').read(), '.h')

if __name__ == '__main__':
    import os
    os.system(f'cc -DNDEBUG -g -o testassert testassert.c')

if __name__ == '__main__':
    import os
    os.system(f'./testassert')

if __name__ == '__main__':
    import os
    os.system(f'python -c \'assert 2 + 2 == 5; print("Foo")\'')

if __name__ == '__main__':
    import os
    os.system(f'python -O -c \'assert 2 + 2 == 5; print("Foo")\'')

def fun() -> None:
    assert 2 + 2 == 5

if __name__ == '__main__':
    quiz("If we invoke `fun()` and the assertion fails,"
         " which information do we get?",
         [
             "The failing condition (`2 + 2 == 5`)",
             "The location of the assertion in the program",
             "The list of callers",
             "All of the above"
         ], '123456789 % 5')

if __name__ == '__main__':
    with ExpectError():
        fun()

## Checking Preconditions
## ----------------------

if __name__ == '__main__':
    print('\n## Checking Preconditions')



def square_root(x):  # type: ignore
    assert x >= 0
    ...   # compute square root in y

if __name__ == '__main__':
    with ExpectError():
        square_root(-1)

def square_root(x):  # type: ignore
    assert isinstance(x, (int, float))
    assert x >= 0
    ...   # compute square root in y

if __name__ == '__main__':
    square_root(4) 

if __name__ == '__main__':
    square_root(4.0)

if __name__ == '__main__':
    with ExpectError():
        square_root('4')

if __name__ == '__main__':
    quiz("If we did not check for the type of `x`, "
         "would the assertion `x >= 0` still catch a bad call?",
         [
             "Yes, since `>=` is only defined between numbers",
             "No, because an empty list or string would evaluate to 0"
         ], '0b10 - 0b01')

if __name__ == '__main__':
    with ExpectError():
        '4' >= 0  # type: ignore

## Checking Results
## ----------------

if __name__ == '__main__':
    print('\n## Checking Results')



def square_root(x):  # type: ignore
    assert x >= 0
    ...   # compute square root in y
    assert y * y == x

if __name__ == '__main__':
    quiz("Why could the assertion fail despite `square_root()` being correct?",
         [
             "We need to compute `y ** 2`, not `y * y`",
             "We may encounter rounding errors",
             "The value of `x` may have changed during computation",
             "The interpreter / compiler may be buggy"
         ], '0b110011 - 0o61')

import math

if __name__ == '__main__':
    math.sqrt(2.0) * math.sqrt(2.0)

if __name__ == '__main__':
    math.sqrt(2.0) * math.sqrt(2.0) == 2.0

def square_root(x):  # type: ignore
    assert x >= 0
    ...   # compute square root in y
    epsilon = 0.000001
    assert abs(y * y - x) < epsilon

if __name__ == '__main__':
    math.isclose(math.sqrt(2.0) * math.sqrt(2.0), 2.0)

def square_root(x):  # type: ignore
    assert x >= 0
    ...   # compute square root in y
    assert math.isclose(y * y, x)

def square_root(x):  # type: ignore
    assert x >= 0  # precondition

    approx = None
    guess = x / 2
    while approx != guess:
        approx = guess
        guess = (approx + x / approx) / 2

    assert math.isclose(approx * approx, x)
    return approx

if __name__ == '__main__':
    square_root(4.0)

if __name__ == '__main__':
    square_root(12345.0)

### Assertions and Tests

if __name__ == '__main__':
    print('\n### Assertions and Tests')



if __name__ == '__main__':
    for x in range(1, 10000):
        y = square_root(x)

if __name__ == '__main__':
    quiz("Is there a value for x that satisfies the precondition, "
        "but fails the postcondition?",
         [ 
             "Yes",
             "No"
         ], 'int("Y" in "Yes")')

### Partial Checks

if __name__ == '__main__':
    print('\n### Partial Checks')



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
    remove_html_markup("I am a text with <strong>HTML markup</strong>")

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

    # postcondition
    assert '<' not in out and '>' not in out

    return out

if __name__ == '__main__':
    quiz("Which of these inputs causes the assertion to fail?",
        [
            '`<foo>bar</foo>`',
            '`"foo"`',
            '`>foo<`',
            '`"x > y"`'
        ], '1 + 1 -(-1) + (1 * -1) + 1 ** (1 - 1) + 1')

if __name__ == '__main__':
    remove_html_markup('"foo"')

if __name__ == '__main__':
    with ExpectError():
        remove_html_markup('"x > y"')

### Assertions and Documentation

if __name__ == '__main__':
    print('\n### Assertions and Documentation')



def some_obscure_function(x: int, y: int, z: int) -> int:
    result = int(...)  # type: ignore
    assert x == y == z or result > min(x, y, z)
    assert x == y == z or result < max(x, y, z)
    return result

if __name__ == '__main__':
    quiz("What does this function do?",
        [
            "It returns the minimum value out of `x`, `y`, `z`",
            "It returns the middle value out of `x`, `y`, `z`",
            "It returns the maximum value out of `x`, `y`, `z`",
        ], 'int(0.5 ** math.cos(math.pi))', globals())

### Using Assertions to Trivially Locate Defects

if __name__ == '__main__':
    print('\n### Using Assertions to Trivially Locate Defects')



if __name__ == '__main__':
    quiz("Which function is faulty here?",
         [
            "`g()` because it raises an exception",
            "`f()`"
            " because it violates the precondition of `g()`",
            "Both `f()` and `g()`"
            " because they are incompatible",
            "None of the above"
         ], 'math.factorial(int(math.tau / math.pi))', globals())

## Checking Data Structures
## ------------------------

if __name__ == '__main__':
    print('\n## Checking Data Structures')



### Times and Time Bombs

if __name__ == '__main__':
    print('\n### Times and Time Bombs')



class Time:
    def __init__(self, hours: int = 0, minutes: int = 0, seconds: int = 0) -> None:
        self._hours = hours
        self._minutes = minutes
        self._seconds = seconds

class Time(Time):
    def hours(self) -> int:
        return self._hours

    def minutes(self) -> int:
        return self._minutes

    def seconds(self) -> int:
        return self._seconds

class Time(Time):
    def __repr__(self) -> str:
        return f"{self.hours():02}:{self.minutes():02}:{self.seconds():02}"

if __name__ == '__main__':
    t = Time(23, 57, 0)
    t

if __name__ == '__main__':
    t = Time(-1, 0, 0)
    t

if __name__ == '__main__':
    t = Time("High noon")  # type: ignore

if __name__ == '__main__':
    with ExpectError():  # This fails in Python 3.9
        print(t)

class Time(Time):
    def __init__(self, hours: int = 0, minutes: int = 0, seconds: int = 0) -> None:
        assert 0 <= hours <= 23
        assert 0 <= minutes <= 59
        assert 0 <= seconds <= 60  # Includes leap seconds (ISO8601)

        self._hours = hours
        self._minutes = minutes
        self._seconds = seconds

if __name__ == '__main__':
    with ExpectError():
        t = Time(-23, 0, 0)

### Invariant Checkers

if __name__ == '__main__':
    print('\n### Invariant Checkers')



class Time(Time):
    def set_hours(self, hours: int) -> None:
        assert 0 <= hours <= 23
        self._hours = hours

#### Excursion: Checked Getters and Setters in Python

if __name__ == '__main__':
    print('\n#### Excursion: Checked Getters and Setters in Python')



class MyTime(Time):
    @property  # type: ignore
    def hours(self) -> int:
        return self._hours

    @hours.setter
    def hours(self, new_hours: int) -> None:
        assert 0 <= new_hours <= 23
        self._hours = new_hours

if __name__ == '__main__':
    my_time = MyTime(11, 30, 0)
    my_time.hours

if __name__ == '__main__':
    my_time.hours = 12  # type: ignore

if __name__ == '__main__':
    with ExpectError():
        my_time.hours = 25  # type: ignore

#### End of Excursion

if __name__ == '__main__':
    print('\n#### End of Excursion')



class Time(Time):
    def repOK(self) -> bool:
        assert 0 <= self.hours() <= 23
        assert 0 <= self.minutes() <= 59
        assert 0 <= self.seconds() <= 60
        return True

class Time(Time):
    def __init__(self, hours: int = 0, minutes: int = 0, seconds: int = 0) -> None:
        self._hours = hours
        self._minutes = minutes
        self._seconds = seconds
        assert self.repOK()

    def set_hours(self, hours: int) -> None:
        self._hours = hours
        assert self.repOK()

if __name__ == '__main__':
    with ExpectError():
        t = Time(-23, 0, 0)

if __name__ == '__main__':
    Time(1.5)  # type: ignore

class Time(Time):
    def repOK(self) -> bool:
        assert isinstance(self.hours(), int)
        assert isinstance(self.minutes(), int)
        assert isinstance(self.seconds(), int)

        assert 0 <= self.hours() <= 23
        assert 0 <= self.minutes() <= 59
        assert 0 <= self.seconds() <= 60
        return True

if __name__ == '__main__':
    Time(14, 0, 0)

if __name__ == '__main__':
    with ExpectError():
        t = Time("After midnight")  # type: ignore

class Time(Time):
    def seconds_since_midnight(self) -> int:
        return self.hours() * 3600 + self.minutes() * 60 + self.seconds()

    def advance(self, seconds_offset: int) -> None:
        old_seconds = self.seconds_since_midnight()

        ...  # Advance the clock

        assert (self.seconds_since_midnight() ==
                (old_seconds + seconds_offset) % (24 * 60 * 60))

class BetterTime(Time):
    def advance(self, seconds_offset: int) -> None:
        assert self.repOK()
        old_seconds = self.seconds_since_midnight()

        ...  # Advance the clock

        assert (self.seconds_since_midnight() ==
                (old_seconds + seconds_offset) % (24 * 60 * 60))
        assert self.repOK()

### Large Data Structures

if __name__ == '__main__':
    print('\n### Large Data Structures')



class RedBlackTree:
    RED = 'red'
    BLACK = 'black'
    ...

class RedBlackNode:
    def __init__(self) -> None:
        self.parent = None
        self.color = RedBlackTree.BLACK
    pass

class RedBlackTree(RedBlackTree):
    def redNodesHaveOnlyBlackChildren(self) -> bool:
        return True

    def equalNumberOfBlackNodesOnSubtrees(self) -> bool:
        return True

    def treeIsAcyclic(self) -> bool:
        return True

    def parentsAreConsistent(self) -> bool:
        return True

    def __init__(self) -> None:
        self._root = RedBlackNode()
        self._root.parent = None
        self._root.color = self.BLACK

class RedBlackTree(RedBlackTree):
    def repOK(self) -> bool:
        assert self.rootHasNoParent()
        assert self.rootIsBlack()
        assert self.redNodesHaveOnlyBlackChildren()
        assert self.equalNumberOfBlackNodesOnSubtrees()
        assert self.treeIsAcyclic()
        assert self.parentsAreConsistent()
        return True

class RedBlackTree(RedBlackTree):
    def rootHasNoParent(self) -> bool:
        return self._root.parent is None

    def rootIsBlack(self) -> bool:
        return self._root.color == self.BLACK
    ...

from typing import Any, List

class RedBlackTree(RedBlackTree):
    def insert(self, item: Any) -> None:
        assert self.repOK()
        ...  # four pages of code
        assert self.repOK()

    def delete(self, item: Any) -> None:
        assert self.repOK()
        ...  # five pages of code
        assert self.repOK()

class RedBlackTree(RedBlackTree):
    def __init__(self, checkRepOK: bool = False) -> None:
        ...
        self.checkRepOK = checkRepOK

    def repOK(self) -> bool:
        if not self.checkRepOK:
            return True

        assert self.rootHasNoParent()
        assert self.rootIsBlack()
        ...
        return True

## System Invariants
## -----------------

if __name__ == '__main__':
    print('\n## System Invariants')



if __name__ == '__main__':
    with ExpectError():
        index = 10
        "foo"[index]

### The C Memory Model

if __name__ == '__main__':
    print('\n### The C Memory Model')



if __name__ == '__main__':
    open('testoverflow.c', 'w').write(r'''
#include <stdio.h>

// Access memory out of bounds
int main(int argc, char *argv[]) {
    int index = 10;
    return "foo"[index];  // BOOM
}
''');

if __name__ == '__main__':
    print_content(open('testoverflow.c').read())

if __name__ == '__main__':
    import os
    os.system(f'cc -g -o testoverflow testoverflow.c')

if __name__ == '__main__':
    import os
    os.system(f'./testoverflow')

#### Excursion: A C Memory Model Simulator

if __name__ == '__main__':
    print('\n#### Excursion: A C Memory Model Simulator')



class Memory:
    def __init__(self, size: int = 10) -> None:
        self.size: int = size
        self.memory: List[Any] = [None for i in range(size)]

    def read(self, address: int) -> Any:
        return self.memory[address]

    def write(self, address: int, item: Any) -> None:
        self.memory[address] = item

    def __repr__(self) -> str:
        return repr(self.memory)

if __name__ == '__main__':
    mem: Memory = Memory()

if __name__ == '__main__':
    mem

if __name__ == '__main__':
    mem.write(0, 'a')

if __name__ == '__main__':
    mem

if __name__ == '__main__':
    mem.read(0)

class Memory(Memory):
    def __getitem__(self, address: int) -> Any:
        return self.read(address)

    def __setitem__(self, address: int, item: Any) -> None:
        self.write(address, item)

if __name__ == '__main__':
    mem_with_index: Memory = Memory()
    mem_with_index[1] = 'a'

if __name__ == '__main__':
    mem_with_index

if __name__ == '__main__':
    mem_with_index[1]

if __name__ == '__main__':
    from IPython.display import display, Markdown, HTML

class Memory(Memory):
    def show_header(self) -> str:
        out = "|Address|"
        for address in range(self.size):
            out += f"{address}|"
        return out + '\n'

    def show_sep(self) -> str:
        out = "|:---|"
        for address in range(self.size):
            out += ":---|"
        return out + '\n'

    def show_contents(self) -> str:
        out = "|Content|"
        for address in range(self.size):
            contents = self.memory[address]
            if contents is not None:
                out += f"{repr(contents)}|"
            else:
                out += " |"
        return out + '\n'

    def __repr__(self) -> str:
        return self.show_header() + self.show_sep() + self.show_contents()

    def _repr_markdown_(self) -> str:
        return repr(self)

if __name__ == '__main__':
    mem_with_table: Memory = Memory()
    for i in range(mem_with_table.size):
        mem_with_table[i] = 10 * i
    mem_with_table

#### End of Excursion

if __name__ == '__main__':
    print('\n#### End of Excursion')



if __name__ == '__main__':
    mem_with_table: Memory = Memory(20)
    mem_with_table[5] = 'f'
    mem_with_table[6] = 'o'
    mem_with_table[7] = 'o'
    mem_with_table

### Dynamic Memory

if __name__ == '__main__':
    print('\n### Dynamic Memory')



if __name__ == '__main__':
    open('testuseafterfree.c', 'w').write(r'''
#include <stdlib.h>

// Access a chunk of memory after it has been given back to the system
int main(int argc, char *argv[]) {
    int *array = malloc(100 * sizeof(int));
    free(array);
    return array[10];  // BOOM
}
''');

if __name__ == '__main__':
    print_content(open('testuseafterfree.c').read())

if __name__ == '__main__':
    import os
    os.system(f'cc -g -o testuseafterfree testuseafterfree.c')

if __name__ == '__main__':
    import os
    os.system(f'./testuseafterfree')

#### Excursion: Dynamic Memory in C

if __name__ == '__main__':
    print('\n#### Excursion: Dynamic Memory in C')



class DynamicMemory(Memory):
    # Address at which our list of blocks starts
    BLOCK_LIST_START = 0

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)

        # Before each block, we reserve two items:
        # One pointing to the next block (-1 = END)
        self.memory[self.BLOCK_LIST_START] = -1
        # One giving the length of the current block (<0: freed)
        self.memory[self.BLOCK_LIST_START + 1] = 0

    def allocate(self, block_size: int) -> int:
        """Allocate a block of memory"""
        # traverse block list 
        # until we find a free block of appropriate size
        chunk = self.BLOCK_LIST_START

        while chunk < self.size:
            next_chunk = self.memory[chunk]
            chunk_length = self.memory[chunk + 1]

            if chunk_length < 0 and abs(chunk_length) >= block_size:
                # Reuse this free block
                self.memory[chunk + 1] = abs(chunk_length)
                return chunk + 2

            if next_chunk < 0:
                # End of list - allocate new block
                next_chunk = chunk + block_size + 2
                if next_chunk >= self.size:
                    break

                self.memory[chunk] = next_chunk
                self.memory[chunk + 1] = block_size
                self.memory[next_chunk] = -1
                self.memory[next_chunk + 1] = 0
                base = chunk + 2
                return base

            # Go to next block
            chunk = next_chunk

        raise MemoryError("Out of Memory")

    def free(self, base: int) -> None:
        """Free a block of memory"""
        # Mark block as available
        chunk = base - 2
        self.memory[chunk + 1] = -abs(self.memory[chunk + 1])

class DynamicMemory(DynamicMemory):
    def show_header(self) -> str:
        out = "|Address|"
        color = "black"
        chunk = self.BLOCK_LIST_START
        allocated = False

        # States and colors
        for address in range(self.size):
            if address == chunk:
                color = "blue"
                next_chunk = self.memory[address]
            elif address == chunk + 1:
                color = "blue"
                allocated = self.memory[address] > 0
                chunk = next_chunk
            elif allocated:
                color = "black"
            else:
                color = "lightgrey"

            item = f'<span style="color: {color}">{address}</span>'
            out += f"{item}|"
        return out + '\n'

if __name__ == '__main__':
    dynamic_mem: DynamicMemory = DynamicMemory(10)

if __name__ == '__main__':
    dynamic_mem

if __name__ == '__main__':
    dynamic_mem.allocate(2)

if __name__ == '__main__':
    dynamic_mem

if __name__ == '__main__':
    dynamic_mem.allocate(2)

if __name__ == '__main__':
    dynamic_mem

if __name__ == '__main__':
    dynamic_mem.free(2)

if __name__ == '__main__':
    dynamic_mem

if __name__ == '__main__':
    dynamic_mem.allocate(1)

if __name__ == '__main__':
    dynamic_mem

if __name__ == '__main__':
    with ExpectError():
        dynamic_mem.allocate(1)

#### End of Excursion

if __name__ == '__main__':
    print('\n#### End of Excursion')



if __name__ == '__main__':
    dynamic_mem: DynamicMemory = DynamicMemory(13)
    dynamic_mem

if __name__ == '__main__':
    p1 = dynamic_mem.allocate(3)
    p1

if __name__ == '__main__':
    dynamic_mem

if __name__ == '__main__':
    p2 = dynamic_mem.allocate(4)
    p2

if __name__ == '__main__':
    dynamic_mem[p1] = 123
    dynamic_mem[p2] = 'x'

if __name__ == '__main__':
    dynamic_mem

if __name__ == '__main__':
    dynamic_mem.free(p1)

if __name__ == '__main__':
    dynamic_mem

if __name__ == '__main__':
    dynamic_mem[p1]

if __name__ == '__main__':
    p3 = dynamic_mem.allocate(2)
    dynamic_mem[p3] = 'y'
    dynamic_mem

if __name__ == '__main__':
    dynamic_mem[p1]

from .ExpectError import ExpectTimeout

if __name__ == '__main__':
    dynamic_mem[p3 + 3] = 0

if __name__ == '__main__':
    dynamic_mem

if __name__ == '__main__':
    with ExpectTimeout(1):
        dynamic_mem.allocate(1)

### Managed Memory

if __name__ == '__main__':
    print('\n### Managed Memory')



#### Excursion: Managed Memory

if __name__ == '__main__':
    print('\n#### Excursion: Managed Memory')



class ManagedMemory(DynamicMemory):
    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.initialized = [False for i in range(self.size)]
        self.allocated = [False for i in range(self.size)]

class ManagedMemory(ManagedMemory):
    def write(self, address: int, item: Any) -> None:
        assert self.allocated[address], \
            "Writing into unallocated memory"
        self.memory[address] = item
        self.initialized[address] = True

    def read(self, address: int) -> Any:
        assert self.allocated[address], \
            "Reading from unallocated memory"
        assert self.initialized[address], \
            "Reading from uninitialized memory"
        return self.memory[address]

class ManagedMemory(ManagedMemory):
    def allocate(self, block_size: int) -> int:
        base = super().allocate(block_size)
        for i in range(block_size):
            self.allocated[base + i] = True
            self.initialized[base + i] = False
        return base

    def free(self, base: int) -> None:
        assert self.allocated[base], \
            "Freeing memory that is already freed"
        block_size = self.memory[base - 1]
        for i in range(block_size):
            self.allocated[base + i] = False
            self.initialized[base + i] = False
        super().free(base)

class ManagedMemory(ManagedMemory):
    def show_contents(self) -> str:
        return (self.show_allocated() + 
               self.show_initialized() +
            DynamicMemory.show_contents(self))

    def show_allocated(self) -> str:
        out = "|Allocated|"
        for address in range(self.size):
            if self.allocated[address]:
                out += "Y|"
            else:
                out += " |"
        return out + '\n'

    def show_initialized(self) -> str:
        out = "|Initialized|"
        for address in range(self.size):
            if self.initialized[address]:
                out += "Y|"
            else:
                out += " |"
        return out + '\n'

#### End of Excursion

if __name__ == '__main__':
    print('\n#### End of Excursion')



if __name__ == '__main__':
    managed_mem: ManagedMemory = ManagedMemory()
    managed_mem

if __name__ == '__main__':
    p = managed_mem.allocate(3)
    managed_mem

if __name__ == '__main__':
    managed_mem[p] = 10
    managed_mem[p + 1] = 20

if __name__ == '__main__':
    managed_mem

if __name__ == '__main__':
    with ExpectError():
        x = managed_mem[p + 2]

if __name__ == '__main__':
    managed_mem.free(p)
    managed_mem

if __name__ == '__main__':
    with ExpectError():
        managed_mem[p] = 10

if __name__ == '__main__':
    with ExpectError():
        managed_mem.free(p)

### Checking Memory Usage with Valgrind

if __name__ == '__main__':
    print('\n### Checking Memory Usage with Valgrind')



if __name__ == '__main__':
    print_content(open('testuseafterfree.c').read())

if __name__ == '__main__':
    import os
    os.system(f'valgrind ./testuseafterfree')

if __name__ == '__main__':
    print_content(open('testoverflow.c').read())

if __name__ == '__main__':
    import os
    os.system(f'valgrind ./testoverflow')

### Checking Memory Usage with Memory Sanitizer

if __name__ == '__main__':
    print('\n### Checking Memory Usage with Memory Sanitizer')



if __name__ == '__main__':
    import os
    os.system(f'cc -fsanitize=address -o testuseafterfree testuseafterfree.c')

if __name__ == '__main__':
    import os
    os.system(f'./testuseafterfree')

if __name__ == '__main__':
    import os
    os.system(f'cc -fsanitize=address -o testoverflow testoverflow.c')

if __name__ == '__main__':
    import os
    os.system(f'./testoverflow')

## When Should Invariants be Checked?
## ----------------------------------

if __name__ == '__main__':
    print('\n## When Should Invariants be Checked?')



### Assertions are not Production Code

if __name__ == '__main__':
    print('\n### Assertions are not Production Code')



### For System Preconditions, Use Production Code 

if __name__ == '__main__':
    print('\n### For System Preconditions, Use Production Code ')



### Consider Leaving Some Assertions On

if __name__ == '__main__':
    print('\n### Consider Leaving Some Assertions On')



### Define How Your Application Should Handle Internal Errors

if __name__ == '__main__':
    print('\n### Define How Your Application Should Handle Internal Errors')



## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



def my_square_root(x):  # type: ignore
    assert x >= 0
    y = square_root(x)
    assert math.isclose(y * y, x)
    return y

if __name__ == '__main__':
    with ExpectError():
        y = my_square_root(-1)

if __name__ == '__main__':
    managed_mem = ManagedMemory()
    managed_mem

if __name__ == '__main__':
    with ExpectError():
        x = managed_mem[2]

## Lessons Learned
## ---------------

if __name__ == '__main__':
    print('\n## Lessons Learned')



import os
import shutil

if __name__ == '__main__':
    for path in [
                    'assert.h',
                    'testassert',
                    'testassert.c',
                    'testassert.dSYM',
                    'testoverflow',
                    'testoverflow.c',
                    'testoverflow.dSYM',
                    'testuseafterfree',
                    'testuseafterfree.c',
                    'testuseafterfree.dSYM',
                ]:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

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



### Exercise 1 – Storage Assertions

if __name__ == '__main__':
    print('\n### Exercise 1 – Storage Assertions')



import shelve

if __name__ == '__main__':
    d = shelve.open('mydb')

if __name__ == '__main__':
    d['123'] = 123

if __name__ == '__main__':
    d['123']

if __name__ == '__main__':
    d.close()

if __name__ == '__main__':
    d = shelve.open('mydb')

if __name__ == '__main__':
    d['123']

if __name__ == '__main__':
    d.close()

from typing import Sequence, Any, Callable, Optional, Type, Tuple, Any
from typing import Dict, Union, Set, List, FrozenSet, cast

from types import TracebackType

class Storage:
    def __init__(self, dbname: str) -> None:
        self.dbname = dbname

    def __enter__(self) -> Any:
        self.db = shelve.open(self.dbname)
        return self

    def __exit__(self, exc_tp: Type, exc_value: BaseException, 
                 exc_traceback: TracebackType) -> Optional[bool]:
        self.db.close()
        return None

    def __getitem__(self, key: str) -> Any:
        return self.db[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.db[key] = value

if __name__ == '__main__':
    with Storage('mydb') as storage:
        print(storage['123'])

#### Task 1 – Local Consistency

if __name__ == '__main__':
    print('\n#### Task 1 – Local Consistency')



class Storage(Storage):
    def __setitem__(self, key: str, value: Any) -> None:
        self.db[key] = value
        assert self.db[key] == value

#### Task 2 – Global Consistency

if __name__ == '__main__':
    print('\n#### Task 2 – Global Consistency')



class ShadowStorage:
    def __init__(self, dbname: str) -> None:
        self.dbname = dbname

    def __enter__(self) -> Any:
        self.db = shelve.open(self.dbname)
        self.memdb = {}
        for key in self.db.keys():
            self.memdb[key] = self.db[key]
        assert self.repOK()
        return self

    def __exit__(self, exc_tp: Type, exc_value: BaseException, 
                 exc_traceback: TracebackType) -> Optional[bool]:
        self.db.close()
        return None

    def __getitem__(self, key: str) -> Any:
        assert self.repOK()
        return self.db[key]

    def __setitem__(self, key: str, value: Any) -> None:
        assert self.repOK()
        self.memdb[key] = self.db[key] = value
        assert self.repOK()

    def repOK(self) -> bool:
        assert self.db.keys() == self.memdb.keys(), f"{self.dbname}: Differing keys"
        for key in self.memdb.keys():
            assert self.db[key] == self.memdb[key], \
                f"{self.dbname}: Differing values for {repr(key)}"
        return True

if __name__ == '__main__':
    with ShadowStorage('mydb') as storage:
        storage['456'] = 456
        print(storage['123'])

if __name__ == '__main__':
    try:
        os.remove('mydb.db')  # on macOS
    except FileNotFoundError:
        pass

if __name__ == '__main__':
    try:
        os.remove('mydb')  # on Linux
    except FileNotFoundError:
        pass
