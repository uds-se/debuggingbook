#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Debugging Book".
# Web site: https://www.debuggingbook.org/html/Assertions.html
# Last change: 2021-01-17 18:22:51+01:00
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


# # Asserting Expectations

if __name__ == "__main__":
    print('# Asserting Expectations')




if __name__ == "__main__":
    from bookutils import YouTubeVideo
    YouTubeVideo("9mI9sbKFkwU")


if __name__ == "__main__":
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)


if __package__ is None or __package__ == "":
    from bookutils import quiz
else:
    from .bookutils import quiz


if __package__ is None or __package__ == "":
    import Tracer
else:
    from . import Tracer


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




# ## Introducing Assertions

if __name__ == "__main__":
    print('\n## Introducing Assertions')




# ### Assertions

if __name__ == "__main__":
    print('\n### Assertions')




if __name__ == "__main__":
    assert True


if __package__ is None or __package__ == "":
    from ExpectError import ExpectError
else:
    from .ExpectError import ExpectError


if __name__ == "__main__":
    with ExpectError():
        assert False


def test_square_root():
    assert square_root(4) == 2
    assert square_root(9) == 3
    ...

def my_own_assert(cond):
    if not cond:
        raise AssertionError

if __name__ == "__main__":
    with ExpectError():
        my_own_assert(2 + 2 == 5)


# ### Assertion Diagnostics

if __name__ == "__main__":
    print('\n### Assertion Diagnostics')




if __name__ == "__main__":
    open('testassert.c', 'w').write(r'''
#include <stdio.h>
#include "assert.h"

int main(int argc, char *argv[]) {
    assert(2 + 2 == 5);
    printf("Foo\n");
}
''');


if __package__ is None or __package__ == "":
    from bookutils import print_content
else:
    from .bookutils import print_content


if __name__ == "__main__":
    print_content(open('testassert.c').read())


if __name__ == "__main__":
    import os
    os.system(f'cc -g -o testassert testassert.c')


if __name__ == "__main__":
    import os
    os.system(f'./testassert')


if __name__ == "__main__":
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


if __name__ == "__main__":
    print_content(open('assert.h').read())


if __name__ == "__main__":
    import os
    os.system(f'cc -DNDEBUG -g -o testassert testassert.c')


if __name__ == "__main__":
    import os
    os.system(f'./testassert')


if __name__ == "__main__":
    import os
    os.system(f'python -c \'assert 2 + 2 == 5; print("Foo")\'')


if __name__ == "__main__":
    import os
    os.system(f'python -O -c \'assert 2 + 2 == 5; print("Foo")\'')


def fun():
    assert 2 + 2 == 5

if __name__ == "__main__":
    quiz("If we invoke `fun()` and the assertion fails,"
         " which information do we get?",
         [
             "The failing condition (`2 + 2 == 5`)",
             "The location of the assertion in the program",
             "The list of callers",
             "All of the above"
         ],
         123456789 % 5
        )


if __name__ == "__main__":
    with ExpectError():
        fun()


# ## Checking Preconditions

if __name__ == "__main__":
    print('\n## Checking Preconditions')




def square_root(x):
    assert x >= 0
    ...   # compute square root in y

if __name__ == "__main__":
    with ExpectError():
        square_root(-1)


def square_root(x):
    assert isinstance(x, (int, float))
    assert x >= 0
    ...   # compute square root in y

if __name__ == "__main__":
    square_root(4) 


if __name__ == "__main__":
    square_root(4.0)


if __name__ == "__main__":
    with ExpectError():
        square_root('4')


if __name__ == "__main__":
    quiz("If we did not check for the type of `x`, "
         "would the assertion `x >= 0` still catch a bad call?",
         [
             "Yes, since `>=` is only defined between numbers",
             "No, because an empty list or string would evaluate to 0"
         ],
         0b10 - 0b01
        )


if __name__ == "__main__":
    with ExpectError():
        '4' >= 0


# ## Checking Results

if __name__ == "__main__":
    print('\n## Checking Results')




def square_root(x):
    assert x >= 0
    ...   # compute square root in y
    assert y * y == x

if __name__ == "__main__":
    quiz("Why could the assertion fail despite `square_root()` being correct?",
         [
             "We need to compute `y ** 2`, not `y * y`",
             "We may encounter rounding errors",
             "The value of `x` may have changed during computation",
             "The interpreter / compiler may be buggy"
         ],
        0b110011 - 0o61
        )


import math

if __name__ == "__main__":
    math.sqrt(2.0) * math.sqrt(2.0)


if __name__ == "__main__":
    math.sqrt(2.0) * math.sqrt(2.0) == 2.0


def square_root(x):
    assert x >= 0
    ...   # compute square root in y
    epsilon = 0.000001
    assert abs(y * y - x) < epsilon

if __name__ == "__main__":
    math.isclose(math.sqrt(2.0) * math.sqrt(2.0), 2.0)


def square_root(x):
    assert x >= 0
    ...   # compute square root in y
    assert math.isclose(y * y, x)

def square_root(x):
    assert x >= 0  # precondition

    approx = None
    guess = x / 2
    while approx != guess:
        approx = guess
        guess = (approx + x / approx) / 2

    assert math.isclose(approx * approx, x)
    return approx

if __name__ == "__main__":
    square_root(4.0)


if __name__ == "__main__":
    square_root(12345.0)


# ### Assertions and Tests

if __name__ == "__main__":
    print('\n### Assertions and Tests')




if __name__ == "__main__":
    for x in range(1, 10000):
        y = square_root(x)


if __name__ == "__main__":
    quiz("Is there a value for x that satisfies the precondition, "
        "but fails the postcondition?",
         [ "Yes", "No" ],
         int("Y" in "Yes"))


# ### Partial Checks

if __name__ == "__main__":
    print('\n### Partial Checks')




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
    remove_html_markup("I am a text with <strong>HTML markup</strong>")


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

    # postcondition
    assert '<' not in out and '>' not in out

    return out

if __name__ == "__main__":
    quiz("Which of these inputs causes the assertion to fail?",
        [
            '`<foo>bar</foo>`',
            '`"foo"`',
            '`>foo<`',
            '`"x > y"`'
        ], 1 + 1 -(-1) + (1 * -1) + 1 ** (1 - 1) + 1)


if __name__ == "__main__":
    remove_html_markup('"foo"')


if __name__ == "__main__":
    with ExpectError():
        remove_html_markup('"x > y"')


# ### Assertions and Documentation

if __name__ == "__main__":
    print('\n### Assertions and Documentation')




def some_obscure_function(x, y, z):
    result = ...
    assert x == y == z or result > min(x, y, z)
    assert x == y == z or result < max(x, y, z)
    return result

if __name__ == "__main__":
    quiz("What does this function do?",
        [
            "It returns the minimum value out of `x`, `y`, `z`",
            "It returns the middle value out of `x`, `y`, `z`",
            "It returns the maximum value out of `x`, `y`, `z`",
        ],
          int(0.5 ** math.cos(math.pi))
        )


# ### Using Assertions to Trivially Locate Defects

if __name__ == "__main__":
    print('\n### Using Assertions to Trivially Locate Defects')




if __name__ == "__main__":
    quiz("Which function is faulty here?",
         [
            "`g()` because it raises an exception",
            "`f()`"
            " because it violates the precondition of `g()`",
            "Both `f()` and `g()`"
            " because they are incompatible",
            "None of the above"
         ],
         math.factorial(int(math.tau / math.pi))
        )


# ## Checking Data Structures

if __name__ == "__main__":
    print('\n## Checking Data Structures')




# ### Times and Time Bombs

if __name__ == "__main__":
    print('\n### Times and Time Bombs')




class Time(object):
    def __init__(self, hours=0, minutes=0, seconds=0):
        self._hours = hours
        self._minutes = minutes
        self._seconds = seconds

class Time(Time):
    def hours(self):
        return self._hours

    def minutes(self):
        return self._minutes

    def seconds(self):
        return self._seconds

class Time(Time):
    def __repr__(self):
        return f"{self.hours():02}:{self.minutes():02}:{self.seconds():02}"

if __name__ == "__main__":
    t = Time(23, 57, 0)
    t


if __name__ == "__main__":
    t = Time(-1, 0, 0)
    t


if __name__ == "__main__":
    t = Time("High noon")


if __name__ == "__main__":
    with ExpectError():
        print(t)


class Time(Time):
    def __init__(self, hours=0, minutes=0, seconds=0):
        assert 0 <= hours <= 23
        assert 0 <= minutes <= 59
        assert 0 <= seconds <= 60  # Includes leap seconds (ISO8601)

        self._hours = hours
        self._minutes = minutes
        self._seconds = seconds

if __name__ == "__main__":
    with ExpectError():
        t = Time(-23, 0, 0)


# ### Invariant Checkers

if __name__ == "__main__":
    print('\n### Invariant Checkers')




class Time(Time):
    def set_hours(self, hours):
        assert 0 <= hours <= 23
        self._hours = hours

# #### Excursion: Checked Getters and Setters in Python

if __name__ == "__main__":
    print('\n#### Excursion: Checked Getters and Setters in Python')




class MyTime(Time):
    @property
    def hours(self):
        return self._hours

    @hours.setter
    def hours(self, new_hours):
        assert 0 <= new_hours <= 23
        self._hours = new_hours

if __name__ == "__main__":
    t = MyTime(11, 30, 0)
    t.hours


if __name__ == "__main__":
    t.hours = 12


if __name__ == "__main__":
    with ExpectError():
        t.hours = 25


# #### End of Excursion

if __name__ == "__main__":
    print('\n#### End of Excursion')




class Time(Time):
    def repOK(self):
        assert 0 <= self.hours() <= 23
        assert 0 <= self.minutes() <= 59
        assert 0 <= self.seconds() <= 60
        return True

class Time(Time):
    def __init__(self, hours=0, minutes=0, seconds=0):
        self._hours = hours
        self._minutes = minutes
        self._seconds = seconds
        assert self.repOK()

    def set_hours(self, hours):
        self._hours = hours
        assert self.repOK()

if __name__ == "__main__":
    with ExpectError():
        t = Time(-23, 0, 0)


if __name__ == "__main__":
    Time(1.5)


class Time(Time):
    def repOK(self):
        assert isinstance(self.hours(), int)
        assert isinstance(self.minutes(), int)
        assert isinstance(self.seconds(), int)

        assert 0 <= self.hours() <= 23
        assert 0 <= self.minutes() <= 59
        assert 0 <= self.seconds() <= 60
        return True

if __name__ == "__main__":
    Time(14, 0, 0)


if __name__ == "__main__":
    with ExpectError():
        t = Time("After midnight")


class Time(Time):
    def advance(self, seconds_offset):
        # Some complex computation
        ...

class Time(Time):
    def seconds_since_midnight(self):
        return self.hours() * 3600 + self.minutes() * 60 + self.seconds()

    def advance(self, seconds_offset):
        old_seconds = self.seconds_since_midnight()
        ...  # Advance the clock
        assert (self.seconds_since_midnight() ==
                (old_seconds + seconds_offset) % (24 * 60 * 60))

class Time(Time):
    def advance(self, seconds_offset):
        assert self.repOK()
        old_seconds = self.seconds_since_midnight()

        ...  # Advance the clock

        assert (self.seconds_since_midnight() ==
                (old_seconds + seconds_offset) % (24 * 60 * 60))
        assert self.repOK()

# ### Large Data Structures

if __name__ == "__main__":
    print('\n### Large Data Structures')




class RedBlackTree:
    def repOK():
        assert self.rootHasNoParent()
        assert self.rootIsBlack()
        assert self.redNodesHaveOnlyBlackChildren()
        assert self.equalNumberOfBlackNodesOnSubtrees()
        assert self.treeIsAcyclic()
        assert self.parentsAreConsistent()
        return True

class RedBlackTree(RedBlackTree):
    def rootHasNoParent(self):
        return self._root.parent is None
    def rootIsBlack(self):
        return self._root.color == self.BLACK
    ...

class RedBlackTree(RedBlackTree):
    def insert(self, item):
        assert self.repOK()
        ...  # four pages of code
        assert self.repOK()

    def delete(self, item):
        assert self.repOK()
        ...  # five pages of code
        assert self.repOK()

class RedBlackTree(RedBlackTree):
    def __init__(self, checkRepOK=False):
        ...
        self.checkRepOK = checkRepOK

    def repOK(self):
        if not self.checkRepOK:
            return True

        assert self.rootHasNoParent()
        assert self.rootIsBlack()
        ...

# ## System Invariants

if __name__ == "__main__":
    print('\n## System Invariants')




if __name__ == "__main__":
    with ExpectError():
        index = 10
        "foo"[index]


# ### The C Memory Model

if __name__ == "__main__":
    print('\n### The C Memory Model')




if __name__ == "__main__":
    open('testoverflow.c', 'w').write(r'''
#include <stdio.h>

// Access memory out of bounds
int main(int argc, char *argv[]) {
    int index = 10;
    return "foo"[index];  // BOOM
}
''');


if __name__ == "__main__":
    print_content(open('testoverflow.c').read())


if __name__ == "__main__":
    import os
    os.system(f'cc -g -o testoverflow testoverflow.c')


if __name__ == "__main__":
    import os
    os.system(f'./testoverflow')


# #### Excursion: A C Memory Model Simulator

if __name__ == "__main__":
    print('\n#### Excursion: A C Memory Model Simulator')




class Memory:
    def __init__(self, size=10):
        self.size = size
        self.memory = [None for i in range(size)]

    def read(self, address):
        return self.memory[address]
    
    def write(self, address, item):
        self.memory[address] = item

if __name__ == "__main__":
    m = Memory()
    m.write(0, 'a')
    m.read(0)


class Memory(Memory):
    def __getitem__(self, address):
        return self.read(address)
    def __setitem__(self, address, item):
        self.write(address, item)

if __name__ == "__main__":
    m = Memory()
    m[0] = "a"
    m[0]


if __name__ == "__main__":
    from IPython.display import display, Markdown, HTML


class Memory(Memory):
    def show_header(self):
        out = "|Address|"
        for address in range(self.size):
            out += f"{address}|"
        return out + '\n'

    def show_sep(self):
        out = "|:---|"
        for address in range(self.size):
            out += ":---|"
        return out + '\n'

    def show_contents(self):
        out = "|Content|"
        for address in range(self.size):
            contents = self.memory[address]
            if contents is not None:
                out += f"{repr(contents)}|"
            else:
                out += " |"
        return out + '\n'

    def __repr__(self):
        return self.show_header() + self.show_sep() + self.show_contents()

    def show(self):
        return Markdown(repr(self))

if __name__ == "__main__":
    m = Memory()
    for i in range(m.size):
        m[i] = 10 * i
    m.show()


# #### End of Excursion

if __name__ == "__main__":
    print('\n#### End of Excursion')




if __name__ == "__main__":
    m = Memory(20)
    m[5:7] = 'foo'
    m.show()


# ### Dynamic Memory

if __name__ == "__main__":
    print('\n### Dynamic Memory')




if __name__ == "__main__":
    open('testuseafterfree.c', 'w').write(r'''
#include <stdlib.h>

// Access a chunk of memory after it has been given back to the system
int main(int argc, char *argv[]) {
    int *array = malloc(100 * sizeof(int));
    free(array);
    return array[10];  // BOOM
}
''');


if __name__ == "__main__":
    print_content(open('testuseafterfree.c').read())


if __name__ == "__main__":
    import os
    os.system(f'cc -g -o testuseafterfree testuseafterfree.c')


if __name__ == "__main__":
    import os
    os.system(f'./testuseafterfree')


# #### Excursion: Dynamic Memory in C

if __name__ == "__main__":
    print('\n#### Excursion: Dynamic Memory in C')




class DynamicMemory(Memory):
    # Address at which our list of blocks starts
    BLOCK_LIST_START = 0

    def __init__(self, *args):
        super().__init__(*args)

        # Before each block, we reserve two items:
        # One pointing to the next block (-1 = END)
        self.memory[self.BLOCK_LIST_START] = -1
        # One giving the length of the current block (<0: freed)
        self.memory[self.BLOCK_LIST_START + 1] = 0

    def allocate(self, block_size):
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

    def free(self, base):
        """Free a block of memory"""
        # Mark block as available
        chunk = base - 2
        self.memory[chunk + 1] = -abs(self.memory[chunk + 1])

class DynamicMemory(DynamicMemory):
    def show_header(self):
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

if __name__ == "__main__":
    m = DynamicMemory(10)


if __name__ == "__main__":
    m.show()


if __name__ == "__main__":
    m.allocate(2)


if __name__ == "__main__":
    m.show()


if __name__ == "__main__":
    m.allocate(2)


if __name__ == "__main__":
    m.show()


if __name__ == "__main__":
    m.free(2)


if __name__ == "__main__":
    m.show()


if __name__ == "__main__":
    m.allocate(1)


if __name__ == "__main__":
    m.show()


if __name__ == "__main__":
    with ExpectError():
        m.allocate(1)


# #### End of Excursion

if __name__ == "__main__":
    print('\n#### End of Excursion')




if __name__ == "__main__":
    m = DynamicMemory(13)
    m.show()


if __name__ == "__main__":
    p1 = m.allocate(3)
    p1


if __name__ == "__main__":
    m.show()


if __name__ == "__main__":
    p2 = m.allocate(4)
    p2


if __name__ == "__main__":
    m[p1] = 123
    m[p2] = 'x'


if __name__ == "__main__":
    m.show()


if __name__ == "__main__":
    m.free(p1)


if __name__ == "__main__":
    m.show()


if __name__ == "__main__":
    m[p1]


if __name__ == "__main__":
    p3 = m.allocate(2)
    m[p3] = 'y'
    m.show()


if __name__ == "__main__":
    m[p1]


if __package__ is None or __package__ == "":
    from ExpectError import ExpectTimeout
else:
    from .ExpectError import ExpectTimeout


if __name__ == "__main__":
    m[p3 + 3] = 0


if __name__ == "__main__":
    m.show()


if __name__ == "__main__":
    with ExpectTimeout(1):
        m.allocate(1)


# ### Managed Memory

if __name__ == "__main__":
    print('\n### Managed Memory')




# #### Excursion: Managed Memory

if __name__ == "__main__":
    print('\n#### Excursion: Managed Memory')




class ManagedMemory(DynamicMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.initialized = [False for i in range(self.size)]
        self.allocated = [False for i in range(self.size)]

class ManagedMemory(ManagedMemory):
    def write(self, address, item):
        assert self.allocated[address], \
            "Writing into unallocated memory"
        self.memory[address] = item
        self.initialized[address] = True

    def read(self, address):
        assert self.allocated[address], \
            "Reading from unallocated memory"
        assert self.initialized[address], \
            "Reading from uninitialized memory"
        return self.memory[address]

class ManagedMemory(ManagedMemory):
    def allocate(self, block_size):
        base = super().allocate(block_size)
        for i in range(block_size):
            self.allocated[base + i] = True
            self.initialized[base + i] = False
        return base

    def free(self, base):
        assert self.allocated[base], \
            "Freeing memory that is already freed"
        block_size = self.memory[base - 1]
        for i in range(block_size):
            self.allocated[base + i] = False
            self.initialized[base + i] = False
        super().free(base)

class ManagedMemory(ManagedMemory):
    def show_contents(self):
        return (self.show_allocated() + 
               self.show_initialized() +
            DynamicMemory.show_contents(self))

    def show_allocated(self):
        out = "|Allocated|"
        for address in range(self.size):
            if self.allocated[address]:
                out += "Y|"
            else:
                out += " |"
        return out + '\n'

    def show_initialized(self):
        out = "|Initialized|"
        for address in range(self.size):
            if self.initialized[address]:
                out += "Y|"
            else:
                out += " |"
        return out + '\n'

# #### End of Excursion

if __name__ == "__main__":
    print('\n#### End of Excursion')




if __name__ == "__main__":
    m = ManagedMemory()
    m.show()


if __name__ == "__main__":
    p = m.allocate(3)
    m.show()


if __name__ == "__main__":
    m[p] = 10
    m[p + 1] = 20


if __name__ == "__main__":
    m.show()


if __name__ == "__main__":
    with ExpectError():
        x = m[p + 2]


if __name__ == "__main__":
    m.free(p)
    m.show()


if __name__ == "__main__":
    with ExpectError():
        m[p] = 10


if __name__ == "__main__":
    with ExpectError():
        m.free(p)


# ### Checking Memory Usage with Valgrind

if __name__ == "__main__":
    print('\n### Checking Memory Usage with Valgrind')




if __name__ == "__main__":
    print_content(open('testuseafterfree.c').read())


if __name__ == "__main__":
    import os
    os.system(f'valgrind ./testuseafterfree')


if __name__ == "__main__":
    print_content(open('testoverflow.c').read())


if __name__ == "__main__":
    import os
    os.system(f'valgrind ./testoverflow')


# ### Checking Memory Usage with Memory Sanitizer

if __name__ == "__main__":
    print('\n### Checking Memory Usage with Memory Sanitizer')




if __name__ == "__main__":
    import os
    os.system(f'cc -fsanitize=address -o testuseafterfree testuseafterfree.c')


if __name__ == "__main__":
    import os
    os.system(f'./testuseafterfree')


if __name__ == "__main__":
    import os
    os.system(f'cc -fsanitize=address -o testoverflow testoverflow.c')


if __name__ == "__main__":
    import os
    os.system(f'./testoverflow')


# ## When Should Invariants be Checked?

if __name__ == "__main__":
    print('\n## When Should Invariants be Checked?')




# ### Assertions are not Production Code

if __name__ == "__main__":
    print('\n### Assertions are not Production Code')




# ### For System Preconditions, Use Production Code 

if __name__ == "__main__":
    print('\n### For System Preconditions, Use Production Code ')




# ### Consider Leaving Some Assertions On

if __name__ == "__main__":
    print('\n### Consider Leaving Some Assertions On')




# ### Define How Your Application Should Handle Internal Errors

if __name__ == "__main__":
    print('\n### Define How Your Application Should Handle Internal Errors')




# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




def my_square_root(x):
    assert x >= 0
    y = square_root(x)
    assert math.isclose(y * y, x)
    return y

if __name__ == "__main__":
    with ExpectError():
        y = my_square_root(-1)


if __name__ == "__main__":
    m = ManagedMemory()
    m.show()


if __name__ == "__main__":
    with ExpectError():
        x = m[2]


# ## Lessons Learned

if __name__ == "__main__":
    print('\n## Lessons Learned')




import os
os.system('rm -fr assert.h testassert* testoverflow* testuseafterfree*');

# ## Next Steps

if __name__ == "__main__":
    print('\n## Next Steps')




# ## Background

if __name__ == "__main__":
    print('\n## Background')




# ## Exercises

if __name__ == "__main__":
    print('\n## Exercises')



