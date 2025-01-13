#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# "How Debuggers Work" - a chapter of "The Debugging Book"
# Web site: https://www.debuggingbook.org/html/Debugger.html
# Last change: 2025-01-13 15:53:38+01:00
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
The Debugging Book - How Debuggers Work

This file can be _executed_ as a script, running all experiments:

    $ python Debugger.py

or _imported_ as a package, providing classes, functions, and constants:

    >>> from debuggingbook.Debugger import <identifier>

but before you do so, _read_ it and _interact_ with it at:

    https://www.debuggingbook.org/html/Debugger.html

This chapter provides an interactive debugger for Python functions. The debugger is invoked as

with Debugger():
    function_to_be_observed()
    ...

While running, you can enter _debugger commands_ at the `(debugger)` prompt. Here's an example session:

>>> with Debugger():
>>>     ret = remove_html_markup('abc')
Calling remove_html_markup(s = 'abc')


(debugger) help

break      -- Set a breakpoint in given line. If no line is given, list all breakpoints

continue   -- Resume execution

delete     -- Delete breakpoint in line given by `arg`.
           Without given line, clear all breakpoints

help       -- Give help on given `command`. If no command is given, give help on all

list       -- Show current function. If `arg` is given, show its source code.

print      -- Print an expression. If no expression is given, print all variables

quit       -- Finish execution

step       -- Execute up to the next line


(debugger) break 14

Breakpoints: {14}


(debugger) list

   1> def remove_html_markup(s):  # type: ignore

   2      tag = False

   3      quote = False

   4      out = ""

   5  

   6      for c in s:

   7          if c == '<' and not quote:

   8              tag = True

   9          elif c == '>' and not quote:

  10              tag = False

  11          elif c == '"' or c == "'" and tag:

  12              quote = not quote

  13          elif not tag:

  14#             out = out + c

  15  

  16      return out


(debugger) continue

                                         # tag = False, quote = False, out = '', c = 'a'

14             out = out + c


(debugger) step

                                         # out = 'a'

6     for c in s:


(debugger) print out

out = 'a'


(debugger) quit

The `Debugger` class can be easily extended in subclasses. A new method `NAME_command(self, arg)` will be invoked whenever a command named `NAME` is entered, with `arg` holding given command arguments (empty string if none).

For more details, source, and documentation, see
"The Debugging Book - How Debuggers Work"
at https://www.debuggingbook.org/html/Debugger.html
'''


# Allow to use 'from . import <module>' when run as script (cf. PEP 366)
if __name__ == '__main__' and __package__ is None:
    __package__ = 'debuggingbook'


# How Debuggers Work
# ==================

if __name__ == '__main__':
    print('# How Debuggers Work')



if __name__ == '__main__':
    from .bookutils import YouTubeVideo
    YouTubeVideo("4aZ0t7CWSjA")

if __name__ == '__main__':
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)

import sys

from .Tracer import Tracer

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



## Debuggers
## ---------

if __name__ == '__main__':
    print('\n## Debuggers')



## Debugger Interaction
## --------------------

if __name__ == '__main__':
    print('\n## Debugger Interaction')



from types import FrameType

from typing import Any, Optional, Callable, Dict, List, Tuple, Set, TextIO

class Debugger(Tracer):
    """Interactive Debugger"""

    def __init__(self, *, file: TextIO = sys.stdout) -> None:
        """Create a new interactive debugger."""
        self.stepping: bool = True
        self.breakpoints: Set[int] = set()
        self.interact: bool = True

        self.frame: FrameType
        self.event: Optional[str] = None
        self.arg: Any = None

        self.local_vars: Dict[str, Any] = {}

        super().__init__(file=file)

class Debugger(Debugger):
    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """Tracing function; called at every line. To be overloaded in subclasses."""
        self.frame = frame
        self.local_vars = frame.f_locals  # Dereference exactly once
        self.event = event
        self.arg = arg

        if self.stop_here():
            self.interaction_loop()

class Debugger(Debugger):
    def stop_here(self) -> bool:
        """Return True if we should stop"""
        return self.stepping or self.frame.f_lineno in self.breakpoints

class Debugger(Debugger):
    def interaction_loop(self) -> None:
        """Interact with the user"""
        self.print_debugger_status(self.frame, self.event, self.arg)  # type: ignore

        self.interact = True
        while self.interact:
            command = input("(debugger) ")
            self.execute(command)  # type: ignore

class Debugger(Debugger):
    def step_command(self, arg: str = "") -> None:
        """Execute up to the next line"""

        self.stepping = True
        self.interact = False

class Debugger(Debugger):
    def continue_command(self, arg: str = "") -> None:
        """Resume execution"""

        self.stepping = False
        self.interact = False

class Debugger(Debugger):
    def execute(self, command: str) -> None:
        if command.startswith('s'):
            self.step_command()
        elif command.startswith('c'):
            self.continue_command()

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

from .bookutils import input, next_inputs

if __name__ == '__main__':
    next_inputs(["step", "step", "continue"])

if __name__ == '__main__':
    with Debugger():
        remove_html_markup('abc')

### A Command Dispatcher

if __name__ == '__main__':
    print('\n### A Command Dispatcher')



### Excursion: Implementing execute()

if __name__ == '__main__':
    print('\n### Excursion: Implementing execute()')



class Debugger(Debugger):
    def commands(self) -> List[str]:
        """Return a list of commands"""

        cmds = [method.replace('_command', '')
                for method in dir(self.__class__)
                if method.endswith('_command')]
        cmds.sort()
        return cmds

if __name__ == '__main__':
    d = Debugger()
    d.commands()

class Debugger(Debugger):
    def help_command(self, command: str) -> None:
        ...

    def command_method(self, command: str) -> Optional[Callable[[str], None]]:
        """Convert `command` into the method to be called.
           If the method is not found, return `None` instead."""

        if command.startswith('#'):
            return None  # Comment

        possible_cmds = [possible_cmd for possible_cmd in self.commands()
                         if possible_cmd.startswith(command)]
        if len(possible_cmds) != 1:
            self.help_command(command)
            return None

        cmd = possible_cmds[0]
        return getattr(self, cmd + '_command')

if __name__ == '__main__':
    d = Debugger()
    d.command_method("step")

if __name__ == '__main__':
    d = Debugger()
    d.command_method("s")

class Debugger(Debugger):
    def execute(self, command: str) -> None:
        """Execute `command`"""

        sep = command.find(' ')
        if sep > 0:
            cmd = command[:sep].strip()
            arg = command[sep + 1:].strip()
        else:
            cmd = command.strip()
            arg = ""

        method = self.command_method(cmd)
        if method:
            method(arg)

class Debugger(Debugger):
    def help_command(self, command: str = "") -> None:
        """Give help on given `command`. If no command is given, give help on all"""

        if command:
            possible_cmds = [possible_cmd for possible_cmd in self.commands()
                             if possible_cmd.startswith(command)]

            if len(possible_cmds) == 0:
                self.log(f"Unknown command {repr(command)}. Possible commands are:")
                possible_cmds = self.commands()
            elif len(possible_cmds) > 1:
                self.log(f"Ambiguous command {repr(command)}. Possible expansions are:")
        else:
            possible_cmds = self.commands()

        for cmd in possible_cmds:
            method = self.command_method(cmd)
            self.log(f"{cmd:10} -- {method.__doc__}")

if __name__ == '__main__':
    d = Debugger()
    d.execute("help")

if __name__ == '__main__':
    d = Debugger()
    d.execute("foo")

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



## Printing Values
## ---------------

if __name__ == '__main__':
    print('\n## Printing Values')



class Debugger(Debugger):
    def print_command(self, arg: str = "") -> None:
        """Print an expression. If no expression is given, print all variables"""

        vars = self.local_vars
        self.log("\n".join([f"{var} = {repr(value)}" for var, value in vars.items()]))

if __name__ == '__main__':
    next_inputs(["step", "step", "step", "print", "continue"]);

if __name__ == '__main__':
    with Debugger():
        remove_html_markup('abc')

class Debugger(Debugger):
    def print_command(self, arg: str = "") -> None:
        """Print an expression. If no expression is given, print all variables"""

        vars = self.local_vars

        if not arg:
            self.log("\n".join([f"{var} = {repr(value)}" for var, value in vars.items()]))
        else:
            try:
                self.log(f"{arg} = {repr(eval(arg, globals(), vars))}")
            except Exception as err:
                self.log(f"{err.__class__.__name__}: {err}")

if __name__ == '__main__':
    next_inputs(["p s", "c"]);

if __name__ == '__main__':
    with Debugger():
        remove_html_markup('abc')

if __name__ == '__main__':
    next_inputs(["print (s[0], 2 + 2)", "continue"]);

if __name__ == '__main__':
    with Debugger():
        remove_html_markup('abc')

if __name__ == '__main__':
    next_inputs(["help print", "continue"]);

if __name__ == '__main__':
    with Debugger():
        remove_html_markup('abc')

## Listing Source Code
## -------------------

if __name__ == '__main__':
    print('\n## Listing Source Code')



import inspect

from .bookutils import getsourcelines  # like inspect.getsourcelines(), but in color

class Debugger(Debugger):
    def list_command(self, arg: str = "") -> None:
        """Show current function."""

        source_lines, line_number = getsourcelines(self.frame.f_code)

        for line in source_lines:
            self.log(f'{line_number:4} {line}', end='')
            line_number += 1

if __name__ == '__main__':
    next_inputs(["list", "continue"]);

if __name__ == '__main__':
    with Debugger():
        remove_html_markup('abc')

## Setting Breakpoints
## -------------------

if __name__ == '__main__':
    print('\n## Setting Breakpoints')



class Debugger(Debugger):
    def break_command(self, arg: str = "") -> None:
        """Set a breakpoint in given line. If no line is given, list all breakpoints"""

        if arg:
            self.breakpoints.add(int(arg))
        self.log("Breakpoints:", self.breakpoints)

if __name__ == '__main__':
    _, remove_html_markup_starting_line_number = \
        inspect.getsourcelines(remove_html_markup)
    next_inputs([f"break {remove_html_markup_starting_line_number + 13}",
                 "continue", "print", "continue", "continue", "continue"]);

if __name__ == '__main__':
    with Debugger():
        remove_html_markup('abc')

from .bookutils import quiz

if __name__ == '__main__':
    quiz("What happens if we enter the command `break 2 + 3`?",
         [
             "A breakpoint is set in Line 2.",
             "A breakpoint is set in Line 5.",
             "Two breakpoints are set in Lines 2 and 3.",
             "The debugger raises a `ValueError` exception."
         ], '12345 % 7')

## Deleting Breakpoints
## --------------------

if __name__ == '__main__':
    print('\n## Deleting Breakpoints')



class Debugger(Debugger):
    def delete_command(self, arg: str = "") -> None:
        """Delete breakpoint in line given by `arg`.
           Without given line, clear all breakpoints"""

        if arg:
            try:
                self.breakpoints.remove(int(arg))
            except KeyError:
                self.log(f"No such breakpoint: {arg}")
        else:
            self.breakpoints = set()
        self.log("Breakpoints:", self.breakpoints)

if __name__ == '__main__':
    next_inputs([f"break {remove_html_markup_starting_line_number + 15}",
                 "continue", "print",
                 f"delete {remove_html_markup_starting_line_number + 15}",
                 "continue"]);

if __name__ == '__main__':
    with Debugger():
        remove_html_markup('abc')

if __name__ == '__main__':
    quiz("What does the command `delete` (without argument) do?",
        [
            "It deletes all breakpoints",
            "It deletes the source code",
            "It lists all breakpoints",
            "It stops execution"
        ],
        '[n for n in range(2 // 2, 2 * 2) if n % 2 / 2]'
        )

## Listings with Benefits
## ----------------------

if __name__ == '__main__':
    print('\n## Listings with Benefits')



class Debugger(Debugger):
    def list_command(self, arg: str = "") -> None:
        """Show current function. If `arg` is given, show its source code."""

        try:
            if arg:
                obj = eval(arg)
                source_lines, line_number = inspect.getsourcelines(obj)
                current_line = -1
            else:
                source_lines, line_number = \
                    getsourcelines(self.frame.f_code)
                current_line = self.frame.f_lineno
        except Exception as err:
            self.log(f"{err.__class__.__name__}: {err}")
            source_lines = []
            line_number = 0

        for line in source_lines:
            spacer = ' '
            if line_number == current_line:
                spacer = '>'
            elif line_number in self.breakpoints:
                spacer = '#'
            self.log(f'{line_number:4}{spacer} {line}', end='')
            line_number += 1

if __name__ == '__main__':
    _, remove_html_markup_starting_line_number = \
        inspect.getsourcelines(remove_html_markup)
    next_inputs([f"break {remove_html_markup_starting_line_number + 13}",
                 "list", "continue", "delete", "list", "continue"]);

if __name__ == '__main__':
    with Debugger():
        remove_html_markup('abc')

### Quitting

if __name__ == '__main__':
    print('\n### Quitting')



class Debugger(Debugger):
    def quit_command(self, arg: str = "") -> None:
        """Finish execution"""

        self.breakpoints = set()
        self.stepping = False
        self.interact = False

if __name__ == '__main__':
    next_inputs(["help", "quit"]);

if __name__ == '__main__':
    with Debugger():
        remove_html_markup('abc')

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



if __name__ == '__main__':
    _, remove_html_markup_starting_line_number = \
        inspect.getsourcelines(remove_html_markup)
    next_inputs(["help", f"break {remove_html_markup_starting_line_number + 13}",
                 "list", "continue", "step", "print out", "quit"])
    pass

if __name__ == '__main__':
    with Debugger():
        ret = remove_html_markup('abc')

from .ClassDiagram import display_class_hierarchy

if __name__ == '__main__':
    display_class_hierarchy(Debugger, 
                            public_methods=[
                                Tracer.__init__,
                                Tracer.__enter__,
                                Tracer.__exit__,
                                Tracer.traceit,
                                Debugger.__init__,
                            ],
        project='debuggingbook')

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



## Exercises
## ---------

if __name__ == '__main__':
    print('\n## Exercises')



### Exercise 1: Changing State

if __name__ == '__main__':
    print('\n### Exercise 1: Changing State')



class Debugger(Debugger):
    def assign_command(self, arg: str) -> None:
        """Use as 'assign VAR=VALUE'. Assign VALUE to local variable VAR."""

        sep = arg.find('=')
        if sep > 0:
            var = arg[:sep].strip()
            expr = arg[sep + 1:].strip()
        else:
            self.help_command("assign")
            return

        vars = self.local_vars
        try:
            vars[var] = eval(expr, self.frame.f_globals, vars)
        except Exception as err:
            self.log(f"{err.__class__.__name__}: {err}")

if __name__ == '__main__':
    next_inputs(["assign s = 'xyz'", "print", "step", "print", "step",
                 "assign tag = True", "assign s = 'abc'", "print",
                 "step", "print", "continue"]);

if __name__ == '__main__':
    with Debugger():
        remove_html_markup('abc')

### Exercise 2: More Commands

if __name__ == '__main__':
    print('\n### Exercise 2: More Commands')



#### Named breakpoints ("break")

if __name__ == '__main__':
    print('\n#### Named breakpoints ("break")')



#### Step over functions ("next")

if __name__ == '__main__':
    print('\n#### Step over functions ("next")')



#### Print call stack ("where")

if __name__ == '__main__':
    print('\n#### Print call stack ("where")')



#### Move up and down the call stack ("up" and "down")

if __name__ == '__main__':
    print('\n#### Move up and down the call stack ("up" and "down")')



#### Execute until line ("until")

if __name__ == '__main__':
    print('\n#### Execute until line ("until")')



#### Execute until return ("finish")

if __name__ == '__main__':
    print('\n#### Execute until return ("finish")')



#### Watchpoints ("watch")

if __name__ == '__main__':
    print('\n#### Watchpoints ("watch")')



### Exercise 3: Time-Travel Debugging

if __name__ == '__main__':
    print('\n### Exercise 3: Time-Travel Debugging')



#### Part 1: Recording Values

if __name__ == '__main__':
    print('\n#### Part 1: Recording Values')



#### Part 2: Command Line Interface

if __name__ == '__main__':
    print('\n#### Part 2: Command Line Interface')



#### Part 3: Graphical User Interface

if __name__ == '__main__':
    print('\n#### Part 3: Graphical User Interface')



if __name__ == '__main__':
    recording: List[Tuple[int, Dict[str, Any]]] = [
        (10, {'x': 25}),
        (11, {'x': 25}),
        (12, {'x': 26, 'a': "abc"}),
        (13, {'x': 26, 'a': "abc"}),
        (10, {'x': 30}),
        (11, {'x': 30}),
        (12, {'x': 31, 'a': "def"}),
        (13, {'x': 31, 'a': "def"}),
        (10, {'x': 35}),
        (11, {'x': 35}),
        (12, {'x': 36, 'a': "ghi"}),
        (13, {'x': 36, 'a': "ghi"}),
    ]

from .bookutils import HTML

def slider(rec: List[Tuple[int, Dict[str, Any]]]) -> str:
    lines_over_time = [line for (line, var) in rec]
    vars_over_time = []
    for (line, vars) in rec:
        vars_over_time.append(", ".join(f"{var} = {repr(value)}"
                                        for var, value in vars.items()))

    # print(lines_over_time)
    # print(vars_over_time)

    template = f'''
    <div class="time_travel_debugger">
      <input type="range" min="0" max="{len(lines_over_time) - 1}"
      value="0" class="slider" id="time_slider">
      Line <span id="line">{lines_over_time[0]}</span>:
      <span id="vars">{vars_over_time[0]}</span>
    </div>
    <script>
       var lines_over_time = {lines_over_time};
       var vars_over_time = {vars_over_time};

       var time_slider = document.getElementById("time_slider");
       var line = document.getElementById("line");
       var vars = document.getElementById("vars");

       time_slider.oninput = function() {{
          line.innerHTML = lines_over_time[this.value];
          vars.innerHTML = vars_over_time[this.value];
       }}
    </script>
    '''
    # print(template)
    return HTML(template)

if __name__ == '__main__':
    slider(recording)
