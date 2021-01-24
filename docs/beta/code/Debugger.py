#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Debugging Book".
# Web site: https://www.debuggingbook.org/html/Debugger.html
# Last change: 2021-01-23 13:09:42+01:00
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


# # How Debuggers Work

if __name__ == "__main__":
    print('# How Debuggers Work')




if __name__ == "__main__":
    from bookutils import YouTubeVideo
    YouTubeVideo("4aZ0t7CWSjA")


if __name__ == "__main__":
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)


import sys

if __package__ is None or __package__ == "":
    import Tracer
else:
    from . import Tracer


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




# ## Debuggers

if __name__ == "__main__":
    print('\n## Debuggers')




# ## Debugger Interaction

if __name__ == "__main__":
    print('\n## Debugger Interaction')




if __package__ is None or __package__ == "":
    from Tracer import Tracer
else:
    from .Tracer import Tracer


class Debugger(Tracer):
    """Interactive Debugger"""

    def __init__(self, file=sys.stdout):
        """Create a new interactive debugger."""
        self.stepping = True
        self.breakpoints = set()
        self.interact = True

        self.frame = None
        self.local_vars = None
        self.event = None
        self.arg = None

        super().__init__(file)

class Debugger(Debugger):
    def traceit(self, frame, event, arg):
        """Tracing function; called at every line. To be overloaded in subclasses."""
        self.frame = frame
        self.local_vars = frame.f_locals  # Dereference exactly once
        self.event = event
        self.arg = arg

        if self.stop_here():
            self.interaction_loop()

        return self.traceit

class Debugger(Debugger):
    def stop_here(self):
        # Return true if we should stop
        return self.stepping or self.frame.f_lineno in self.breakpoints

class Debugger(Debugger):
    def interaction_loop(self):
        # Interact with the user
        self.print_debugger_status(self.frame, self.event, self.arg)

        self.interact = True
        while self.interact:
            command = input("(debugger) ")
            self.execute(command)

class Debugger(Debugger):
    def step_command(self, arg=""):
        """Execute up to the next line"""
        self.stepping = True
        self.interact = False

class Debugger(Debugger):
    def continue_command(self, arg=""):
        """Resume execution"""
        self.stepping = False
        self.interact = False

class Debugger(Debugger):
    def execute(self, command):
        if command.startswith('s'):
            self.step_command()
        elif command.startswith('c'):
            self.continue_command()

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

if __package__ is None or __package__ == "":
    from bookutils import input, next_inputs
else:
    from .bookutils import input, next_inputs


if __name__ == "__main__":
    next_inputs(["step", "step", "continue"])


if __name__ == "__main__":
    with Debugger():
        remove_html_markup('abc')


if __name__ == "__main__":
    assert not next_inputs()


# ### A Command Dispatcher

if __name__ == "__main__":
    print('\n### A Command Dispatcher')




# ### Excursion: Implementing execute()

if __name__ == "__main__":
    print('\n### Excursion: Implementing execute()')




class Debugger(Debugger):
    def commands(self):
        # Return a list of commands
        cmds = [method.replace('_command', '')
                for method in dir(self.__class__)
                if method.endswith('_command')]
        cmds.sort()
        return cmds

if __name__ == "__main__":
    d = Debugger()
    d.commands()


class Debugger(Debugger):
    def command_method(self, command):
        # Convert `command` into the method to be called
        if command.startswith('#'):
            return None  # Comment

        possible_cmds = [possible_cmd for possible_cmd in self.commands()
                         if possible_cmd.startswith(command)]
        if len(possible_cmds) != 1:
            self.help_command(command)
            return None

        cmd = possible_cmds[0]
        return getattr(self, cmd + '_command')

if __name__ == "__main__":
    d = Debugger()
    d.command_method("step")


if __name__ == "__main__":
    d = Debugger()
    d.command_method("s")


class Debugger(Debugger):
    def execute(self, command):
        # Execute given command
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
    def help_command(self, command=""):
        """Give help on given command. If no command is given, give help on all"""

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

if __name__ == "__main__":
    d = Debugger()
    d.execute("help")


if __name__ == "__main__":
    d = Debugger()
    d.execute("foo")


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ## Printing Values

if __name__ == "__main__":
    print('\n## Printing Values')




class Debugger(Debugger):
    def print_command(self, arg=""):
        """Print an expression. If no expression is given, print all variables"""
        vars = self.local_vars
        self.log("\n".join([f"{var} = {repr(vars[var])}" for var in vars]))

if __name__ == "__main__":
    next_inputs(["step", "step", "step", "print", "continue"]);


if __name__ == "__main__":
    with Debugger():
        remove_html_markup('abc')


if __name__ == "__main__":
    assert not next_inputs()


class Debugger(Debugger):
    def print_command(self, arg=""):
        """Print an expression. If no expression is given, print all variables"""
        vars = self.local_vars

        if not arg:
            self.log("\n".join([f"{var} = {repr(vars[var])}" for var in vars]))
        else:
            try:
                self.log(f"{arg} = {repr(eval(arg, globals(), vars))}")
            except Exception as err:
                self.log(f"{err.__class__.__name__}: {err}")

if __name__ == "__main__":
    next_inputs(["p s", "c"]);


if __name__ == "__main__":
    with Debugger():
        remove_html_markup('abc')


if __name__ == "__main__":
    assert not next_inputs()


if __name__ == "__main__":
    next_inputs(["print (s[0], 2 + 2)", "continue"]);


if __name__ == "__main__":
    with Debugger():
        remove_html_markup('abc')


if __name__ == "__main__":
    next_inputs(["help print", "continue"]);


if __name__ == "__main__":
    with Debugger():
        remove_html_markup('abc')


if __name__ == "__main__":
    assert not next_inputs()


# ## Listing Source Code

if __name__ == "__main__":
    print('\n## Listing Source Code')




import inspect

if __package__ is None or __package__ == "":
    from bookutils import getsourcelines  # like inspect.getsourcelines(), but in color
else:
    from .bookutils import getsourcelines  # like inspect.getsourcelines(), but in color


class Debugger(Debugger):
    def list_command(self, arg=""):
        """Show current function."""
        source_lines, line_number = getsourcelines(self.frame.f_code)

        for line in source_lines:
            self.log(f'{line_number:4} {line}', end='')
            line_number += 1

if __name__ == "__main__":
    next_inputs(["list", "continue"]);


if __name__ == "__main__":
    with Debugger():
        remove_html_markup('abc')


if __name__ == "__main__":
    assert not next_inputs()


# ## Setting Breakpoints

if __name__ == "__main__":
    print('\n## Setting Breakpoints')




class Debugger(Debugger):
    def break_command(self, arg=""):
        """Set a breakoint in given line. If no line is given, list all breakpoints"""
        if arg:
            self.breakpoints.add(int(arg))
        self.log("Breakpoints:", self.breakpoints)

if __name__ == "__main__":
    _, remove_html_markup_starting_line_number = \
        inspect.getsourcelines(remove_html_markup)
    next_inputs([f"break {remove_html_markup_starting_line_number + 13}",
                 "continue", "print", "continue", "continue", "continue"]);


if __name__ == "__main__":
    with Debugger():
        remove_html_markup('abc')


if __name__ == "__main__":
    assert not next_inputs()


if __package__ is None or __package__ == "":
    from bookutils import quiz
else:
    from .bookutils import quiz


if __name__ == "__main__":
    quiz("What happens if we enter the command `break 2 + 3`?",
         [
             "A breakpoint is set in Line 2.",
             "A breakpoint is set in Line 5.",
             "Two breakpoints are set in Lines 2 and 3.",
             "The debugger raises a `ValueError` exception."
         ], 12345 % 7)


# ## Deleting Breakpoints

if __name__ == "__main__":
    print('\n## Deleting Breakpoints')




class Debugger(Debugger):
    def delete_command(self, arg=""):
        """Delete breakoint in given line. Without given line, clear all breakpoints"""
        if arg:
            try:
                self.breakpoints.remove(int(arg))
            except KeyError:
                self.log(f"No such breakpoint: {arg}")
        else:
            self.breakpoints = set()
        self.log("Breakpoints:", self.breakpoints)

if __name__ == "__main__":
    next_inputs([f"break {remove_html_markup_starting_line_number + 15}",
                 "continue", "print",
                 f"delete {remove_html_markup_starting_line_number + 15}",
                 "continue"]);


if __name__ == "__main__":
    with Debugger():
        remove_html_markup('abc')


if __name__ == "__main__":
    assert not next_inputs()


if __name__ == "__main__":
    quiz("What does the command `delete` (without argument) do?",
        [
            "It deletes all breakpoints",
            "It deletes the source code",
            "It lists all breakpoints",
            "It stops execution"
        ],
        [n for n in range(2 // 2, 2 * 2) if n % 2 / 2]
        )


# ## Listings with Benefits

if __name__ == "__main__":
    print('\n## Listings with Benefits')




class Debugger(Debugger):
    def list_command(self, arg=""):
        """Show current function. If arg is given, show its source code."""
        if arg:
            try:
                obj = eval(arg)
                source_lines, line_number = inspect.getsourcelines(obj)
            except Exception as err:
                self.log(f"{err.__class__.__name__}: {err}")
                return
            current_line = -1
        else:
            source_lines, line_number = \
                getsourcelines(self.frame.f_code)
            current_line = self.frame.f_lineno

        for line in source_lines:
            spacer = ' '
            if line_number == current_line:
                spacer = '>'
            elif line_number in self.breakpoints:
                spacer = '#'
            self.log(f'{line_number:4}{spacer} {line}', end='')
            line_number += 1

if __name__ == "__main__":
    _, remove_html_markup_starting_line_number = \
        inspect.getsourcelines(remove_html_markup)
    next_inputs([f"break {remove_html_markup_starting_line_number + 13}",
                 "list", "continue", "delete", "list", "continue"]);


if __name__ == "__main__":
    with Debugger():
        remove_html_markup('abc')


if __name__ == "__main__":
    assert not next_inputs()


# ### Quitting

if __name__ == "__main__":
    print('\n### Quitting')




class Debugger(Debugger):
    def quit_command(self, arg=""):
        """Finish execution"""
        self.breakpoints = []
        self.stepping = False
        self.interact = False

if __name__ == "__main__":
    next_inputs(["help", "quit"]);


if __name__ == "__main__":
    with Debugger():
        remove_html_markup('abc')


if __name__ == "__main__":
    assert not next_inputs()


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




if __name__ == "__main__":
    _, remove_html_markup_starting_line_number = \
        inspect.getsourcelines(remove_html_markup)
    next_inputs(["help", f"break {remove_html_markup_starting_line_number + 13}",
                 "list", "continue", "step", "print out", "quit"]);


if __name__ == "__main__":
    with Debugger():
        ret = remove_html_markup('abc')


if __name__ == "__main__":
    assert not next_inputs()


if __package__ is None or __package__ == "":
    from ClassDiagram import display_class_hierarchy
else:
    from .ClassDiagram import display_class_hierarchy


if __name__ == "__main__":
    display_class_hierarchy(Debugger, 
                            public_methods=[
                                Tracer.__init__,
                                Tracer.__enter__,
                                Tracer.__exit__,
                                Tracer.traceit,
                                Debugger.__init__,
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




# ### Exercise 1: Changing State

if __name__ == "__main__":
    print('\n### Exercise 1: Changing State')




class Debugger(Debugger):
    def assign_command(self, arg):
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

if __name__ == "__main__":
    next_inputs(["assign s = 'xyz'", "print", "step", "print", "step",
                 "assign tag = True", "assign s = 'abc'", "print",
                 "step", "print", "continue"]);


if __name__ == "__main__":
    with Debugger():
        remove_html_markup('abc')


if __name__ == "__main__":
    assert not next_inputs()


# ### Exercise 2: More Commands

if __name__ == "__main__":
    print('\n### Exercise 2: More Commands')




# #### Named breakpoints ("break")

if __name__ == "__main__":
    print('\n#### Named breakpoints ("break")')




# #### Step over functions ("next")

if __name__ == "__main__":
    print('\n#### Step over functions ("next")')




# #### Print call stack ("where")

if __name__ == "__main__":
    print('\n#### Print call stack ("where")')




# #### Move up and down the call stack ("up" and "down")

if __name__ == "__main__":
    print('\n#### Move up and down the call stack ("up" and "down")')




# #### Execute until line ("until")

if __name__ == "__main__":
    print('\n#### Execute until line ("until")')




# #### Execute until return ("finish")

if __name__ == "__main__":
    print('\n#### Execute until return ("finish")')




# #### Watchpoints ("watch")

if __name__ == "__main__":
    print('\n#### Watchpoints ("watch")')




# ### Exercise 3: Time-Travel Debugging

if __name__ == "__main__":
    print('\n### Exercise 3: Time-Travel Debugging')




# #### Part 1: Recording Values

if __name__ == "__main__":
    print('\n#### Part 1: Recording Values')




# #### Part 2: Command Line Interface

if __name__ == "__main__":
    print('\n#### Part 2: Command Line Interface')




# #### Part 3: Graphical User Interface

if __name__ == "__main__":
    print('\n#### Part 3: Graphical User Interface')




if __name__ == "__main__":
    recording = [
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


if __package__ is None or __package__ == "":
    from bookutils import HTML
else:
    from .bookutils import HTML


def slider(rec):
    lines_over_time = [line for (line, var) in rec]
    vars_over_time = []
    for (line, vars) in rec:
        vars_over_time.append(", ".join(f"{var} = {repr(vars[var])}"
                                        for var in vars))

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

if __name__ == "__main__":
    slider(recording)

