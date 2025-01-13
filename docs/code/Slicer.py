#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# "Tracking Failure Origins" - a chapter of "The Debugging Book"
# Web site: https://www.debuggingbook.org/html/Slicer.html
# Last change: 2025-01-13 15:54:18+01:00
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
The Debugging Book - Tracking Failure Origins

This file can be _executed_ as a script, running all experiments:

    $ python Slicer.py

or _imported_ as a package, providing classes, functions, and constants:

    >>> from debuggingbook.Slicer import <identifier>

but before you do so, _read_ it and _interact_ with it at:

    https://www.debuggingbook.org/html/Slicer.html

This chapter provides a `Slicer` class to automatically determine and visualize dynamic flows and dependencies. When we say that a variable $x$ _depends_ on a variable $y$ (and that $y$ _flows_ into $x$), we distinguish two kinds of dependencies:

* **Data dependency**: $x$ is assigned a value computed from $y$.
* **Control dependency**: A statement involving $x$ is executed _only_ because a _condition_ involving $y$ was evaluated, influencing the execution path.

Such dependencies are crucial for debugging, as they allow determininh the origins of individual values (and notably incorrect values).

To determine dynamic dependencies in a function `func`, use

with Slicer() as slicer:
    


and then `slicer.graph()` or `slicer.code()` to examine dependencies.

You can also explicitly specify the functions to be instrumented, as in 

with Slicer(func, func_1, func_2) as slicer:
    


Here is an example. The `demo()` function computes some number from `x`:

>>> def demo(x: int) -> int:
>>>     z = x
>>>     while x <= z <= 64:
>>>         z *= 2
>>>     return z

By using `with Slicer()`, we first instrument `demo()` and then execute it:

>>> with Slicer() as slicer:
>>>     demo(10)
/var/folders/n2/xd9445p97rb3xh7m1dfx8_4h0006ts/T/ipykernel_2778/2454789564.py:22: DeprecationWarning: ast.Str is deprecated and will be removed in Python 3.14; use ast.Constant instead
  args=[ast.Str(id), value],
/var/folders/n2/xd9445p97rb3xh7m1dfx8_4h0006ts/T/ipykernel_2778/3554319793.py:4: DeprecationWarning: ast.Str is deprecated and will be removed in Python 3.14; use ast.Constant instead
  args=[ast.Str(id), Name(id=id, ctx=Load())],
/var/folders/n2/xd9445p97rb3xh7m1dfx8_4h0006ts/T/ipykernel_2778/882995308.py:12: DeprecationWarning: ast.Num is deprecated and will be removed in Python 3.14; use ast.Constant instead
  keywords=[keyword(arg='pos', value=ast.Num(n + 1))]
/var/folders/n2/xd9445p97rb3xh7m1dfx8_4h0006ts/T/ipykernel_2778/882995308.py:19: DeprecationWarning: ast.NameConstant is deprecated and will be removed in Python 3.14; use ast.Constant instead
  value=ast.NameConstant(value=True)))
/var/folders/n2/xd9445p97rb3xh7m1dfx8_4h0006ts/T/ipykernel_2778/882995308.py:25: DeprecationWarning: ast.Str is deprecated and will be removed in Python 3.14; use ast.Constant instead
  args=[ast.Str(child.arg),


After execution is complete, you can output `slicer` to visualize the dependencies and flows as graph. Data dependencies are shown as black solid edges; control dependencies are shown as grey dashed edges. The arrows indicate influence: If $y$ depends on $x$ (and thus $x$ flows into $y$), then we have an arrow $x \rightarrow y$.
We see how the parameter `x` flows into `z`, which is returned after some computation that is control dependent on a `` involving `z`.

>>> slicer
An alternate representation is `slicer.code()`, annotating the instrumented source code with (backward) dependencies. Data dependencies are shown with `<=`, control dependencies with `<-`; locations (lines) are shown in parentheses.

>>> slicer.code()
     1 def demo(x: int) -> int:
*    2     z = x  # <= x (5)
*    3     while x <= z <= 64:  # <= x (5), z (2), z (4)
*    4         z *= 2  # <= z (2), z (4); <-  (3)
*    5     return z  # <= z (4)


Dependencies can also be retrieved programmatically. The `dependencies()` method returns a `Dependencies` object encapsulating the dependency graph.

The method `all_vars()` returns all variables in the dependency graph. Each variable is encoded as a pair (_name_, _location_) where _location_ is a pair (_codename_, _lineno_).

>>> slicer.dependencies().all_vars()
{('', ( int>, 5)),
 ('', ( int>, 3)),
 ('x', ( int>, 5)),
 ('z', ( int>, 2)),
 ('z', ( int>, 4))}

`code()` and `graph()` methods can also be applied on dependencies. The method `backward_slice(var)` returns a backward slice for the given variable (again given as a pair (_name_, _location_)). To retrieve where `z` in Line 2 came from, use:

>>> _, start_demo = inspect.getsourcelines(demo)
>>> start_demo
1
>>> slicer.dependencies().backward_slice(('z', (demo, start_demo + 1))).graph()  # type: ignore
Here are the classes defined in this chapter. A `Slicer` instruments a program, using a `DependencyTracker` at run time to collect `Dependencies`.

For more details, source, and documentation, see
"The Debugging Book - Tracking Failure Origins"
at https://www.debuggingbook.org/html/Slicer.html
'''


# Allow to use 'from . import <module>' when run as script (cf. PEP 366)
if __name__ == '__main__' and __package__ is None:
    __package__ = 'debuggingbook'


# Tracking Failure Origins
# ========================

if __name__ == '__main__':
    print('# Tracking Failure Origins')



if __name__ == '__main__':
    from .bookutils import YouTubeVideo
    YouTubeVideo("sjf3cOR0lcI")

if __name__ == '__main__':
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)

from .bookutils import quiz, next_inputs, print_content

import inspect
import warnings

from typing import Set, List, Tuple, Any, Callable, Dict, Optional
from typing import Union, Type, Generator, cast

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



## Dependencies
## ------------

if __name__ == '__main__':
    print('\n## Dependencies')



def middle(x, y, z):  # type: ignore
    if y < z:
        if x < y:
            return y
        elif x < z:
            return y
    else:
        if x > y:
            return y
        elif x > z:
            return x
    return z

if __name__ == '__main__':
    m = middle(1, 2, 3)
    m

if __name__ == '__main__':
    m = middle(2, 1, 3)
    m

from . import Debugger  # minor dependency

if __name__ == '__main__':
    next_inputs(["step", "step", "step", "step", "quit"]);

if __name__ == '__main__':
    with Debugger.Debugger():
        middle(2, 1, 3)

### Excursion: Visualizing Dependencies

if __name__ == '__main__':
    print('\n### Excursion: Visualizing Dependencies')



#### A Class for Dependencies

if __name__ == '__main__':
    print('\n#### A Class for Dependencies')



Location = Tuple[Callable, int]
Node = Tuple[str, Location]
Dependency = Dict[Node, Set[Node]]

from .StackInspector import StackInspector

class Dependencies(StackInspector):
    """A dependency graph"""

    def __init__(self, 
                 data: Optional[Dependency] = None,
                 control: Optional[Dependency] = None) -> None:
        """
        Create a dependency graph from `data` and `control`.
        Both `data` and `control` are dictionaries
        holding _nodes_ as keys and sets of nodes as values.
        Each node comes as a tuple (variable_name, location)
        where `variable_name` is a string 
        and `location` is a pair (function, lineno)
        where `function` is a callable and `lineno` is a line number
        denoting a unique location in the code.
        """

        if data is None:
            data = {}
        if control is None:
            control = {}

        self.data = data
        self.control = control

        for var in self.data:
            self.control.setdefault(var, set())
        for var in self.control:
            self.data.setdefault(var, set())

        self.validate()

class Dependencies(Dependencies):
    def validate(self) -> None:
        """Check dependency structure."""
        assert isinstance(self.data, dict)
        assert isinstance(self.control, dict)

        for node in (self.data.keys()) | set(self.control.keys()):
            var_name, location = node
            assert isinstance(var_name, str)
            func, lineno = location
            assert callable(func)
            assert isinstance(lineno, int)

class Dependencies(Dependencies):
    def _source(self, node: Node) -> str:
        # Return source line, or ''
        (name, location) = node
        func, lineno = location
        if not func:  # type: ignore
            # No source
            return ''

        try:
            source_lines, first_lineno = inspect.getsourcelines(func)
        except OSError:
            warnings.warn(f"Couldn't find source "
                          f"for {func} ({func.__name__})")
            return ''

        try:
            line = source_lines[lineno - first_lineno].strip()
        except IndexError:
            return ''

        return line

    def source(self, node: Node) -> str:
        """Return the source code for a given node."""
        line = self._source(node)
        if line:
            return line

        (name, location) = node
        func, lineno = location
        code_name = func.__name__

        if code_name.startswith('<'):
            return code_name
        else:
            return f'<{code_name}()>'

if __name__ == '__main__':
    test_deps = Dependencies()
    test_deps.source(('z', (middle, 1)))

#### Drawing Dependencies

if __name__ == '__main__':
    print('\n#### Drawing Dependencies')



from graphviz import Digraph

import html

class Dependencies(Dependencies):
    NODE_COLOR = 'peachpuff'
    FONT_NAME = 'Courier'  # 'Fira Mono' may produce warnings in 'dot'

    def make_graph(self,
                   name: str = "dependencies",
                   comment: str = "Dependencies") -> Digraph:
        return Digraph(name=name, comment=comment,
            graph_attr={
            },
            node_attr={
                'style': 'filled',
                'shape': 'box',
                'fillcolor': self.NODE_COLOR,
                'fontname': self.FONT_NAME
            },
            edge_attr={
                'fontname': self.FONT_NAME
            })

class Dependencies(Dependencies):
    def graph(self, *, mode: str = 'flow') -> Digraph:
        """
        Draw dependencies. `mode` is either
        * `'flow'`: arrows indicate information flow (from A to B); or
        * `'depend'`: arrows indicate dependencies (B depends on A)
        """
        self.validate()

        g = self.make_graph()
        self.draw_dependencies(g, mode)
        self.add_hierarchy(g)
        return g

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """If the object is output in Jupyter, render dependencies as a SVG graph"""
        return self.graph()._repr_mimebundle_(include, exclude)

class Dependencies(Dependencies):
    def all_vars(self) -> Set[Node]:
        """Return a set of all variables (as `var_name`, `location`) in the dependencies"""
        all_vars = set()
        for var in self.data:
            all_vars.add(var)
            for source in self.data[var]:
                all_vars.add(source)

        for var in self.control:
            all_vars.add(var)
            for source in self.control[var]:
                all_vars.add(source)

        return all_vars

class Dependencies(Dependencies):
    def draw_edge(self, g: Digraph, mode: str,
                  node_from: str, node_to: str, **kwargs: Any) -> None:
        if mode == 'flow':
            g.edge(node_from, node_to, **kwargs)
        elif mode == 'depend':
            g.edge(node_from, node_to, dir="back", **kwargs)
        else:
            raise ValueError("`mode` must be 'flow' or 'depend'")

    def draw_dependencies(self, g: Digraph, mode: str) -> None:
        for var in self.all_vars():
            g.node(self.id(var),
                   label=self.label(var),
                   tooltip=self.tooltip(var))

            if var in self.data:
                for source in self.data[var]:
                    self.draw_edge(g, mode, self.id(source), self.id(var))

            if var in self.control:
                for source in self.control[var]:
                    self.draw_edge(g, mode, self.id(source), self.id(var),
                                   style='dashed', color='grey')

class Dependencies(Dependencies):
    def id(self, var: Node) -> str:
        """Return a unique ID for `var`."""
        id = ""
        # Avoid non-identifier characters
        for c in repr(var):
            if c.isalnum() or c == '_':
                id += c
            if c == ':' or c == ',':
                id += '_'
        return id

    def label(self, var: Node) -> str:
        """Render node `var` using HTML style."""
        (name, location) = var
        source = self.source(var)

        title = html.escape(name)
        if name.startswith('<'):
            title = f'<I>{title}</I>'

        label = f'<B>{title}</B>'
        if source:
            label += (f'<FONT POINT-SIZE="9.0"><BR/><BR/>'
                      f'{html.escape(source)}'
                      f'</FONT>')
        label = f'<{label}>'
        return label

    def tooltip(self, var: Node) -> str:
        """Return a tooltip for node `var`."""
        (name, location) = var
        func, lineno = location
        return f"{func.__name__}:{lineno}"

class Dependencies(Dependencies):
    def add_hierarchy(self, g: Digraph) -> Digraph:
        """Add invisible edges for a proper hierarchy."""
        functions = self.all_functions()
        for func in functions:
            last_var = None
            last_lineno = 0
            for (lineno, var) in functions[func]:
                if last_var is not None and lineno > last_lineno:
                    g.edge(self.id(last_var),
                           self.id(var),
                           style='invis')

                last_var = var
                last_lineno = lineno

        return g

class Dependencies(Dependencies):
    def all_functions(self) -> Dict[Callable, List[Tuple[int, Node]]]:
        """
        Return mapping 
        {`function`: [(`lineno`, `var`), (`lineno`, `var`), ...], ...}
        for all functions in the dependencies.
        """
        functions: Dict[Callable, List[Tuple[int, Node]]] = {}
        for var in self.all_vars():
            (name, location) = var
            func, lineno = location
            if func not in functions:
                functions[func] = []
            functions[func].append((lineno, var))

        for func in functions:
            functions[func].sort()

        return functions

def middle_deps() -> Dependencies:
    return Dependencies({('z', (middle, 1)): set(), ('y', (middle, 1)): set(), ('x', (middle, 1)): set(), ('<test>', (middle, 2)): {('y', (middle, 1)), ('z', (middle, 1))}, ('<test>', (middle, 3)): {('y', (middle, 1)), ('x', (middle, 1))}, ('<test>', (middle, 5)): {('z', (middle, 1)), ('x', (middle, 1))}, ('<middle() return value>', (middle, 6)): {('y', (middle, 1))}}, {('z', (middle, 1)): set(), ('y', (middle, 1)): set(), ('x', (middle, 1)): set(), ('<test>', (middle, 2)): set(), ('<test>', (middle, 3)): {('<test>', (middle, 2))}, ('<test>', (middle, 5)): {('<test>', (middle, 3))}, ('<middle() return value>', (middle, 6)): {('<test>', (middle, 5))}})

if __name__ == '__main__':
    middle_deps()

if __name__ == '__main__':
    middle_deps().graph(mode='depend')

#### Slices

if __name__ == '__main__':
    print('\n#### Slices')



Criterion = Union[str, Location, Node]

class Dependencies(Dependencies):
    def expand_criteria(self, criteria: List[Criterion]) -> List[Node]:
        """Return list of vars matched by `criteria`."""
        all_vars = []
        for criterion in criteria:
            criterion_var = None
            criterion_func = None
            criterion_lineno = None

            if isinstance(criterion, str):
                criterion_var = criterion
            elif len(criterion) == 2 and callable(criterion[0]):
                criterion_func, criterion_lineno = criterion
            elif len(criterion) == 2 and isinstance(criterion[0], str):
                criterion_var = criterion[0]
                criterion_func, criterion_lineno = criterion[1]
            else:
                raise ValueError("Invalid argument")

            for var in self.all_vars():
                (var_name, location) = var
                func, lineno = location

                name_matches = (criterion_func is None or
                                criterion_func == func or
                                criterion_func.__name__ == func.__name__)

                location_matches = (criterion_lineno is None or
                                    criterion_lineno == lineno)

                var_matches = (criterion_var is None or
                               criterion_var == var_name)

                if name_matches and location_matches and var_matches:
                    all_vars.append(var)

        return all_vars

    def backward_slice(self, *criteria: Criterion, 
                       mode: str = 'cd', depth: int = -1) -> Dependencies:
        """
        Create a backward slice from nodes `criteria`.
        `mode` can contain 'c' (draw control dependencies)
        and 'd' (draw data dependencies) (default: 'cd')
        """
        data = {}
        control = {}
        queue = self.expand_criteria(criteria)  # type: ignore
        seen = set()

        while len(queue) > 0 and depth != 0:
            var = queue[0]
            queue = queue[1:]
            seen.add(var)

            if 'd' in mode:
                # Follow data dependencies
                data[var] = self.data[var]
                for next_var in data[var]:
                    if next_var not in seen:
                        queue.append(next_var)
            else:
                data[var] = set()

            if 'c' in mode:
                # Follow control dependencies
                control[var] = self.control[var]
                for next_var in control[var]:
                    if next_var not in seen:
                        queue.append(next_var)
            else:
                control[var] = set()

            depth -= 1

        return Dependencies(data, control)

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



### Data Dependencies

if __name__ == '__main__':
    print('\n### Data Dependencies')



if __name__ == '__main__':
    middle_deps().backward_slice('<middle() return value>', mode='d')  # type: ignore

### Control Dependencies

if __name__ == '__main__':
    print('\n### Control Dependencies')



if __name__ == '__main__':
    middle_deps().backward_slice('<middle() return value>', mode='c', depth=1)  # type: ignore

if __name__ == '__main__':
    middle_deps().backward_slice('<middle() return value>', mode='c')  # type: ignore

### Dependency Graphs

if __name__ == '__main__':
    print('\n### Dependency Graphs')



if __name__ == '__main__':
    middle_deps()

### Showing Dependencies with Code

if __name__ == '__main__':
    print('\n### Showing Dependencies with Code')



#### Excursion: Listing Dependencies

if __name__ == '__main__':
    print('\n#### Excursion: Listing Dependencies')



class Dependencies(Dependencies):
    def format_var(self, var: Node, current_func: Optional[Callable] = None) -> str:
        """Return string for `var` in `current_func`."""
        name, location = var
        func, lineno = location
        if current_func and (func == current_func or func.__name__ == current_func.__name__):
            return f"{name} ({lineno})"
        else:
            return f"{name} ({func.__name__}:{lineno})"

class Dependencies(Dependencies):
    def __str__(self) -> str:
        """Return string representation of dependencies"""
        self.validate()

        out = ""
        for func in self.all_functions():
            code_name = func.__name__

            if out != "":
                out += "\n"
            out += f"{code_name}():\n"

            all_vars = list(set(self.data.keys()) | set(self.control.keys()))
            all_vars.sort(key=lambda var: var[1][1])

            for var in all_vars:
                (name, location) = var
                var_func, var_lineno = location
                var_code_name = var_func.__name__

                if var_code_name != code_name:
                    continue

                all_deps = ""
                for (source, arrow) in [(self.data, "<="), (self.control, "<-")]:
                    deps = ""
                    for data_dep in source[var]:
                        if deps == "":
                            deps = f" {arrow} "
                        else:
                            deps += ", "
                        deps += self.format_var(data_dep, func)

                    if deps != "":
                        if all_deps != "":
                            all_deps += ";"
                        all_deps += deps

                if all_deps == "":
                    continue

                out += ("    " + 
                        self.format_var(var, func) +
                        all_deps + "\n")

        return out

if __name__ == '__main__':
    print(middle_deps())

class Dependencies(Dependencies):
    def repr_var(self, var: Node) -> str:
        name, location = var
        func, lineno = location
        return f"({repr(name)}, ({func.__name__}, {lineno}))"

    def repr_deps(self, var_set: Set[Node]) -> str:
        if len(var_set) == 0:
            return "set()"

        return ("{" +
                ", ".join(f"{self.repr_var(var)}"
                          for var in var_set) +
                "}")

    def repr_dependencies(self, vars: Dependency) -> str:
        return ("{\n        " +
                ",\n        ".join(
                    f"{self.repr_var(var)}: {self.repr_deps(vars[var])}"
                    for var in vars) +
                "}")

    def __repr__(self) -> str:
        """Represent dependencies as a Python expression"""
        # Useful for saving and restoring values
        return (f"Dependencies(\n" +
                f"    data={self.repr_dependencies(self.data)},\n" +
                f" control={self.repr_dependencies(self.control)})")

if __name__ == '__main__':
    print(repr(middle_deps()))

class Dependencies(Dependencies):
    def code(self, *items: Callable, mode: str = 'cd') -> None:
        """
        List `items` on standard output, including dependencies as comments. 
        If `items` is empty, all included functions are listed.
        `mode` can contain 'c' (draw control dependencies) and 'd' (draw data dependencies)
        (default: 'cd').
        """

        if len(items) == 0:
            items = cast(Tuple[Callable], self.all_functions().keys())

        for i, item in enumerate(items):
            if i > 0:
                print()
            self._code(item, mode)

    def _code(self, item: Callable, mode: str) -> None:
        # The functions in dependencies may be (instrumented) copies
        # of the original function. Find the function with the same name.
        func = item
        for fn in self.all_functions():
            if fn == item or fn.__name__ == item.__name__:
                func = fn
                break

        all_vars = self.all_vars()
        slice_locations = set(location for (name, location) in all_vars)

        source_lines, first_lineno = inspect.getsourcelines(func)

        n = first_lineno
        for line in source_lines:
            line_location = (func, n)
            if line_location in slice_locations:
                prefix = "* "
            else:
                prefix = "  "

            print(f"{prefix}{n:4} ", end="")

            comment = ""
            for (mode_control, source, arrow) in [
                ('d', self.data, '<='),
                ('c', self.control, '<-')
            ]:
                if mode_control not in mode:
                    continue

                deps = ""
                for var in source:
                    name, location = var
                    if location == line_location:
                        for dep_var in source[var]:
                            if deps == "":
                                deps = arrow + " "
                            else:
                                deps += ", "
                            deps += self.format_var(dep_var, item)

                if deps != "":
                    if comment != "":
                        comment += "; "
                    comment += deps

            if comment != "":
                line = line.rstrip() + "  # " + comment

            print_content(line.rstrip(), '.py')
            print()
            n += 1

#### End of Excursion

if __name__ == '__main__':
    print('\n#### End of Excursion')



if __name__ == '__main__':
    middle_deps().code()  # type: ignore

if __name__ == '__main__':
    quiz("Which of the following `middle()` code lines should be fixed?",
        [
            "Line 2: `if y < z:`",
            "Line 3: `if x < y:`",
            "Line 5: `elif x < z:`",
            "Line 6: `return z`",
        ], '(1 ** 0 + 1 ** 1) ** (1 ** 2 + 1 ** 3)')

## Slices
## ------

if __name__ == '__main__':
    print('\n## Slices')



from .Intro_Debugging import remove_html_markup

if __name__ == '__main__':
    print_content(inspect.getsource(remove_html_markup), '.py')

if __name__ == '__main__':
    remove_html_markup('<foo>bar</foo>')

def remove_html_markup_deps() -> Dependencies:
    return Dependencies({('s', (remove_html_markup, 136)): set(), ('tag', (remove_html_markup, 137)): set(), ('quote', (remove_html_markup, 138)): set(), ('out', (remove_html_markup, 139)): set(), ('c', (remove_html_markup, 141)): {('s', (remove_html_markup, 136))}, ('<test>', (remove_html_markup, 144)): {('quote', (remove_html_markup, 138)), ('c', (remove_html_markup, 141))}, ('tag', (remove_html_markup, 145)): set(), ('<test>', (remove_html_markup, 146)): {('quote', (remove_html_markup, 138)), ('c', (remove_html_markup, 141))}, ('<test>', (remove_html_markup, 148)): {('c', (remove_html_markup, 141))}, ('<test>', (remove_html_markup, 150)): {('tag', (remove_html_markup, 147)), ('tag', (remove_html_markup, 145))}, ('tag', (remove_html_markup, 147)): set(), ('out', (remove_html_markup, 151)): {('out', (remove_html_markup, 151)), ('c', (remove_html_markup, 141)), ('out', (remove_html_markup, 139))}, ('<remove_html_markup() return value>', (remove_html_markup, 153)): {('<test>', (remove_html_markup, 146)), ('out', (remove_html_markup, 151))}}, {('s', (remove_html_markup, 136)): set(), ('tag', (remove_html_markup, 137)): set(), ('quote', (remove_html_markup, 138)): set(), ('out', (remove_html_markup, 139)): set(), ('c', (remove_html_markup, 141)): set(), ('<test>', (remove_html_markup, 144)): set(), ('tag', (remove_html_markup, 145)): {('<test>', (remove_html_markup, 144))}, ('<test>', (remove_html_markup, 146)): {('<test>', (remove_html_markup, 144))}, ('<test>', (remove_html_markup, 148)): {('<test>', (remove_html_markup, 146))}, ('<test>', (remove_html_markup, 150)): {('<test>', (remove_html_markup, 148))}, ('tag', (remove_html_markup, 147)): {('<test>', (remove_html_markup, 146))}, ('out', (remove_html_markup, 151)): {('<test>', (remove_html_markup, 150))}, ('<remove_html_markup() return value>', (remove_html_markup, 153)): set()})

if __name__ == '__main__':
    remove_html_markup_deps().graph()

if __name__ == '__main__':
    quiz("Why does the first line `tag = False` not influence anything?",
        [
            "Because the input contains only tags",
            "Because `tag` is set to True with the first character",
            "Because `tag` is not read by any variable",
            "Because the input contains no tags",
        ], '(1 << 1 + 1 >> 1)')

if __name__ == '__main__':
    tag_deps = Dependencies({('tag', (remove_html_markup, 145)): set(), ('<test>', (remove_html_markup, 144)): {('quote', (remove_html_markup, 138)), ('c', (remove_html_markup, 141))}, ('quote', (remove_html_markup, 138)): set(), ('c', (remove_html_markup, 141)): {('s', (remove_html_markup, 136))}, ('s', (remove_html_markup, 136)): set()}, {('tag', (remove_html_markup, 145)): {('<test>', (remove_html_markup, 144))}, ('<test>', (remove_html_markup, 144)): set(), ('quote', (remove_html_markup, 138)): set(), ('c', (remove_html_markup, 141)): set(), ('s', (remove_html_markup, 136)): set()})
    tag_deps

if __name__ == '__main__':
    tag_deps.code()

if __name__ == '__main__':
    quiz("How does the slice of `tag = True` change "
         "for a different value of `s`?",
        [
            "Not at all",
            "If `s` contains a quote, the `quote` slice is included, too",
            "If `s` contains no HTML tag, the slice will be empty"
        ], '[1, 2, 3][1:]')

## Tracking Techniques
## -------------------

if __name__ == '__main__':
    print('\n## Tracking Techniques')



### Wrapping Data Objects

if __name__ == '__main__':
    print('\n### Wrapping Data Objects')



class MyInt(int):
    def __new__(cls: Type, value: Any, *args: Any, **kwargs: Any) -> Any:
        return super(cls, cls).__new__(cls, value)

    def __repr__(self) -> str:
        return f"{int(self)}"

if __name__ == '__main__':
    n: MyInt = MyInt(5)

if __name__ == '__main__':
    n, n + 1

if __name__ == '__main__':
    n.origin = "Line 5"  # type: ignore

if __name__ == '__main__':
    n.origin  # type: ignore

### Wrapping Data Accesses

if __name__ == '__main__':
    print('\n### Wrapping Data Accesses')



## A Data Tracker
## --------------

if __name__ == '__main__':
    print('\n## A Data Tracker')



class DataTracker(StackInspector):
    """Track data accesses during execution"""

    def __init__(self, log: bool = False) -> None:
        """Constructor. If `log` is set, turn on logging."""
        self.log = log

class DataTracker(DataTracker):
    def set(self, name: str, value: Any, loads: Optional[Set[str]] = None) -> Any:
        """Track setting `name` to `value`."""
        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: setting {name}")

        return value

class DataTracker(DataTracker):
    def get(self, name: str, value: Any) -> Any:
        """Track getting `value` from `name`."""

        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: getting {name}")

        return value

if __name__ == '__main__':
    _test_data = DataTracker(log=True)
    x = _test_data.set('x', 1)

if __name__ == '__main__':
    _test_data.get('x', x)

## Instrumenting Source Code
## -------------------------

if __name__ == '__main__':
    print('\n## Instrumenting Source Code')



import ast

from .bookutils import show_ast

if __name__ == '__main__':
    middle_tree = ast.parse(inspect.getsource(middle))
    show_ast(middle_tree)

### Tracking Variable Accesses

if __name__ == '__main__':
    print('\n### Tracking Variable Accesses')



from ast import NodeTransformer, NodeVisitor, Name, AST

import typing

DATA_TRACKER = '_data'

def is_internal(id: str) -> bool:
    """Return True if `id` is a built-in function or type"""
    return (id in dir(__builtins__) or id in dir(typing))

if __name__ == '__main__':
    assert is_internal('int')
    assert is_internal('None')
    assert is_internal('Tuple')

class TrackGetTransformer(NodeTransformer):
    def visit_Name(self, node: Name) -> AST:
        self.generic_visit(node)

        if is_internal(node.id):
            # Do not change built-in names and types
            return node

        if node.id == DATA_TRACKER:
            # Do not change own accesses
            return node

        if not isinstance(node.ctx, Load):
            # Only change loads (not stores, not deletions)
            return node

        new_node = make_get_data(node.id)
        ast.copy_location(new_node, node)
        return new_node

from ast import Module, Load, Store, \
    Attribute, With, withitem, keyword, Call, Expr, \
    Assign, AugAssign, AnnAssign, Assert

def make_get_data(id: str, method: str = 'get') -> Call:
    return Call(func=Attribute(value=Name(id=DATA_TRACKER, ctx=Load()), 
                               attr=method, ctx=Load()),
                args=[ast.Str(id), Name(id=id, ctx=Load())],
                keywords=[])

if __name__ == '__main__':
    show_ast(Module(body=[make_get_data("x")], type_ignores=[]))  # type: ignore

if __name__ == '__main__':
    print(ast.dump(ast.parse("_data.get('x', x)")))

if __name__ == '__main__':
    TrackGetTransformer().visit(middle_tree);

def dump_tree(tree: AST) -> None:
    print_content(ast.unparse(tree), '.py')
    ast.fix_missing_locations(tree)  # Must run this before compiling
    _ = compile(cast(ast.Module, tree), '<dump_tree>', 'exec')

if __name__ == '__main__':
    dump_tree(middle_tree)

from types import TracebackType

class DataTrackerTester:
    def __init__(self, tree: AST, func: Callable, log: bool = True) -> None:
        """Constructor. Execute the code in `tree` while instrumenting `func`."""
        # We pass the source file of `func` such that we can retrieve it
        # when accessing the location of the new compiled code
        source = cast(str, inspect.getsourcefile(func))
        self.code = compile(cast(ast.Module, tree), source, 'exec')
        self.func = func
        self.log = log

    def make_data_tracker(self) -> Any:
        return DataTracker(log=self.log)

    def __enter__(self) -> Any:
        """Rewrite function"""
        tracker = self.make_data_tracker()
        globals()[DATA_TRACKER] = tracker
        exec(self.code, globals())
        return tracker

    def __exit__(self, exc_type: Type, exc_value: BaseException,
                 traceback: TracebackType) -> Optional[bool]:
        """Restore function"""
        globals()[self.func.__name__] = self.func
        del globals()[DATA_TRACKER]
        return None

if __name__ == '__main__':
    print_content(inspect.getsource(middle), '.py', start_line_number=1)

if __name__ == '__main__':
    with DataTrackerTester(middle_tree, middle):
        middle(2, 1, 3)

if __name__ == '__main__':
    middle(2, 1, 3)

### Excursion: Tracking Assignments and Assertions

if __name__ == '__main__':
    print('\n### Excursion: Tracking Assignments and Assertions')



if __name__ == '__main__':
    print(ast.dump(ast.parse("_data.set('x', value, loads=(a, b))")))

def make_set_data(id: str, value: Any, 
                  loads: Optional[Set[str]] = None, method: str = 'set') -> Call:
    """
    Construct a subtree _data.`method`('`id`', `value`). 
    If `loads` is set to [X1, X2, ...], make it
    _data.`method`('`id`', `value`, loads=(X1, X2, ...))
    """

    keywords=[]

    if loads:
        keywords = [
            keyword(arg='loads',
                    value=ast.Tuple(
                        elts=[Name(id=load, ctx=Load()) for load in loads],
                        ctx=Load()
                    ))
        ]

    new_node = Call(func=Attribute(value=Name(id=DATA_TRACKER, ctx=Load()),
                                   attr=method, ctx=Load()),
                    args=[ast.Str(id), value],
                    keywords=keywords)

    ast.copy_location(new_node, value)

    return new_node

class LeftmostNameVisitor(NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.leftmost_name: Optional[str] = None

    def visit_Name(self, node: Name) -> None:
        if self.leftmost_name is None:
            self.leftmost_name = node.id
        self.generic_visit(node)

def leftmost_name(tree: AST) -> Optional[str]:
    visitor = LeftmostNameVisitor()
    visitor.visit(tree)
    return visitor.leftmost_name

if __name__ == '__main__':
    leftmost_name(ast.parse('a[x] = 25'))

class StoreVisitor(NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.names: Set[str] = set()

    def visit(self, node: AST) -> None:
        if hasattr(node, 'ctx') and isinstance(node.ctx, Store):  # type: ignore
            name = leftmost_name(node)
            if name:
                self.names.add(name)

        self.generic_visit(node)

def store_names(tree: AST) -> Set[str]:
    visitor = StoreVisitor()
    visitor.visit(tree)
    return visitor.names

if __name__ == '__main__':
    store_names(ast.parse('a[x], b[y], c = 1, 2, 3'))

class LoadVisitor(NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.names: Set[str] = set()

    def visit(self, node: AST) -> None:
        if hasattr(node, 'ctx') and isinstance(node.ctx, Load):  # type: ignore
            name = leftmost_name(node)
            if name is not None:
                self.names.add(name)

        self.generic_visit(node)

def load_names(tree: AST) -> Set[str]:
    visitor = LoadVisitor()
    visitor.visit(tree)
    return visitor.names

if __name__ == '__main__':
    load_names(ast.parse('a[x], b[y], c = 1, 2, 3'))

class TrackSetTransformer(NodeTransformer):
    def visit_Assign(self, node: Assign) -> Assign:
        value = ast.unparse(node.value)
        if value.startswith(DATA_TRACKER + '.set'):
            return node  # Do not apply twice

        for target in node.targets:
            loads = load_names(target)
            for store_name in store_names(target):
                node.value = make_set_data(store_name, node.value, 
                                           loads=loads)
                loads = set()

        return node

class TrackSetTransformer(TrackSetTransformer):
    def visit_AugAssign(self, node: AugAssign) -> AugAssign:
        value = ast.unparse(node.value)
        if value.startswith(DATA_TRACKER):
            return node  # Do not apply twice

        id = cast(str, leftmost_name(node.target))
        node.value = make_set_data(id, node.value, method='augment')

        return node

class DataTracker(DataTracker):
    def augment(self, name: str, value: Any) -> Any:
        """
        Track augmenting `name` with `value`.
        To be overloaded in subclasses.
        """
        self.set(name, self.get(name, value))
        return value

class TrackSetTransformer(TrackSetTransformer):
    def visit_AnnAssign(self, node: AnnAssign) -> AnnAssign:
        if node.value is None:
            return node  # just <var>: <type> without value

        value = ast.unparse(node.value)
        if value.startswith(DATA_TRACKER + '.set'):
            return node  # Do not apply twice

        loads = load_names(node.target)
        for store_name in store_names(node.target):
            node.value = make_set_data(store_name, node.value, 
                                       loads=loads)
            loads = set()

        return node

class TrackSetTransformer(TrackSetTransformer):
    def visit_Assert(self, node: Assert) -> Assert:
        value = ast.unparse(node.test)
        if value.startswith(DATA_TRACKER + '.set'):
            return node  # Do not apply twice

        loads = load_names(node.test)
        node.test = make_set_data("<assertion>", node.test, loads=loads)
        return node

def assign_test(x: int) -> Tuple[int, str]:  # type: ignore
    forty_two: int = 42
    forty_two = forty_two = 42
    a, b = 1, 2
    c = d = [a, b]
    c[d[a]].attr = 47  # type: ignore
    a *= b + 1
    assert a > 0
    return forty_two, "Forty-Two"

if __name__ == '__main__':
    assign_tree = ast.parse(inspect.getsource(assign_test))

if __name__ == '__main__':
    TrackSetTransformer().visit(assign_tree)
    dump_tree(assign_tree)

if __name__ == '__main__':
    TrackGetTransformer().visit(assign_tree)
    dump_tree(assign_tree)

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



### Excursion: Tracking Return Values

if __name__ == '__main__':
    print('\n### Excursion: Tracking Return Values')



class TrackReturnTransformer(NodeTransformer):
    def __init__(self) -> None:
        self.function_name: Optional[str] = None
        super().__init__()

    def visit_FunctionDef(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> AST:
        outer_name = self.function_name
        self.function_name = node.name  # Save current name
        self.generic_visit(node)
        self.function_name = outer_name
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> AST:
        return self.visit_FunctionDef(node)

    def return_value(self, tp: str = "return") -> str:
        if self.function_name is None:
            return f"<{tp} value>"
        else:
            return f"<{self.function_name}() {tp} value>"

    def visit_return_or_yield(self, node: Union[ast.Return, ast.Yield, ast.YieldFrom],
                              tp: str = "return") -> AST:

        if node.value is not None:
            value = ast.unparse(node.value)
            if not value.startswith(DATA_TRACKER + '.set'):
                node.value = make_set_data(self.return_value(tp), node.value)

        return node

    def visit_Return(self, node: ast.Return) -> AST:
        return self.visit_return_or_yield(node, tp="return")

    def visit_Yield(self, node: ast.Yield) -> AST:
        return self.visit_return_or_yield(node, tp="yield")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> AST:
        return self.visit_return_or_yield(node, tp="yield")

if __name__ == '__main__':
    TrackReturnTransformer().visit(middle_tree)
    dump_tree(middle_tree)

if __name__ == '__main__':
    with DataTrackerTester(middle_tree, middle):
        middle(2, 1, 3)

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



### Excursion: Tracking Control

if __name__ == '__main__':
    print('\n### Excursion: Tracking Control')



class TrackControlTransformer(NodeTransformer):
    def visit_If(self, node: ast.If) -> ast.If:
        self.generic_visit(node)
        node.test = self.make_test(node.test)
        node.body = self.make_with(node.body)
        node.orelse = self.make_with(node.orelse)
        return node

class TrackControlTransformer(TrackControlTransformer):
    def make_with(self, block: List[ast.stmt]) -> List[ast.stmt]:
        """Create a subtree 'with _data: `block`'"""
        if len(block) == 0:
            return []

        block_as_text = ast.unparse(block[0])
        if block_as_text.startswith('with ' + DATA_TRACKER):
            return block  # Do not apply twice

        new_node = With(
            items=[
                withitem(
                    context_expr=Name(id=DATA_TRACKER, ctx=Load()),
                    optional_vars=None)
            ],
            body=block
        )
        ast.copy_location(new_node, block[0])
        return [new_node]

class TrackControlTransformer(TrackControlTransformer):
    def make_test(self, test: ast.expr) -> ast.expr:
        test_as_text = ast.unparse(test)
        if test_as_text.startswith(DATA_TRACKER + '.test'):
            return test  # Do not apply twice

        new_test = Call(func=Attribute(value=Name(id=DATA_TRACKER, ctx=Load()),
                                       attr='test',
                                       ctx=Load()),
                         args=[test],
                         keywords=[])
        ast.copy_location(new_test, test)
        return new_test

class TrackControlTransformer(TrackControlTransformer):
    def visit_While(self, node: ast.While) -> ast.While:
        self.generic_visit(node)
        node.test = self.make_test(node.test)
        node.body = self.make_with(node.body)
        node.orelse = self.make_with(node.orelse)
        return node

class TrackControlTransformer(TrackControlTransformer):
    # regular `for` loop
    def visit_For(self, node: Union[ast.For, ast.AsyncFor]) -> AST:
        self.generic_visit(node)
        id = ast.unparse(node.target).strip()
        node.iter = make_set_data(id, node.iter)

        # Uncomment if you want iterators to control their bodies
        # node.body = self.make_with(node.body)
        # node.orelse = self.make_with(node.orelse)
        return node

    # `for` loops in async functions
    def visit_AsyncFor(self, node: ast.AsyncFor) -> AST:
        return self.visit_For(node)

    # `for` clause in comprehensions
    def visit_comprehension(self, node: ast.comprehension) -> AST:
        self.generic_visit(node)
        id = ast.unparse(node.target).strip()
        node.iter = make_set_data(id, node.iter)
        return node

if __name__ == '__main__':
    TrackControlTransformer().visit(middle_tree)
    dump_tree(middle_tree)

class DataTracker(DataTracker):
    def test(self, cond: AST) -> AST:
        """Test condition `cond`. To be overloaded in subclasses."""
        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: testing condition")

        return cond

class DataTracker(DataTracker):
    def __enter__(self) -> Any:
        """Enter `with` block. To be overloaded in subclasses."""
        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: entering block")
        return self

    def __exit__(self, exc_type: Type, exc_value: BaseException, 
                 traceback: TracebackType) -> Optional[bool]:
        """Exit `with` block. To be overloaded in subclasses."""
        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: exiting block")
        return None

if __name__ == '__main__':
    with DataTrackerTester(middle_tree, middle):
        middle(2, 1, 3)

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



### Excursion: Tracking Calls and Arguments

if __name__ == '__main__':
    print('\n### Excursion: Tracking Calls and Arguments')



class TrackCallTransformer(NodeTransformer):
    def make_call(self, node: AST, func: str, 
                  pos: Optional[int] = None, kw: Optional[str] = None) -> Call:
        """Return _data.call(`func`)(`node`)"""
        keywords = []

        # `Num()` and `Str()` are deprecated in favor of `Constant()`
        if pos:
            keywords.append(keyword(arg='pos', value=ast.Num(pos)))
        if kw:
            keywords.append(keyword(arg='kw', value=ast.Str(kw)))

        return Call(func=Attribute(value=Name(id=DATA_TRACKER,
                                              ctx=Load()),
                                   attr=func,
                                   ctx=Load()),
                     args=[node],  # type: ignore
                     keywords=keywords)

    def visit_Call(self, node: Call) -> Call:
        self.generic_visit(node)

        call_as_text = ast.unparse(node)
        if call_as_text.startswith(DATA_TRACKER + '.ret'):
            return node  # Already applied

        func_as_text = ast.unparse(node)
        if func_as_text.startswith(DATA_TRACKER + '.'):
            return node  # Own function

        new_args = []
        for n, arg in enumerate(node.args):
            new_args.append(self.make_call(arg, 'arg', pos=n + 1))
        node.args = cast(List[ast.expr], new_args)

        for kw in node.keywords:
            id = kw.arg if hasattr(kw, 'arg') else None
            kw.value = self.make_call(kw.value, 'arg', kw=id)

        node.func = self.make_call(node.func, 'call')
        return self.make_call(node, 'ret')

def test_call() -> int:
    x = middle(1, 2, z=middle(1, 2, 3))
    return x

if __name__ == '__main__':
    call_tree = ast.parse(inspect.getsource(test_call))
    dump_tree(call_tree)

if __name__ == '__main__':
    TrackCallTransformer().visit(call_tree);

if __name__ == '__main__':
    dump_tree(call_tree)

def f() -> bool:
    return math.isclose(1, 1.0)

if __name__ == '__main__':
    f_tree = ast.parse(inspect.getsource(f))
    dump_tree(f_tree)

if __name__ == '__main__':
    TrackCallTransformer().visit(f_tree);

if __name__ == '__main__':
    dump_tree(f_tree)

class DataTracker(DataTracker):
    def arg(self, value: Any, pos: Optional[int] = None, kw: Optional[str] = None) -> Any:
        """
        Track `value` being passed as argument.
        `pos` (if given) is the argument position (starting with 1).
        `kw` (if given) is the argument keyword.
        """

        if self.log:
            caller_func, lineno = self.caller_location()
            info = ""
            if pos:
                info += f" #{pos}"
            if kw:
                info += f" {repr(kw)}"

            print(f"{caller_func.__name__}:{lineno}: pushing arg{info}")

        return value

class DataTracker(DataTracker):
    def ret(self, value: Any) -> Any:
        """Track `value` being used as return value."""
        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: returned from call")

        return value

class DataTracker(DataTracker):
    def instrument_call(self, func: Callable) -> Callable:
        """Instrument a call to `func`. To be implemented in subclasses."""
        return func

    def call(self, func: Callable) -> Callable:
        """Track a call to `func`."""
        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: calling {func}")

        return self.instrument_call(func)

if __name__ == '__main__':
    dump_tree(call_tree)

if __name__ == '__main__':
    with DataTrackerTester(call_tree, test_call):
        test_call()

if __name__ == '__main__':
    test_call()

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



### Excursion: Tracking Parameters

if __name__ == '__main__':
    print('\n### Excursion: Tracking Parameters')



if __name__ == '__main__':
    print(ast.dump(ast.parse("_data.param('x', x, pos=1, last=True)")))

class TrackParamsTransformer(NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)

        named_args = []
        for child in ast.iter_child_nodes(node.args):
            if isinstance(child, ast.arg):
                named_args.append(child)

        create_stmts = []
        for n, child in enumerate(named_args):
            keywords=[keyword(arg='pos', value=ast.Num(n + 1))]
            if child is node.args.vararg:
                keywords.append(keyword(arg='vararg', value=ast.Str('*')))
            if child is node.args.kwarg:
                keywords.append(keyword(arg='vararg', value=ast.Str('**')))
            if n == len(named_args) - 1:
                keywords.append(keyword(arg='last',
                                        value=ast.NameConstant(value=True)))

            create_stmt = Expr(
                value=Call(
                    func=Attribute(value=Name(id=DATA_TRACKER, ctx=Load()),
                                   attr='param', ctx=Load()),
                    args=[ast.Str(child.arg),
                          Name(id=child.arg, ctx=Load())
                         ],
                    keywords=keywords
                )
            )
            ast.copy_location(create_stmt, node)
            create_stmts.append(create_stmt)

        node.body = cast(List[ast.stmt], create_stmts) + node.body
        return node

if __name__ == '__main__':
    TrackParamsTransformer().visit(middle_tree)
    dump_tree(middle_tree)

class DataTracker(DataTracker):
    def param(self, name: str, value: Any, 
              pos: Optional[int] = None, vararg: str = '', last: bool = False) -> Any:
        """
        At the beginning of a function, track parameter `name` being set to `value`.
        `pos` is the position of the argument (starting with 1).
        `vararg` is "*" if `name` is a vararg parameter (as in *args),
        and "**" is `name` is a kwargs parameter (as in *kwargs).
        `last` is True if `name` is the last parameter.
        """
        if self.log:
            caller_func, lineno = self.caller_location()
            info = ""
            if pos is not None:
                info += f" #{pos}"

            print(f"{caller_func.__name__}:{lineno}: initializing {vararg}{name}{info}")

        return self.set(name, value)

if __name__ == '__main__':
    with DataTrackerTester(middle_tree, middle):
        middle(2, 1, 3)

def args_test(x, *args, **kwargs):  # type: ignore
    print(x, *args, **kwargs)

if __name__ == '__main__':
    args_tree = ast.parse(inspect.getsource(args_test))
    TrackParamsTransformer().visit(args_tree)
    dump_tree(args_tree)

if __name__ == '__main__':
    with DataTrackerTester(args_tree, args_test):
        args_test(1, 2, 3)

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



if __name__ == '__main__':
    dump_tree(middle_tree)

if __name__ == '__main__':
    with DataTrackerTester(middle_tree, middle):
        m = middle(2, 1, 3)
    m

### Excursion: Transformer Stress Test

if __name__ == '__main__':
    print('\n### Excursion: Transformer Stress Test')



from . import Assertions  # minor dependency
from . import Debugger  # minor dependency

if __name__ == '__main__':
    for module in [Assertions, Debugger, inspect, ast]:
        module_tree = ast.parse(inspect.getsource(module))
        assert isinstance(module_tree, ast.Module)

        TrackCallTransformer().visit(module_tree)
        TrackSetTransformer().visit(module_tree)
        TrackGetTransformer().visit(module_tree)
        TrackControlTransformer().visit(module_tree)
        TrackReturnTransformer().visit(module_tree)
        TrackParamsTransformer().visit(module_tree)
        # dump_tree(module_tree)
        ast.fix_missing_locations(module_tree)  # Must run this before compiling

        module_code = compile(module_tree, '<stress_test>', 'exec')
        print(f"{repr(module.__name__)} instrumented successfully.")

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



## Tracking Dependencies
## ---------------------

if __name__ == '__main__':
    print('\n## Tracking Dependencies')



class DependencyTracker(DataTracker):
    """Track dependencies during execution"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Constructor. Arguments are passed to DataTracker.__init__()"""
        super().__init__(*args, **kwargs)

        self.origins: Dict[str, Location] = {}  # Where current variables were last set
        self.data_dependencies: Dependency = {}  # As with Dependencies, above
        self.control_dependencies: Dependency = {}

        self.last_read: List[str] = []  # List of last read variables
        self.last_checked_location = (StackInspector.unknown, 1)
        self._ignore_location_change = False

        self.data: List[List[str]] = [[]]  # Data stack
        self.control: List[List[str]] = [[]]  # Control stack

        self.frames: List[Dict[Union[int, str], Any]] = [{}]  # Argument stack
        self.args: Dict[Union[int, str], Any] = {}  # Current args

### Data Dependencies

if __name__ == '__main__':
    print('\n### Data Dependencies')



#### Reading Variables

if __name__ == '__main__':
    print('\n#### Reading Variables')



class DependencyTracker(DependencyTracker):
    def get(self, name: str, value: Any) -> Any:
        """Track a read access for variable `name` with value `value`"""
        self.check_location()
        self.last_read.append(name)
        return super().get(name, value)

    def check_location(self) -> None:
        pass  # More on that below

if __name__ == '__main__':
    x = 5
    y = 3

if __name__ == '__main__':
    _test_data = DependencyTracker(log=True)
    _test_data.get('x', x) + _test_data.get('y', y)

if __name__ == '__main__':
    _test_data.last_read

#### Checking Locations

if __name__ == '__main__':
    print('\n#### Checking Locations')



class DependencyTracker(DependencyTracker):
    def clear_read(self) -> None:
        """Clear set of read variables"""
        if self.log:
            direct_caller = inspect.currentframe().f_back.f_code.co_name  # type: ignore
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: "
                  f"clearing read variables {self.last_read} "
                  f"(from {direct_caller})")

        self.last_read = []

    def check_location(self) -> None:
        """If we are in a new location, clear set of read variables"""
        location = self.caller_location()
        func, lineno = location
        last_func, last_lineno = self.last_checked_location

        if self.last_checked_location != location:
            if self._ignore_location_change:
                self._ignore_location_change = False
            elif func.__name__.startswith('<'):
                # Entering list comprehension, eval(), exec(), ...
                pass
            elif last_func.__name__.startswith('<'):
                # Exiting list comprehension, eval(), exec(), ...
                pass
            else:
                # Standard case
                self.clear_read()

        self.last_checked_location = location

class DependencyTracker(DependencyTracker):
    def ignore_next_location_change(self) -> None:
        self._ignore_location_change = True

    def ignore_location_change(self) -> None:
        self.last_checked_location = self.caller_location()

if __name__ == '__main__':
    _test_data = DependencyTracker()

if __name__ == '__main__':
    _test_data.get('x', x) + _test_data.get('y', y)

if __name__ == '__main__':
    _test_data.last_read

if __name__ == '__main__':
    a = 42
    b = -1
    _test_data.get('a', a) + _test_data.get('b', b)

if __name__ == '__main__':
    _test_data.last_read

#### Setting Variables

if __name__ == '__main__':
    print('\n#### Setting Variables')



import itertools

class DependencyTracker(DependencyTracker):
    TEST = '<test>'  # Name of pseudo-variables for testing conditions

    def set(self, name: str, value: Any, loads: Optional[Set[str]] = None) -> Any:
        """Add a dependency for `name` = `value`"""

        def add_dependencies(dependencies: Set[Node], 
                             vars_read: List[str], tp: str) -> None:
            """Add origins of `vars_read` to `dependencies`."""
            for var_read in vars_read:
                if var_read in self.origins:
                    if var_read == self.TEST and tp == "data":
                        # Can't have data dependencies on conditions
                        continue

                    origin = self.origins[var_read]
                    dependencies.add((var_read, origin))

                    if self.log:
                        origin_func, origin_lineno = origin
                        caller_func, lineno = self.caller_location()
                        print(f"{caller_func.__name__}:{lineno}: "
                              f"new {tp} dependency: "
                              f"{name} <= {var_read} "
                              f"({origin_func.__name__}:{origin_lineno})")

        self.check_location()
        ret = super().set(name, value)
        location = self.caller_location()

        add_dependencies(self.data_dependencies.setdefault
                         ((name, location), set()),
                         self.last_read, tp="data")
        add_dependencies(self.control_dependencies.setdefault
                         ((name, location), set()),
                         cast(List[str], itertools.chain.from_iterable(self.control)),
                         tp="control")

        self.origins[name] = location

        # Reset read info for next line
        self.last_read = [name]

        # Next line is a new location
        self._ignore_location_change = False

        return ret

    def dependencies(self) -> Dependencies:
        """Return dependencies"""
        return Dependencies(self.data_dependencies,
                            self.control_dependencies)

if __name__ == '__main__':
    _test_data = DependencyTracker()
    x = _test_data.set('x', 1)
    y = _test_data.set('y', _test_data.get('x', x))
    z = _test_data.set('z', _test_data.get('x', x) + _test_data.get('y', y))

if __name__ == '__main__':
    _test_data.origins

if __name__ == '__main__':
    _test_data.data_dependencies

if __name__ == '__main__':
    _test_data.dependencies().graph()

### Excursion: Control Dependencies

if __name__ == '__main__':
    print('\n### Excursion: Control Dependencies')



class DependencyTracker(DependencyTracker):
    def test(self, value: Any) -> Any:
        """Track a test for condition `value`"""
        self.set(self.TEST, value)
        return super().test(value)

class DependencyTracker(DependencyTracker):
    def __enter__(self) -> Any:
        """Track entering an if/while/for block"""
        self.control.append(self.last_read)
        self.clear_read()
        return super().__enter__()

class DependencyTracker(DependencyTracker):
    def __exit__(self, exc_type: Type, exc_value: BaseException,
                 traceback: TracebackType) -> Optional[bool]:
        """Track exiting an if/while/for block"""
        self.clear_read()
        self.last_read = self.control.pop()
        self.ignore_next_location_change()
        return super().__exit__(exc_type, exc_value, traceback)

if __name__ == '__main__':
    _test_data = DependencyTracker()
    x = _test_data.set('x', 1)
    y = _test_data.set('y', _test_data.get('x', x))

if __name__ == '__main__':
    if _test_data.test(_test_data.get('x', x) >= _test_data.get('y', y)):
        with _test_data:
            z = _test_data.set('z',
                               _test_data.get('x', x) + _test_data.get('y', y))

if __name__ == '__main__':
    _test_data.control_dependencies

if __name__ == '__main__':
    _test_data.dependencies()

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



### Excursion: Calls and Returns

if __name__ == '__main__':
    print('\n### Excursion: Calls and Returns')



import copy

class DependencyTracker(DependencyTracker):
    def call(self, func: Callable) -> Callable:
        """Track a call of function `func`"""
        func = super().call(func)

        if inspect.isgeneratorfunction(func):
            return self.call_generator(func)

        # Save context
        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: "
                  f"saving read variables {self.last_read}")

        self.data.append(self.last_read)
        self.clear_read()
        self.ignore_next_location_change()

        self.frames.append(self.args)
        self.args = {}

        return func

class DependencyTracker(DependencyTracker):
    def ret(self, value: Any) -> Any:
        """Track a function return"""
        value = super().ret(value)

        if self.in_generator():
            return self.ret_generator(value)

        # Restore old context and add return value
        ret_name = None
        for var in self.last_read:
            if var.startswith("<"):  # "<return value>"
                ret_name = var

        self.last_read = self.data.pop()
        if ret_name:
            self.last_read.append(ret_name)

        if self.args:
            # We return from an uninstrumented function:
            # Make return value depend on all args
            for key, deps in self.args.items():
                self.last_read += deps

        self.ignore_location_change()

        self.args = self.frames.pop()

        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: "
                  f"restored read variables {self.last_read}")

        return value

class DependencyTracker(DependencyTracker):
    def in_generator(self) -> bool:
        """True if we are calling a generator function"""
        return len(self.data) > 0 and self.data[-1] is None

    def call_generator(self, func: Callable) -> Callable:
        """Track a call of a generator function"""
        # Mark the fact that we're in a generator with `None` values
        self.data.append(None)  # type: ignore
        self.frames.append(None)  # type: ignore
        assert self.in_generator()

        self.clear_read()
        return func

    def ret_generator(self, generator: Any) -> Any:
        """Track the return of a generator function"""
        # Pop the two 'None' values pushed earlier
        self.data.pop()
        self.frames.pop()

        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: "
                  f"wrapping generator {generator} (args={self.args})")

        # At this point, we already have collected the args.
        # The returned generator depends on all of them.
        for arg in self.args:
            self.last_read += self.args[arg]

        # Wrap the generator such that the args are restored 
        # when it is actually invoked, such that we can map them
        # to parameters.
        saved_args = copy.deepcopy(self.args)

        def wrapper() -> Generator[Any, None, None]:
            self.args = saved_args
            if self.log:
                caller_func, lineno = self.caller_location()
                print(f"{caller_func.__name__}:{lineno}: "
                  f"calling generator (args={self.args})")

            self.ignore_next_location_change()
            yield from generator

        return wrapper()

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



### Excursion: Function Arguments

if __name__ == '__main__':
    print('\n### Excursion: Function Arguments')



class DependencyTracker(DependencyTracker):
    def arg(self, value: Any, pos: Optional[int] = None, kw: Optional[str] = None) -> Any:
        """
        Track passing an argument `value`
        (with given position `pos` 1..n or keyword `kw`)
        """
        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: "
                  f"saving args read {self.last_read}")

        if pos:
            self.args[pos] = self.last_read
        if kw:
            self.args[kw] = self.last_read

        self.clear_read()
        return super().arg(value, pos, kw)

class DependencyTracker(DependencyTracker):
    def param(self, name: str, value: Any,
              pos: Optional[int] = None, vararg: str = "", last: bool = False) -> Any:
        """
        Track getting a parameter `name` with value `value`
        (with given position `pos`).
        vararg parameters are indicated by setting `varargs` to 
        '*' (*args) or '**' (**kwargs)
        """
        self.clear_read()

        if vararg == '*':
            # We over-approximate by setting `args` to _all_ positional args
            for index in self.args:
                if isinstance(index, int) and pos is not None and index >= pos:
                    self.last_read += self.args[index]
        elif vararg == '**':
            # We over-approximate by setting `kwargs` to _all_ passed keyword args
            for index in self.args:
                if isinstance(index, str):
                    self.last_read += self.args[index]
        elif name in self.args:
            self.last_read = self.args[name]
        elif pos in self.args:
            self.last_read = self.args[pos]

        if self.log:
            caller_func, lineno = self.caller_location()
            print(f"{caller_func.__name__}:{lineno}: "
                  f"restored params read {self.last_read}")

        self.ignore_location_change()
        ret = super().param(name, value, pos)

        if last:
            self.clear_read()
            self.args = {}  # Mark `args` as processed

        return ret

def call_test() -> int:
    c = 47

    def sq(n: int) -> int:
        return n * n

    def gen(e: int) -> Generator[int, None, None]:
        yield e * c

    def just_x(x: Any, y: Any) -> Any:
        return x

    a = 42
    b = gen(a)
    d = list(b)[0]

    xs = [1, 2, 3, 4]
    ys = [sq(elem) for elem in xs if elem > 2]

    return just_x(just_x(d, y=b), ys[0])

if __name__ == '__main__':
    call_test()

if __name__ == '__main__':
    call_tree = ast.parse(inspect.getsource(call_test))
    TrackCallTransformer().visit(call_tree)
    TrackSetTransformer().visit(call_tree)
    TrackGetTransformer().visit(call_tree)
    TrackControlTransformer().visit(call_tree)
    TrackReturnTransformer().visit(call_tree)
    TrackParamsTransformer().visit(call_tree)
    dump_tree(call_tree)

class DependencyTrackerTester(DataTrackerTester):
    def make_data_tracker(self) -> DependencyTracker:
        return DependencyTracker(log=self.log)

if __name__ == '__main__':
    with DependencyTrackerTester(call_tree, call_test, log=False) as call_deps:
        call_test()

if __name__ == '__main__':
    call_deps.dependencies()

if __name__ == '__main__':
    call_deps.dependencies().code()

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



### Excursion: Diagnostics

if __name__ == '__main__':
    print('\n### Excursion: Diagnostics')



import re

class Dependencies(Dependencies):
    def validate(self) -> None:
        """Perform a simple syntactic validation of dependencies"""
        super().validate()

        for var in self.all_vars():
            source = self.source(var)
            if not source:
                continue
            if source.startswith('<'):
                continue   # no source

            for dep_var in self.data[var] | self.control[var]:
                dep_name, dep_location = dep_var

                if dep_name == DependencyTracker.TEST:
                    continue  # dependency on <test>

                if dep_name.endswith(' value>'):
                    if source.find('(') < 0:
                        warnings.warn(f"Warning: {self.format_var(var)} "
                                  f"depends on {self.format_var(dep_var)}, "
                                  f"but {repr(source)} does not "
                                  f"seem to have a call")
                    continue

                if source.startswith('def'):
                    continue   # function call

                rx = re.compile(r'\b' + dep_name + r'\b')
                if rx.search(source) is None:
                    warnings.warn(f"{self.format_var(var)} "
                              f"depends on {self.format_var(dep_var)}, "
                              f"but {repr(dep_name)} does not occur "
                              f"in {repr(source)}")

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



## Slicing Code
## ------------

if __name__ == '__main__':
    print('\n## Slicing Code')



### An Instrumenter Base Class

if __name__ == '__main__':
    print('\n### An Instrumenter Base Class')



class Instrumenter(StackInspector):
    """Instrument functions for dynamic tracking"""

    def __init__(self, *items_to_instrument: Callable,
                 globals: Optional[Dict[str, Any]] = None,
                 log: Union[bool, int] = False) -> None:
        """
        Create an instrumenter.
        `items_to_instrument` is a list of items to instrument.
        `globals` is a namespace to use (default: caller's globals())
        """

        self.log = log
        self.items_to_instrument: List[Callable] = list(items_to_instrument)
        self.instrumented_items: Set[Any] = set()

        if globals is None:
            globals = self.caller_globals()
        self.globals = globals

    def __enter__(self) -> Any:
        """Instrument sources"""
        items = self.items_to_instrument
        if not items:
            items = self.default_items_to_instrument()

        for item in items:
            self.instrument(item)

        return self

    def default_items_to_instrument(self) -> List[Callable]:
        return []

    def instrument(self, item: Any) -> Any:
        """Instrument `item`. To be overloaded in subclasses."""
        if self.log:
            print("Instrumenting", item)
        self.instrumented_items.add(item)
        return item

class Instrumenter(Instrumenter):
    def __exit__(self, exc_type: Type, exc_value: BaseException,
                 traceback: TracebackType) -> Optional[bool]:
        """Restore sources"""
        self.restore()
        return None

    def restore(self) -> None:
        for item in self.instrumented_items:
            self.globals[item.__name__] = item

if __name__ == '__main__':
    with Instrumenter(middle, log=True) as ins:
        pass

### The Slicer Class

if __name__ == '__main__':
    print('\n### The Slicer Class')



class Slicer(Instrumenter):
    """Track dependencies in an execution"""

    def __init__(self, *items_to_instrument: Any,
                 dependency_tracker: Optional[DependencyTracker] = None,
                 globals: Optional[Dict[str, Any]] = None,
                 log: Union[bool, int] = False):
        """Create a slicer.
        `items_to_instrument` are Python functions or modules with source code.
        `dependency_tracker` is the tracker to be used (default: DependencyTracker).
        `globals` is the namespace to be used(default: caller's `globals()`)
        `log`=True or `log` > 0 turns on logging
        """
        super().__init__(*items_to_instrument, globals=globals, log=log)

        if dependency_tracker is None:
            dependency_tracker = DependencyTracker(log=(log > 1))
        self.dependency_tracker = dependency_tracker

        self.saved_dependencies = None

    def default_items_to_instrument(self) -> List[Callable]:
        raise ValueError("Need one or more items to instrument")

class Slicer(Slicer):
    def parse(self, item: Any) -> AST:
        """Parse `item`, returning its AST"""
        source_lines, lineno = inspect.getsourcelines(item)
        source = "".join(source_lines)

        if self.log >= 2:
            print_content(source, '.py', start_line_number=lineno)
            print()
            print()

        tree = ast.parse(source)
        ast.increment_lineno(tree, lineno - 1)
        return tree

class Slicer(Slicer):
    def transformers(self) -> List[NodeTransformer]:
        """List of transformers to apply. To be extended in subclasses."""
        return [
            TrackCallTransformer(),
            TrackSetTransformer(),
            TrackGetTransformer(),
            TrackControlTransformer(),
            TrackReturnTransformer(),
            TrackParamsTransformer()
        ]

    def transform(self, tree: AST) -> AST:
        """Apply transformers on `tree`. May be extended in subclasses."""
        # Apply transformers
        for transformer in self.transformers():
            if self.log >= 3:
                print(transformer.__class__.__name__ + ':')

            transformer.visit(tree)
            ast.fix_missing_locations(tree)
            if self.log >= 3:
                print_content(ast.unparse(tree), '.py')
                print()
                print()

        if 0 < self.log < 3:
            print_content(ast.unparse(tree), '.py')
            print()
            print()

        return tree

class Slicer(Slicer):
    def execute(self, tree: AST, item: Any) -> None:
        """Compile and execute `tree`. May be extended in subclasses."""

        # We pass the source file of `item` such that we can retrieve it
        # when accessing the location of the new compiled code
        source = cast(str, inspect.getsourcefile(item))
        code = compile(cast(ast.Module, tree), source, 'exec')

        # Enable dependency tracker
        self.globals[DATA_TRACKER] = self.dependency_tracker

        # Execute the code, resulting in a redefinition of item
        exec(code, self.globals)

class Slicer(Slicer):
    def instrument(self, item: Any) -> Any:
        """Instrument `item`, transforming its source code, and re-defining it."""
        if is_internal(item.__name__):
            return item  # Do not instrument `print()` and the like

        if inspect.isbuiltin(item):
            return item  # No source code

        item = super().instrument(item)
        tree = self.parse(item)
        tree = self.transform(tree)
        self.execute(tree, item)

        new_item = self.globals[item.__name__]
        return new_item

class Slicer(Slicer):
    def restore(self) -> None:
        """Restore original code."""
        if DATA_TRACKER in self.globals:
            self.saved_dependencies = self.globals[DATA_TRACKER]
            del self.globals[DATA_TRACKER]

        super().restore()

class Slicer(Slicer):
    def dependencies(self) -> Dependencies:
        """Return collected dependencies."""
        if self.saved_dependencies is None:
            return Dependencies({}, {})
        return self.saved_dependencies.dependencies()

    def code(self, *args: Any, **kwargs: Any) -> None:
        """Show code of instrumented items, annotated with dependencies."""
        first = True
        for item in self.instrumented_items:
            if not first:
                print()
            self.dependencies().code(item, *args, **kwargs)  # type: ignore
            first = False

    def graph(self, *args: Any, **kwargs: Any) -> Digraph:
        """Show dependency graph."""
        return self.dependencies().graph(*args, **kwargs)  # type: ignore

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """If the object is output in Jupyter, render dependencies as a SVG graph"""
        return self.graph()._repr_mimebundle_(include, exclude)

if __name__ == '__main__':
    with Slicer(middle) as slicer:
        m = middle(2, 1, 3)
    m

if __name__ == '__main__':
    print(slicer.dependencies())

if __name__ == '__main__':
    slicer.code()

if __name__ == '__main__':
    slicer

if __name__ == '__main__':
    print(repr(slicer.dependencies()))

### Diagnostics

if __name__ == '__main__':
    print('\n### Diagnostics')



## More Examples
## -------------

if __name__ == '__main__':
    print('\n## More Examples')



### Square Root

if __name__ == '__main__':
    print('\n### Square Root')



import math

from .Assertions import square_root  # minor dependency

if __name__ == '__main__':
    print_content(inspect.getsource(square_root), '.py')

if __name__ == '__main__':
    with Slicer(square_root, log=True) as root_slicer:
        y = square_root(2.0)

if __name__ == '__main__':
    root_slicer

if __name__ == '__main__':
    root_slicer.code()

if __name__ == '__main__':
    quiz("Why don't `assert` statements induce control dependencies?",
         [
             "We have no special handling of `raise` statements",
             "We have no special handling of exceptions",
             "Assertions are not supposed to act as controlling mechanisms",
             "All of the above",
         ], '(1 * 1 << 1 * 1 << 1 * 1)')

### Removing HTML Markup

if __name__ == '__main__':
    print('\n### Removing HTML Markup')



if __name__ == '__main__':
    with Slicer(remove_html_markup) as rhm_slicer:
        s = remove_html_markup("<foo>bar</foo>")

if __name__ == '__main__':
    rhm_slicer

if __name__ == '__main__':
    rhm_slicer.code()

if __name__ == '__main__':
    _, start_remove_html_markup = inspect.getsourcelines(remove_html_markup)
    start_remove_html_markup

if __name__ == '__main__':
    slicing_criterion = ('tag', (remove_html_markup,
                                 start_remove_html_markup + 9))
    tag_deps = rhm_slicer.dependencies().backward_slice(slicing_criterion)  # type: ignore
    tag_deps

### Calls and Augmented Assign

if __name__ == '__main__':
    print('\n### Calls and Augmented Assign')



def add_to(n, m):  # type: ignore
    n += m
    return n

def mul_with(x, y):  # type: ignore
    x *= y
    return x

def test_math() -> None:
    return mul_with(1, add_to(2, 3))

if __name__ == '__main__':
    with Slicer(add_to, mul_with, test_math) as math_slicer:
        test_math()

if __name__ == '__main__':
    math_slicer

if __name__ == '__main__':
    math_slicer.code()

## Dynamic Instrumentation
## -----------------------

if __name__ == '__main__':
    print('\n## Dynamic Instrumentation')



### Excursion: Implementing Dynamic Instrumentation

if __name__ == '__main__':
    print('\n### Excursion: Implementing Dynamic Instrumentation')



class WithVisitor(NodeVisitor):
    def __init__(self) -> None:
        self.withs: List[ast.With] = []

    def visit_With(self, node: ast.With) -> AST:
        self.withs.append(node)
        return self.generic_visit(node)

class Slicer(Slicer):
    def our_with_block(self) -> ast.With:
        """Return the currently active `with` block."""
        frame = self.caller_frame()
        source_lines, starting_lineno = inspect.getsourcelines(frame)
        starting_lineno = max(starting_lineno, 1)
        if len(source_lines) == 1:
            # We only get one `with` line, rather than the full block
            # This happens in Jupyter notebooks with iPython 8.1.0 and later.
            # Here's a hacky workaround to get the cell contents:
            # https://stackoverflow.com/questions/51566497/getting-the-source-of-an-object-defined-in-a-jupyter-notebook
            source_lines = inspect.linecache.getlines(inspect.getfile(frame))  # type: ignore
            starting_lineno = 1

        source_ast = ast.parse(''.join(source_lines))
        wv = WithVisitor()
        wv.visit(source_ast)

        for with_ast in wv.withs:
            if starting_lineno + (with_ast.lineno - 1) == frame.f_lineno:
                return with_ast

        raise ValueError("Cannot find 'with' block")

class CallCollector(NodeVisitor):
    def __init__(self) -> None:
        self.calls: Set[str] = set()

    def visit_Call(self, node: ast.Call) -> AST:
        caller_id = ast.unparse(node.func).strip()
        self.calls.add(caller_id)
        return self.generic_visit(node)

class Slicer(Slicer):
    def calls_in_our_with_block(self) -> Set[str]:
        """Return a set of function names called in the `with` block."""
        block_ast = self.our_with_block()
        cc = CallCollector()
        for stmt in block_ast.body:
            cc.visit(stmt)
        return cc.calls

class Slicer(Slicer):
    def funcs_in_our_with_block(self) -> List[Callable]:
        funcs = []
        for id in self.calls_in_our_with_block():
            func = self.search_func(id)
            if func:
                funcs.append(func)

        return funcs

class Slicer(Slicer):
    def default_items_to_instrument(self) -> List[Callable]:
        # In _data.call(), return instrumented function
        self.dependency_tracker.instrument_call = self.instrument  # type: ignore

        # Start instrumenting the functions in our `with` block
        return self.funcs_in_our_with_block()

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



def fun_1(x: int) -> int:
    return x

def fun_2(x: int) -> int:
    return fun_1(x)

if __name__ == '__main__':
    with Slicer(log=True) as slicer:
        fun_2(10)

if __name__ == '__main__':
    slicer

## More Applications
## -----------------

if __name__ == '__main__':
    print('\n## More Applications')



### Verifying Information Flows

if __name__ == '__main__':
    print('\n### Verifying Information Flows')



import hashlib

from .bookutils import input, next_inputs

SECRET_HASH_DIGEST = '59f2da35bcc39525b87932b4cc1f3d68'

def password_checker() -> bool:
    """Request a password. Return True if correct."""
    secret_password = input("Enter secret password: ")
    password_digest = hashlib.md5(secret_password.encode('utf-8')).hexdigest()

    if password_digest == SECRET_HASH_DIGEST:
        return True
    else:
        return False

if __name__ == '__main__':
    next_inputs(['secret123'])

if __name__ == '__main__':
    with Slicer() as slicer:
        valid_pwd = password_checker()

if __name__ == '__main__':
    slicer

if __name__ == '__main__':
    secret_answers = [
        'automated',
        'debugging',
        'is',
        'fun'
    ]

    quiz("What is the secret password, actually?", 
         [f"`{repr(s)}`" for s in secret_answers],
         min([i + 1 for i, ans in enumerate(secret_answers) 
              if hashlib.md5(ans.encode('utf-8')).hexdigest() == 
                  SECRET_HASH_DIGEST])
        )

### Assessing Test Quality

if __name__ == '__main__':
    print('\n### Assessing Test Quality')



if __name__ == '__main__':
    _, start_square_root = inspect.getsourcelines(square_root)

if __name__ == '__main__':
    print_content(inspect.getsource(square_root), '.py',
                  start_line_number=start_square_root)

def square_root_unchecked(x):  # type: ignore
    assert True  # <-- new "precondition"

    approx = None
    guess = x / 2
    while approx != guess:
        approx = guess
        guess = (approx + x / approx) / 2

    assert True  # <-- new "postcondition"
    return approx

if __name__ == '__main__':
    postcondition_lineno = start_square_root + 9
    postcondition_lineno

if __name__ == '__main__':
    with Slicer() as slicer:
        y = square_root(4)

    slicer.dependencies().backward_slice((square_root, postcondition_lineno))

if __name__ == '__main__':
    with Slicer() as slicer:
        y = square_root_unchecked(4)

    slicer.dependencies().backward_slice((square_root, postcondition_lineno))

### Use in Statistical Debugging

if __name__ == '__main__':
    print('\n### Use in Statistical Debugging')



## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



def demo(x: int) -> int:
    z = x
    while x <= z <= 64:
        z *= 2
    return z

if __name__ == '__main__':
    with Slicer() as slicer:
        demo(10)

if __name__ == '__main__':
    slicer

if __name__ == '__main__':
    slicer.code()

if __name__ == '__main__':
    slicer.dependencies().all_vars()

if __name__ == '__main__':
    _, start_demo = inspect.getsourcelines(demo)
    start_demo

if __name__ == '__main__':
    slicer.dependencies().backward_slice(('z', (demo, start_demo + 1))).graph()  # type: ignore

from .ClassDiagram import display_class_hierarchy, class_tree

if __name__ == '__main__':
    assert class_tree(Slicer)[0][0] == Slicer

if __name__ == '__main__':
    display_class_hierarchy([Slicer, DependencyTracker, 
                             StackInspector, Dependencies],
                            abstract_classes=[
                                StackInspector,
                                Instrumenter
                            ],
                            public_methods=[
                                StackInspector.caller_frame,
                                StackInspector.caller_function,
                                StackInspector.caller_globals,
                                StackInspector.caller_locals,
                                StackInspector.caller_location,
                                StackInspector.search_frame,
                                StackInspector.search_func,
                                Instrumenter.__init__,
                                Instrumenter.__enter__,
                                Instrumenter.__exit__,
                                Instrumenter.instrument,
                                Slicer.__init__,
                                Slicer.code,
                                Slicer.dependencies,
                                Slicer.graph,
                                Slicer._repr_mimebundle_,
                                DataTracker.__init__,
                                DataTracker.__enter__,
                                DataTracker.__exit__,
                                DataTracker.arg,
                                DataTracker.augment,
                                DataTracker.call,
                                DataTracker.get,
                                DataTracker.param,
                                DataTracker.ret,
                                DataTracker.set,
                                DataTracker.test,
                                DataTracker.__repr__,
                                DependencyTracker.__init__,
                                DependencyTracker.__enter__,
                                DependencyTracker.__exit__,
                                DependencyTracker.arg,
                                # DependencyTracker.augment,
                                DependencyTracker.call,
                                DependencyTracker.get,
                                DependencyTracker.param,
                                DependencyTracker.ret,
                                DependencyTracker.set,
                                DependencyTracker.test,
                                DependencyTracker.__repr__,
                                Dependencies.__init__,
                                Dependencies.__repr__,
                                Dependencies.__str__,
                                Dependencies._repr_mimebundle_,
                                Dependencies.code,
                                Dependencies.graph,
                                Dependencies.backward_slice,
                                Dependencies.all_functions,
                                Dependencies.all_vars,
                            ],
                            project='debuggingbook')

## Things that do not Work
## -----------------------

if __name__ == '__main__':
    print('\n## Things that do not Work')



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



### Exercise 1: Control Slices

if __name__ == '__main__':
    print('\n### Exercise 1: Control Slices')



### Exercise 2: Incremental Exploration

if __name__ == '__main__':
    print('\n### Exercise 2: Incremental Exploration')



### Exercise 3: Forward Slicing

if __name__ == '__main__':
    print('\n### Exercise 3: Forward Slicing')



### Exercise 4: Code with Forward Dependencies

if __name__ == '__main__':
    print('\n### Exercise 4: Code with Forward Dependencies')



### Exercise 5: Flow Assertions

if __name__ == '__main__':
    print('\n### Exercise 5: Flow Assertions')



def assert_flow(target: Any, source: List[Any]) -> bool:
    """
    Raise an `AssertionError` if the dependencies of `target`
    are not equal to `source`.
    """
    ...
    return True

def demo4() -> int:
    x = 25
    y = 26
    assert_flow(y, [x])  # ensures that `y` depends on `x` only
    return y

if __name__ == '__main__':
    with Slicer() as slicer:
        demo4()

### Exercise 6: Checked Coverage

if __name__ == '__main__':
    print('\n### Exercise 6: Checked Coverage')


