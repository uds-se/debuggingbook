#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Fuzzing Book".
# Web site: https://www.fuzzingbook.org/html/Slicer.html
# Last change: 2020-12-21 13:29:01+01:00
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


# # Tracking Failure Origins

if __name__ == "__main__":
    print('# Tracking Failure Origins')




if __name__ == "__main__":
    from bookutils import YouTubeVideo
    # YouTubeVideo("w4u5gCgPlmg")


if __name__ == "__main__":
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




# ## Dependencies

if __name__ == "__main__":
    print('\n## Dependencies')




def middle(x, y, z):
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

if __name__ == "__main__":
    m = middle(1, 2, 3)
    m


if __name__ == "__main__":
    m = middle(2, 1, 3)
    m


# ### Excursion: Visualizing Dependencies

if __name__ == "__main__":
    print('\n### Excursion: Visualizing Dependencies')




# #### A Class for Dependencies

if __name__ == "__main__":
    print('\n#### A Class for Dependencies')




class Dependencies(object):
    def __init__(self, data=None, control=None):
        """
Create a dependency graph from data and control. Both are dictionaries
holding _nodes_ as keys and lists of nodes as values.
Each node comes as a tuple (variable_name, location)
where `variable_name` is a string 
and `location` is a pair (code_name, lineno)
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
    def validate(self):
        """Check dependency structure."""
        assert isinstance(self.data, dict)
        assert isinstance(self.control, dict)
        
        for node in (self.data.keys()) | set(self.control.keys()):
            var_name, location = node
            assert isinstance(var_name, str)
            func_name, lineno = location
            assert isinstance(func_name, str)
            assert isinstance(lineno, int)

import inspect

class Dependencies(Dependencies):
    def source(self, node):
        """Return the source code for a given node."""
        (name, location) = node
        code_name, lineno = location
        if code_name not in globals():
            return ''

        fun = globals()[code_name]
        source_lines, first_lineno = inspect.getsourcelines(fun)

        try:
            line = source_lines[lineno - first_lineno].strip()
        except IndexError:
            line = ''

        return line

if __name__ == "__main__":
    test_deps = Dependencies()
    test_deps.source(('z', ('middle', 1)))


# #### Drawing Dependencies

if __name__ == "__main__":
    print('\n#### Drawing Dependencies')




if __name__ == "__main__":
    from graphviz import Digraph, nohtml


import html

class Dependencies(Dependencies):
    NODE_COLOR = 'peachpuff'
    FONT_NAME = 'Fira Mono'

    def make_graph(self, name="dependencies", comment="Dependencies"):
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
    def graph(self):
        """Draw dependencies."""
        self.validate()

        g = self.make_graph()
        self.draw_dependencies(g)
        self.add_hierarchy(g)
        return g

class Dependencies(Dependencies):
    def all_vars(self):
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
    def draw_dependencies(self, g):
        for var in self.all_vars():
            g.node(self.id(var),
                   label=self.label(var),
                   tooltip=self.tooltip(var))

            if var in self.data:
                for source in self.data[var]:
                    g.edge(self.id(source), self.id(var))

            if var in self.control:
                for source in self.control[var]:
                    g.edge(self.id(source), self.id(var),
                           style='dashed', color='grey')

class Dependencies(Dependencies):
    def id(self, var):
        """Return a unique ID for VAR."""
        id = ""
        # Avoid non-identifier characters
        for c in repr(var):
            if c.isalnum() or c == '_':
                id += c
            if c == ':' or c == ',':
                id += '_'
        return id

    def label(self, var):
        """Render node VAR using HTML style."""
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

    def tooltip(self, var):
        """Return a tooltip for node VAR."""
        (name, location) = var
        code_name, lineno = location
        return f"{code_name}:{lineno}"

class Dependencies(Dependencies):
    def add_hierarchy(self, g):
        """Add invisible edges for a proper hierarchy."""
        code_names = self.all_codes()
        for code_name in code_names:
            last_var = None
            last_lineno = 0
            for (lineno, var) in code_names[code_name]:
                if last_var is not None and lineno > last_lineno:
                    g.edge(self.id(last_var),
                           self.id(var),
                           style='invis')

                last_var = var
                last_lineno = lineno

        return g

class Dependencies(Dependencies):
    def all_codes(self):
        code_names = {}
        for var in self.all_vars():
            (name, location) = var
            code_name, lineno = location
            if code_name not in code_names:
                code_names[code_name] = []
            code_names[code_name].append((lineno, var))

        for code_name in code_names:
            code_names[code_name].sort()

        return code_names

def middle_deps():
    return Dependencies({('z', ('middle', 1)): set(), ('y', ('middle', 1)): set(), ('x', ('middle', 1)): set(), ('<test>', ('middle', 2)): {('y', ('middle', 1)), ('z', ('middle', 1))}, ('<test>', ('middle', 3)): {('y', ('middle', 1)), ('x', ('middle', 1))}, ('<test>', ('middle', 5)): {('z', ('middle', 1)), ('x', ('middle', 1))}, ('<middle() return value>', ('middle', 6)): {('y', ('middle', 1))}}, {('z', ('middle', 1)): set(), ('y', ('middle', 1)): set(), ('x', ('middle', 1)): set(), ('<test>', ('middle', 2)): set(), ('<test>', ('middle', 3)): {('<test>', ('middle', 2))}, ('<test>', ('middle', 5)): {('<test>', ('middle', 3))}, ('<middle() return value>', ('middle', 6)): {('<test>', ('middle', 5))}})

if __name__ == "__main__":
    middle_deps().graph()


# #### Slices

if __name__ == "__main__":
    print('\n#### Slices')




class Dependencies(Dependencies):
    def expand_items(self, items):
        all_items = []
        for item in items:
            if isinstance(item, str):
                for var in self.all_vars():
                    (name, location) = var
                    if name == item:
                        all_items.append(var)
            else:
                all_items.append(item)

        return all_items

    def backward_slice(self, *items, mode="cd", depth=-1):
        """Create a backward slice from nodes ITEMS."""
        data = {}
        control = {}
        queue = self.expand_items(items)
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
                
            depth = depth - 1

        return Dependencies(data, control)

# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ### Data Dependencies

if __name__ == "__main__":
    print('\n### Data Dependencies')




if __name__ == "__main__":
    # ignore
    middle_deps().backward_slice('<middle() return value>', mode='d').graph()


# ### Control Dependencies

if __name__ == "__main__":
    print('\n### Control Dependencies')




if __name__ == "__main__":
    # ignore
    middle_deps().backward_slice('<middle() return value>', mode='c', depth=1).graph()


if __name__ == "__main__":
    # ignore
    middle_deps().backward_slice('<middle() return value>', mode='c').graph()


# ### Dependency Graphs

if __name__ == "__main__":
    print('\n### Dependency Graphs')




if __name__ == "__main__":
    # ignore
    middle_deps().graph()


# ### Showing Dependencies with Code

if __name__ == "__main__":
    print('\n### Showing Dependencies with Code')




# #### Excursion: Listing Dependencies

if __name__ == "__main__":
    print('\n#### Excursion: Listing Dependencies')




class Dependencies(Dependencies):
    def format_var(self, var, current_location=None):
        """Return string for VAR in CURRENT_LOCATION."""
        name, location = var
        location_name, lineno = location
        if location_name != current_location:
            return f"{name} ({location_name}:{lineno})"
        else:
            return f"{name} ({lineno})"

class Dependencies(Dependencies):
    def __str__(self):
        self.validate()

        out = ""
        for code_name in self.all_codes():
            if out != "":
                out += "\n"
            out += f"{code_name}():\n"

            all_vars = list(set(self.data.keys()) | set(self.control.keys()))
            all_vars.sort(key=lambda var: var[1][1])

            for var in all_vars:
                (name, location) = var
                var_code, var_lineno = location
                if var_code != code_name:
                    continue

                all_deps = ""
                for (source, arrow) in [(self.data, "<="), (self.control, "<-")]:
                    deps = ""
                    for data_dep in source[var]:
                        if deps == "":
                            deps = f" {arrow} "
                        else:
                            deps += ", "
                        deps += self.format_var(data_dep, code_name)

                    if deps != "":
                        if all_deps != "":
                            all_deps += ";"
                        all_deps += deps

                if all_deps == "":
                    continue

                out += ("    " + 
                        self.format_var(var, code_name) +
                        all_deps + "\n")

        return out

    def __repr__(self):
        # Useful for saving and restoring values
        return f"Dependencies({self.data}, {self.control})"

if __name__ == "__main__":
    print(middle_deps())


if __package__ is None or __package__ == "":
    from bookutils import print_content
else:
    from .bookutils import print_content


class Dependencies(Dependencies):
    def code(self, item, mode='cd'):
        """List ITEM on standard output, including dependencies as comments."""
        all_vars = self.all_vars()
        slice_locations = set(location for name, location in all_vars)

        source_lines, first_lineno = inspect.getsourcelines(item)

        n = first_lineno
        for line in source_lines:
            line_location = (item.__name__, n)
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
                            deps += self.format_var(dep_var, item.__name__)

                if deps != "":
                    if comment != "":
                        comment += "; "
                    comment += deps

            if comment != "":
                line = line.rstrip() + "  # " + comment

            print_content(line.rstrip(), '.py')
            print()
            n += 1

# #### End of Excursion

if __name__ == "__main__":
    print('\n#### End of Excursion')




if __name__ == "__main__":
    # ignore
    middle_deps().code(middle)


if __package__ is None or __package__ == "":
    from bookutils import quiz
else:
    from .bookutils import quiz


if __name__ == "__main__":
    quiz("Which of the following <samp>middle()</samp> code lines should be fixed?",
        [
            "Line 2: <samp>if y < z:</samp>",
            "Line 3: <samp>if x < y:</samp>",
            "Line 5: <samp>elif x < z:</samp>",
            "Line 6: <samp>return z</samp>",
        ], (1 ** 0 + 1 ** 1) ** (1 ** 2 + 1 ** 3))


# ## Tracking Techniques

if __name__ == "__main__":
    print('\n## Tracking Techniques')




# ### Wrapping Data Objects

if __name__ == "__main__":
    print('\n### Wrapping Data Objects')




class MyInt(int):
    def __new__(cls, value, *args, **kwargs):
        return super(cls, cls).__new__(cls, value)

    def __repr__(self):
        return f"{int(self)}"

if __name__ == "__main__":
    x = MyInt(5)


if __name__ == "__main__":
    x, x + 1


if __name__ == "__main__":
    x.origin = "Line 5"


if __name__ == "__main__":
    x.origin


# ### Wrapping Data Accesses

if __name__ == "__main__":
    print('\n### Wrapping Data Accesses')




# ## A Data Tracker

if __name__ == "__main__":
    print('\n## A Data Tracker')




class DataTracker(object):
    def __init__(self, log=False):
        """Initialize. If LOG is set, turn on logging."""
        self.log = log

class DataTracker(DataTracker):
    def caller_frame(self):
        """Return the frame of the caller."""
        frame = inspect.currentframe()
        while ('self' in frame.f_locals and 
               isinstance(frame.f_locals['self'], self.__class__)):
               frame = frame.f_back
        return frame

    def caller_location(self):
        """Return the location (code name, lineno) of the caller."""
        frame = self.caller_frame()
        if frame.f_code.co_name in globals():
            return frame.f_code.co_name, frame.f_lineno
        else:
            return frame.f_code.co_filename, frame.f_lineno

class DataTracker(DataTracker):
    def set(self, name, value):
        """Track setting NAME to VALUE."""
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: setting {name}")

        return value

class DataTracker(DataTracker):
    def get(self, name, value):
        """Track getting VALUE from NAME."""

        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: getting {name}")
            
        return value

class DataTracker(DataTracker):
    def __repr__(self):
        return super().__repr__()

if __name__ == "__main__":
    _test_data = DataTracker(log=True)
    x = _test_data.set('x', 1)


if __name__ == "__main__":
    _test_data.get('x', x)


# ## Instrumenting Source Code

if __name__ == "__main__":
    print('\n## Instrumenting Source Code')




import ast
import astor

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
    middle_tree = ast.parse(inspect.getsource(middle))
    show_ast(middle_tree)


# ### Tracking Variable Access

if __name__ == "__main__":
    print('\n### Tracking Variable Access')




from ast import NodeTransformer

DATA_TRACKER = '_data'

class TrackGetTransformer(NodeTransformer):
    def visit_Name(self, node):
        self.generic_visit(node)

        if node.id in dir(__builtins__):
            # Do not change built-in names
            return node
        
        if node.id == DATA_TRACKER:
            # Do not change own accesses
            return node
        
        if isinstance(node.ctx, Store):
            # Do not change stores
            return node

        return make_get_data(node.id)

from ast import Module, Subscript, Name, Load, Store, \
    Assign, AugAssign, Attribute, Constant, \
    With, withitem, Return, Index, Str, Call, Expr

def make_get_data(id, method='get'):
    return Call(func=Attribute(value=Name(id=DATA_TRACKER, ctx=Load()), 
                               attr=method, ctx=Load()),
                args=[Str(s=id), Name(id=id, ctx=Load())],
                keywords=[])

if __name__ == "__main__":
    show_ast(Module(body=[make_get_data("x")]))


if __name__ == "__main__":
    print(ast.dump(ast.parse("_data.get('x', x)")))


if __name__ == "__main__":
    TrackGetTransformer().visit(middle_tree);


def dump_tree(tree):
    print(astor.to_source(tree))
    ast.fix_missing_locations(tree)  # Must run this before compiling
    _ = compile(tree, '<string>', 'exec')

if __name__ == "__main__":
    dump_tree(middle_tree)


class DataTrackerTester(object):
    def __init__(self, tree, func, log=True):
        self.code = compile(tree, '<string>', 'exec')
        self.func = func
        self.log = log
        
    def make_data_tracker(self):
        return DataTracker(log=self.log)

    def __enter__(self):
        """Rewrite function"""
        tracker = self.make_data_tracker()
        globals()[DATA_TRACKER] = tracker
        exec(self.code, globals())
        return tracker

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore function"""
        globals()[self.func.__name__] = self.func
        del globals()[DATA_TRACKER]

if __name__ == "__main__":
    print_content(inspect.getsource(middle), '.py', start_line_number=1)


if __name__ == "__main__":
    with DataTrackerTester(middle_tree, middle):
        middle(2, 1, 3)


if __name__ == "__main__":
    middle(2, 1, 3)


# ### Excursion: Tracking Assignments

if __name__ == "__main__":
    print('\n### Excursion: Tracking Assignments')




if __name__ == "__main__":
    print(ast.dump(ast.parse("_data.set('x', value)")))


def make_set_data(id, value, method='set'):
    """Construct a subtree _data.METHOD('ID', VALUE)"""
    return Call(func=Attribute(value=Name(id=DATA_TRACKER, ctx=Load()), 
                               attr=method, ctx=Load()), 
                args=[Str(s=id), value], 
                keywords=[])

class TrackSetTransformer(NodeTransformer):
    def visit_Assign(self, node):
        value = astor.to_source(node.value)
        if value.startswith(DATA_TRACKER):
            return node  # Do not apply twice
        
        id = astor.to_source(node).split(' = ')[0].strip()
        node.value = make_set_data(id, node.value)
        return node

class TrackSetTransformer(TrackSetTransformer):
    def visit_AugAssign(self, node):
        value = astor.to_source(node.value)
        if value.startswith(DATA_TRACKER):
            return node  # Do not apply twice
        
        id = astor.to_source(node.target).strip()
        node.value = make_set_data(id, node.value, method='augment')
        return node

class DataTracker(DataTracker):
    def augment(self, name, value):
        """Track augmenting NAME with VALUE."""
        self.set(name, self.get(name, value))
        return value

def assign_test(x):
    forty_two = 42
    a, b, c = 1, 2, 3
    c[x] = 47
    foo *= bar + 1

if __name__ == "__main__":
    assign_tree = ast.parse(inspect.getsource(assign_test))


if __name__ == "__main__":
    TrackSetTransformer().visit(assign_tree)
    dump_tree(assign_tree)


if __name__ == "__main__":
    TrackGetTransformer().visit(assign_tree)
    dump_tree(assign_tree)


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ### Excursion: Tracking Return Values

if __name__ == "__main__":
    print('\n### Excursion: Tracking Return Values')




class TrackReturnTransformer(NodeTransformer):
    def __init__(self):
        self.function_name = None
        super().__init__()

    def visit_FunctionDef(self, node):
        outer_name = self.function_name
        self.function_name = node.name  # Save current name
        self.generic_visit(node)
        self.function_name = outer_name
        return node

    def return_value(self):
        if self.function_name is None:
            return "<return value>"
        else:
            return f"<{self.function_name}() return value>"

    def visit_Return(self, node):
        if node.value is not None:
            value = astor.to_source(node.value)
            if not value.startswith(DATA_TRACKER + '.set'):
                node.value = make_set_data(self.return_value(), node.value)
        return node

if __name__ == "__main__":
    TrackReturnTransformer().visit(middle_tree)
    dump_tree(middle_tree)


if __name__ == "__main__":
    with DataTrackerTester(middle_tree, middle):
        middle(2, 1, 3)


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ### Excursion: Tracking Control

if __name__ == "__main__":
    print('\n### Excursion: Tracking Control')




class TrackControlTransformer(NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        node.test = self.make_test(node.test)
        node.body = self.make_with(node.body)
        node.orelse = self.make_with(node.orelse)
        return node

class TrackControlTransformer(TrackControlTransformer):
    def make_with(self, block):
        """Create a subtree 'with _data: BLOCK'"""
        if len(block) == 0:
            return []
        
        block_as_text = astor.to_source(block[0])
        if block_as_text.startswith('with ' + DATA_TRACKER):
            return block  # Do not apply twice

        return [With(
            items=[
                withitem(
                    context_expr=Name(id=DATA_TRACKER, ctx=Load()),
                    optional_vars=None)
            ],
            body=block
        )]

class TrackControlTransformer(TrackControlTransformer):
    def make_test(self, test):
        test_as_text = astor.to_source(test)
        if test_as_text.startswith(DATA_TRACKER + '.test'):
            return test

        return Call(func=Attribute(value=Name(id=DATA_TRACKER, ctx=Load()),
                                   attr='test',
                                   ctx=Load()),
                     args=[test],
                     keywords=[])

class TrackControlTransformer(TrackControlTransformer):
    def visit_While(self, node):
        self.generic_visit(node)
        node.test = self.make_test(node.test)
        node.body = self.make_with(node.body)
        node.orelse = self.make_with(node.orelse)
        return node

class TrackControlTransformer(TrackControlTransformer):
    def visit_For(self, node):
        self.generic_visit(node)
        id = astor.to_source(node.target).strip()
        node.iter = make_set_data(id, node.iter)

        # Uncomment if you want iterators to control their bodies
        # node.body = self.make_with(node.body)
        # node.orelse = self.make_with(node.orelse)
        return node

if __name__ == "__main__":
    TrackControlTransformer().visit(middle_tree)
    dump_tree(middle_tree)


class DataTracker(DataTracker):
    def test(self, cond):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: testing condition")

        return cond

class DataTracker(DataTracker):
    def __enter__(self):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: entering block")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: exiting block")

if __name__ == "__main__":
    with DataTrackerTester(middle_tree, middle):
        middle(2, 1, 3)


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ### Excursion: Tracking Calls and Arguments

if __name__ == "__main__":
    print('\n### Excursion: Tracking Calls and Arguments')




class TrackCallTransformer(NodeTransformer):
    def make_call(self, node, func):
        """Return _data.call(FUNC)(NODE)"""
        return Call(func=Attribute(value=Name(id=DATA_TRACKER,
                                              ctx=Load()),
                                   attr=func,
                                   ctx=Load()),
                     args=[node],
                     keywords=[])

    def visit_Call(self, node):
        self.generic_visit(node)

        call_as_text = astor.to_source(node)
        if call_as_text.startswith(DATA_TRACKER + '.ret'):
            return node  # Already applied
        
        func_as_text = astor.to_source(node)
        if func_as_text.startswith(DATA_TRACKER + '.'):
            return node  # Own function

        new_args = []
        for arg in node.args:
            new_args.append(self.make_call(arg, 'arg'))
        node.args = new_args

        for kw in node.keywords:
            kw.value = self.make_call(kw.value, 'arg')

        node.func = self.make_call(node.func, 'call')
        return self.make_call(node, 'ret')

def test_call():
    x = middle(1, 2, middle(1, 2, 3))

if __name__ == "__main__":
    call_tree = ast.parse(inspect.getsource(test_call))
    dump_tree(call_tree)


if __name__ == "__main__":
    TrackGetTransformer().visit(call_tree);


if __name__ == "__main__":
    dump_tree(call_tree)


def f():
    return math.isclose(1, 1.0)

if __name__ == "__main__":
    f_tree = ast.parse(inspect.getsource(f))
    dump_tree(f_tree)


if __name__ == "__main__":
    TrackCallTransformer().visit(f_tree);


if __name__ == "__main__":
    dump_tree(f_tree)


class DataTracker(DataTracker):
    def arg(self, value):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: pushing arg")

        return value

class DataTracker(DataTracker):
    def ret(self, value):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: returned from call")

        return value

class DataTracker(DataTracker):
    def call(self, func):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: calling {func}")

        return func

if __name__ == "__main__":
    with DataTrackerTester(call_tree, test_call):
        test_call()


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ### Excursion: Tracking Parameters

if __name__ == "__main__":
    print('\n### Excursion: Tracking Parameters')




if __name__ == "__main__":
    print(ast.dump(ast.parse("_data.param('x', x)")))


class TrackParamsTransformer(NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)

        named_args = []
        for child in ast.iter_child_nodes(node.args):
            if isinstance(child, ast.arg):
                named_args.append(child.arg)

        create_stmts = []
        for arg in named_args:
            create_stmt = Expr(
                value=Call(
                    func=Attribute(value=Name(id=DATA_TRACKER, ctx=Load()),
                                   attr='param', ctx=Load()),
                    args=[Str(s=arg), Name(id=arg, ctx=Load())],
                    keywords=[]
                )
            )
            create_stmts.append(create_stmt)
            
        # Reverse parameters such that we can later easily match them
        # with passed arguments (evaluated left to right)
        create_stmts.reverse()

        node.body = create_stmts + node.body
        return node

if __name__ == "__main__":
    TrackParamsTransformer().visit(middle_tree)
    dump_tree(middle_tree)


class DataTracker(DataTracker):
    def param(self, name, value):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: initializing {name}")

        return self.set(name, value)

if __name__ == "__main__":
    with DataTrackerTester(middle_tree, middle):
        middle(2, 1, 3)


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




if __name__ == "__main__":
    dump_tree(middle_tree)


if __name__ == "__main__":
    with DataTrackerTester(middle_tree, middle):
        m = middle(2, 1, 3)
    m


# ## Tracking Dependencies

if __name__ == "__main__":
    print('\n## Tracking Dependencies')




class DependencyTracker(DataTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.origins = {}  # Where current variables were last set
        self.data_dependencies = {}  # As with Dependencies, above
        self.control_dependencies = {}

        self.last_read = []  # List of last read variables
        self.last_checked_location = None
        self._ignore_location_change = False

        self.data = [[]]  # Data stack
        self.control = [[]]  # Control stack

        self.args = []  # Argument stack

# ### Data Dependencies

if __name__ == "__main__":
    print('\n### Data Dependencies')




# #### Reading Variables

if __name__ == "__main__":
    print('\n#### Reading Variables')




class DependencyTracker(DependencyTracker):
    def get(self, name, value):
        self.check_location()
        self.last_read.append(name)
        return super().get(name, value)
    
    def check_location(self):
        pass  # More on that below

if __name__ == "__main__":
    x = 5
    y = 3


if __name__ == "__main__":
    _test_data = DependencyTracker(log=True)
    _test_data.get('x', x) + _test_data.get('y', y)


if __name__ == "__main__":
    _test_data.last_read


# #### Checking Locations

if __name__ == "__main__":
    print('\n#### Checking Locations')




class DependencyTracker(DependencyTracker):
    def clear_read(self):
        if self.log:
            direct_caller = inspect.currentframe().f_back.f_code.co_name
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: "
                  f"clearing read variables {self.last_read} "
                  f"(from {direct_caller})")

        self.last_read = []

    def check_location(self):
        location = self.caller_location()
        if self.last_checked_location != location:
            if self._ignore_location_change:
                self._ignore_location_change = False
            else:
                self.clear_read()

        self.last_checked_location = location

class DependencyTracker(DependencyTracker):
    def ignore_next_location_change(self):
        self._ignore_location_change = True

    def ignore_location_change(self):
        self.last_checked_location = self.caller_location()

if __name__ == "__main__":
    _test_data = DependencyTracker()


if __name__ == "__main__":
    _test_data.get('x', x) + _test_data.get('y', y)


if __name__ == "__main__":
    _test_data.last_read


if __name__ == "__main__":
    a = 42
    b = -1
    _test_data.get('a', a) + _test_data.get('b', b)


if __name__ == "__main__":
    _test_data.last_read


# #### Setting Variables

if __name__ == "__main__":
    print('\n#### Setting Variables')




import itertools

class DependencyTracker(DependencyTracker):
    def set(self, name, value):

        def add_dependencies(dependencies, vars_read, tp):
            """Add origins of VARS_READ to DEPENDENCIES."""
            for var_read in vars_read:
                if var_read in self.origins:
                    origin = self.origins[var_read]
                    dependencies.add((var_read, origin))

                    if self.log:
                        origin_name, origin_lineno = origin
                        code_name, lineno = self.caller_location()
                        print(f"{code_name}:{lineno}: "
                              f"new {tp} dependency: "
                              f"{name} <= {var_read} "
                              f"({origin_name}:{origin_lineno})")

        self.check_location()
        ret = super().set(name, value)
        location = self.caller_location()

        add_dependencies(self.data_dependencies.setdefault
                         ((name, location), set()),
                         self.last_read, tp="data")
        add_dependencies(self.control_dependencies.setdefault
                         ((name, location), set()),
                         itertools.chain.from_iterable(self.control),
                         tp="control")

        self.origins[name] = location

        # Reset read info for next line
        # self.clear_read()
        self.last_read = [name]

        return ret

    def dependencies(self):
        return Dependencies(self.data_dependencies,
                            self.control_dependencies)

if __name__ == "__main__":
    _test_data = DependencyTracker()
    x = _test_data.set('x', 1)
    y = _test_data.set('y', _test_data.get('x', x))
    z = _test_data.set('z', _test_data.get('x', x) + _test_data.get('y', y))


if __name__ == "__main__":
    _test_data.origins


if __name__ == "__main__":
    _test_data.data_dependencies


if __name__ == "__main__":
    # ignore
    _test_data.dependencies().graph()


# ### Control Dependencies

if __name__ == "__main__":
    print('\n### Control Dependencies')




class DependencyTracker(DependencyTracker):
    TEST = '<test>'

    def test(self, value):
        self.set(self.TEST, value)
        return super().test(value)

class DependencyTracker(DependencyTracker):
    def __enter__(self):
        self.control.append(self.last_read)
        self.clear_read()
        super().__enter__()

class DependencyTracker(DependencyTracker):
    def __exit__(self, exc_type, exc_value, traceback):
        self.clear_read()
        self.last_read = self.control.pop()
        self.ignore_next_location_change()
        super().__exit__(exc_type, exc_value, traceback)

if __name__ == "__main__":
    _test_data = DependencyTracker()
    x = _test_data.set('x', 1)
    y = _test_data.set('y', _test_data.get('x', x))


if __name__ == "__main__":
    if _test_data.test(_test_data.get('x', x) >= _test_data.get('y', y)):
        with _test_data:
            z = _test_data.set('z',
                               _test_data.get('x', x) + _test_data.get('y', y))


if __name__ == "__main__":
    _test_data.control_dependencies


if __name__ == "__main__":
    # ignore
    _test_data.dependencies().graph()


# ### Calls and Returns

if __name__ == "__main__":
    print('\n### Calls and Returns')




class DependencyTracker(DependencyTracker):
    def call(self, fun):
        # Save context
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: "
                  f"saving read variables {self.last_read}")

        self.data.append(self.last_read)
        self.clear_read()
        self.ignore_next_location_change()

        return super().call(fun)

    def ret(self, value):
        super().ret(value)

        # Restore old context and add return value
        ret_name = None
        for var in self.last_read:
            if var.startswith("<"):  # "<return value>"
                ret_name = var

        self.last_read = self.data.pop()
        if ret_name is not None:
            self.last_read.append(ret_name)
        self.ignore_location_change()

        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: "
                  f"restored read variables {self.last_read}")

        return value

# ### Function Arguments

if __name__ == "__main__":
    print('\n### Function Arguments')




class DependencyTracker(DependencyTracker):
    def arg(self, value):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: "
                  f"saving arg reads {self.last_read}")

        self.args.append(self.last_read)
        self.clear_read()
        return super().arg(value)

    def param(self, name, value):
        self.clear_read()
        if self.args:
            self.last_read = self.args.pop()
            self.ignore_location_change()

        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: "
                  f"restored param {self.last_read}")

        return super().param(name, value)

def call_test():
    def just_x(x, y):
        return x

    a = 42
    b = 47
    return just_x(a, b)

if __name__ == "__main__":
    call_tree = ast.parse(inspect.getsource(call_test))
    TrackCallTransformer().visit(call_tree)
    TrackSetTransformer().visit(call_tree)
    TrackGetTransformer().visit(call_tree)
    TrackControlTransformer().visit(call_tree)
    TrackReturnTransformer().visit(call_tree)
    TrackParamsTransformer().visit(call_tree)
    dump_tree(call_tree)


class DependencyTrackerTester(DataTrackerTester):
    def make_data_tracker(self):
        return DependencyTracker(log=self.log)

if __name__ == "__main__":
    with DependencyTrackerTester(call_tree, call_test, log=False) as deps:
        call_test()


if __name__ == "__main__":
    deps.dependencies().graph()


# ### Diagnostics

if __name__ == "__main__":
    print('\n### Diagnostics')




import re
import sys

class Dependencies(Dependencies):
    def validate(self):
        super().validate()

        for var in self.all_vars():
            source = self.source(var)
            if not source:
                continue

            for dep_var in self.data[var] | self.control[var]:
                dep_name, dep_location = dep_var

                if dep_name == DependencyTracker.TEST:
                    continue

                if dep_name.endswith('return value>'):
                    if source.find('(') < 0:
                        print(f"Warning: {self.format_var(var)} "
                              f"depends on {self.format_var(dep_var)}, "
                              f"but {repr(source)} does not "
                              f"seem to have a call",
                              file=sys.stderr
                             )
                    continue

                if source.startswith('def'):
                    continue   # function call

                rx = re.compile(r'\b' + dep_name + r'\b')
                if rx.search(source) is None:
                    print(f"Warning: {self.format_var(var)} "
                          f"depends on {self.format_var(dep_var)}, "
                          f"but {repr(dep_name)} does not occur "
                          f"in {repr(source)}",
                          file=sys.stderr
                         )

# ## Slicing Code

if __name__ == "__main__":
    print('\n## Slicing Code')




class Instrumenter(object):
    def __init__(self, *items_to_instrument, log=False):
        self.log = log
        self.items_to_instrument = items_to_instrument

    def __enter__(self):
        """Instrument sources"""
        for item in self.items_to_instrument:
            self.instrument(item)
        return self

    def instrument(self, item):
        if self.log:
            print("Instrumenting", item)

class Instrumenter(Instrumenter):
    def __exit__(self, exc_type, exc_value, traceback):
        """Restore sources"""
        self.restore()

    def restore(self):
        for item in self.items_to_instrument:
            globals()[item.__name__] = item

if __name__ == "__main__":
    with Instrumenter(middle) as ins:
        pass


class Slicer(Instrumenter):
    def __init__(self, *items_to_instrument, 
                 dependency_tracker=None,
                 log=False):
        super().__init__(*items_to_instrument, log=log)
        if len(items_to_instrument) == 0:
            raise ValueError("Need one or more items to instrument")

        if dependency_tracker is None:
            dependency_tracker = DependencyTracker(log=(log > 1))
        self.dependency_tracker = dependency_tracker
        self.saved_dependencies = None

class Slicer(Slicer):
    def transformers(self):
        """List of transformers to apply. To be extended in subclasses."""
        return [
            TrackCallTransformer(),
            TrackSetTransformer(),
            TrackGetTransformer(),
            TrackControlTransformer(),
            TrackReturnTransformer(),
            TrackParamsTransformer()
        ]

    def instrument(self, item):
        """Instrument ITEM, transforming its source code using."""
        source_lines, lineno = inspect.getsourcelines(item)
        tree = ast.parse("".join(source_lines))
        ast.increment_lineno(tree, lineno - 1)

        if self.log:
            print(f"Instrumenting {item}:")

            if self.log >= 2:
                n = lineno
                for line in source_lines:
                    print(f"{n:4} {line.rstrip()}")
                    n += 1
                print()

        for transformer in self.transformers():
            if self.log >= 3:
                print(transformer.__class__.__name__ + ':')

            transformer.visit(tree)
            ast.fix_missing_locations(tree)
            if self.log >= 3:
                print(astor.to_source(tree))

        if self.log:
            print(astor.to_source(tree))

        code = compile(tree, '<string>', 'exec')
        exec(code, globals())
        globals()[DATA_TRACKER] = self.dependency_tracker

    def restore(self):
        """Restore original code."""
        if DATA_TRACKER in globals():
            self.saved_dependencies = globals()[DATA_TRACKER]
            del globals()[DATA_TRACKER]
        super().restore()

class Slicer(Slicer):
    def dependencies(self):
        if self.saved_dependencies is None:
            return Dependencies({}, {})
        return self.saved_dependencies.dependencies()

    def code(self, *args, **kwargs):
        first = True
        for item in self.items_to_instrument:
            if not first:
                print()
            self.dependencies().code(item, *args, **kwargs)
            first = False

    def graph(self, *args, **kwargs):
        return self.dependencies().graph(*args, **kwargs)

if __name__ == "__main__":
    with Slicer(middle) as slicer:
        m = middle(2, 1, 3)
    m


if __name__ == "__main__":
    print(slicer.dependencies())


if __name__ == "__main__":
    slicer.dependencies()


if __name__ == "__main__":
    slicer.code()


if __name__ == "__main__":
    middle(2, 1, 3)


if __name__ == "__main__":
    with Slicer(middle) as middle_slicer:
        y = middle(2, 1, 3)


if __name__ == "__main__":
    middle_slicer.graph()


if __name__ == "__main__":
    print(middle_slicer.dependencies())


# ## More Examples

if __name__ == "__main__":
    print('\n## More Examples')




import math

def f():
    math.isclose

if __name__ == "__main__":
    with Slicer(f, log=2) as f_slicer:
        f()


if __package__ is None or __package__ == "":
    from Assertions import square_root
else:
    from .Assertions import square_root


if __name__ == "__main__":
    with Slicer(square_root, log=True) as root_slicer:
        y = square_root(2.0)


if __name__ == "__main__":
    root_slicer.graph()


if __name__ == "__main__":
    root_slicer.code()


if __name__ == "__main__":
    root_slicer.dependencies()


if __package__ is None or __package__ == "":
    from Intro_Debugging import remove_html_markup
else:
    from .Intro_Debugging import remove_html_markup


if __name__ == "__main__":
    with Slicer(remove_html_markup, log=True) as rhm_slicer:
        s = remove_html_markup("<foo>bar</foo>")


if __name__ == "__main__":
    rhm_slicer.graph()


if __name__ == "__main__":
    rhm_slicer.code()


if __name__ == "__main__":
    rhm_slicer.dependencies().backward_slice('tag', mode='c').graph()


def add_to(n, m):
    n += m
    return n

def mul_with(x, y):
    x *= y
    return x

def test_math():
    return mul_with(1, add_to(2, 2))

if __name__ == "__main__":
    with Slicer(add_to, mul_with, test_math) as math_slicer:
        test_math()


if __name__ == "__main__":
    math_slicer.graph()


if __name__ == "__main__":
    math_slicer.code()


if __name__ == "__main__":
    print(math_slicer.dependencies())


# ## Things that do not Work

if __name__ == "__main__":
    print('\n## Things that do not Work')




# ### Data Structures

if __name__ == "__main__":
    print('\n### Data Structures')




def test_array():
    x = [1, 2, 3]
    y = 4
    x[2] = y
    return x[2]

if __name__ == "__main__":
    with Slicer(test_array, log=True) as array_slicer:
        test_array()


if __name__ == "__main__":
    array_slicer.graph()


# ### Attributes

if __name__ == "__main__":
    print('\n### Attributes')




class X(object):
    pass

def test_attributes(y):
    x = X()
    x.attr = y
    return x.attr

if __name__ == "__main__":
    with Slicer(test_attributes, log=True) as attr_slicer:
        test_attributes(10)


if __name__ == "__main__":
    attr_slicer.graph()


if __package__ is None or __package__ == "":
    from ExpectError import ExpectError
else:
    from .ExpectError import ExpectError


if __name__ == "__main__":
    with ExpectError():
        with Slicer() as slicer:
            y = square_root(9)


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




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




# ### Exercise 1: _Title_

if __name__ == "__main__":
    print('\n### Exercise 1: _Title_')




if __name__ == "__main__":
    # Some code that is part of the exercise
    pass


if __name__ == "__main__":
    # Some code for the solution
    2 + 2


# ### Exercise 2: _Title_

if __name__ == "__main__":
    print('\n### Exercise 2: _Title_')



