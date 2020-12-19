#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Fuzzing Book".
# Web site: https://www.fuzzingbook.org/html/Slicer.html
# Last change: 2020-12-19 19:29:41+01:00
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




# ### Wrapping Values

if __name__ == "__main__":
    print('\n### Wrapping Values')




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


# ### Tracking Data Accesses

if __name__ == "__main__":
    print('\n### Tracking Data Accesses')




# ## A Data Store

if __name__ == "__main__":
    print('\n## A Data Store')




class DataStore(dict):
    def __init__(self, *args, log=False):
        """Initialize. If LOG is set, turn on logging."""
        super().__init__(*args)
        self.log = log

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
        return frame.f_code.co_name, frame.f_lineno

class DataStore(DataStore):
    def __setitem__(self, name, value):
        """Set NAME to VALUE."""
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: setting {name}")

        return super().__setitem__(name, value)

class DataStore(DataStore):
    def __getitem__(self, name):
        """Return NAME.
        If NAME is not stored, return a local variable NAME.
        If there is no local variable NAME, 
        return the global variable NAME."""

        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: getting {name}")

        if name in self:
            return super().__getitem__(name)

        frame = self.caller_frame()
        if name in frame.f_locals:
            return frame.f_locals[name]

        if name in globals():
            return globals()[name]
        
        raise NameError(f"name {repr(name)} is not defined")

class DataStore(DataStore):
    def __repr__(self):
        return super().__repr__()

if __name__ == "__main__":
    _test_data = DataStore(log=True)
    _test_data['x'] = 1


if __name__ == "__main__":
    _test_data['x']


if __name__ == "__main__":
    y = 3
    _test_data['y']


if __package__ is None or __package__ == "":
    from ExpectError import ExpectError
else:
    from .ExpectError import ExpectError


if __name__ == "__main__":
    with ExpectError():
        _test_data['z']


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

class TrackAccessTransformer(NodeTransformer):
    def visit_Name(self, node):
        if node.id in dir(__builtins__):
            # Do not change built-in names
            return node

        return make_data_access(node.id, node.ctx)

from ast import Module, Subscript, Name, Load, Store, \
    Assign, Attribute, With, withitem, Return, Index, Str, Call, Expr

DATA_STORE = '_data'

def make_data_access(id, ctx):
    return Subscript(
        value=Name(id=DATA_STORE, ctx=Load()),
        slice=Index(value=Str(s=id)),
        ctx=ctx
    )

if __name__ == "__main__":
    show_ast(Module(body=[make_data_access("x", Load())]))


if __name__ == "__main__":
    print(ast.dump(ast.parse("_data['x']")))


if __name__ == "__main__":
    TrackAccessTransformer().visit(middle_tree);


def dump_tree(tree):
    print(astor.to_source(tree))
    ast.fix_missing_locations(tree)  # Must run this before compiling
    _ = compile(tree, '<string>', 'exec')

if __name__ == "__main__":
    dump_tree(middle_tree)


class TreeTester(object):
    def __init__(self, tree, func):
        self.code = compile(tree, '<string>', 'exec')
        self.func = func

    def __enter__(self):
        globals()[DATA_STORE] = DataStore(log=True)
        exec(self.code, globals())
    
    def __exit__(self, exc_type, exc_value, traceback):
        globals()[self.func.__name__] = self.func
        del globals()[DATA_STORE]

if __name__ == "__main__":
    print_content(inspect.getsource(middle), '.py', start_line_number=1)


if __name__ == "__main__":
    with TreeTester(middle_tree, middle):
        middle(2, 1, 3)


if __name__ == "__main__":
    middle(2, 1, 3)


# ### Excursion: Tracking Return Values

if __name__ == "__main__":
    print('\n### Excursion: Tracking Return Values')




class TrackReturnTransformer(NodeTransformer):
    def __init__(self):
        self.function_name = None
        super().__init__()

    def visit_FunctionDef(self, node):
        self.function_name = node.name  # Save current function name
        self.generic_visit(node)
        return node

    def return_value(self):
        if self.function_name is None:
            return "<return value>"
        else:
            return f"<{self.function_name}() return value>"

    def visit_Return(self, node):
        assign_node = Assign(
                targets=[make_data_access(self.return_value(), Store())],
                value=node.value
            )
        return_node = Return(
                value=make_data_access(self.return_value(), Load())
            )
        ast.copy_location(assign_node, node)
        ast.copy_location(return_node, node)

        return [
            assign_node,
            return_node
        ]

if __name__ == "__main__":
    TrackReturnTransformer().visit(middle_tree)
    dump_tree(middle_tree)


if __name__ == "__main__":
    with TreeTester(middle_tree, middle):
        middle(2, 1, 3)


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ### Excursion: Tracking Control

if __name__ == "__main__":
    print('\n### Excursion: Tracking Control')




class TrackControlTransformer(NodeTransformer):
    def make_with(self, block):
        if len(block) == 0:
            return []

        return [With(
            items=[
                withitem(
                    context_expr=Name(id=DATA_STORE, ctx=Load()),
                    optional_vars=None)
            ],
            body=block
        )]

    def make_test(self, test):
        return Call(func=Attribute(value=Name(id=DATA_STORE, ctx=Load()),
                                   attr='test',
                                   ctx=Load()),
                     args=[test],
                     keywords=[])

    def visit_If(self, node):
        self.generic_visit(node)
        node.test = self.make_test(node.test)
        node.body = self.make_with(node.body)
        node.orelse = self.make_with(node.orelse)
        return node

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
        node.body = self.make_with(node.body)
        node.orelse = self.make_with(node.orelse)
        return node

if __name__ == "__main__":
    TrackControlTransformer().visit(middle_tree)
    dump_tree(middle_tree)


class DataStore(DataStore):
    def test(self, cond):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: testing condition")

        return cond

class DataStore(DataStore):
    def __enter__(self):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: entering block")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: exiting block")

if __name__ == "__main__":
    with TreeTester(middle_tree, middle):
        middle(2, 1, 3)


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ### Excursion: Tracking Calls and Arguments

if __name__ == "__main__":
    print('\n### Excursion: Tracking Calls and Arguments')




class TrackCallTransformer(NodeTransformer):
    def make_call(self, node, fun):
        return Call(func=Attribute(value=Name(id=DATA_STORE,
                                              ctx=Load()),
                                   attr=fun,
                                   ctx=Load()),
                     args=[node],
                     keywords=[])

    def visit_Call(self, node):
        self.generic_visit(node)

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
    TrackCallTransformer().visit(call_tree);


if __name__ == "__main__":
    dump_tree(call_tree)


class DataStore(DataStore):
    def arg(self, value):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: pushing arg")

        return value

class DataStore(DataStore):
    def ret(self, value):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: returned from call")

        return value

class DataStore(DataStore):
    def call(self, func):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: calling {func}")

        return func

if __name__ == "__main__":
    with TreeTester(call_tree, test_call):
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
                    func=Attribute(value=Name(id=DATA_STORE, ctx=Load()),
                                   attr='param', ctx=Load()),
                    args=[Str(s=arg), Name(id=arg, ctx=Load())],
                    keywords=[]
                )
            )
            create_stmts.append(create_stmt)
        create_stmts.reverse()

        node.body = create_stmts + node.body
        return node

if __name__ == "__main__":
    TrackParamsTransformer().visit(middle_tree)
    dump_tree(middle_tree)


class DataStore(DataStore):
    def param(self, name, value):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: initializing {name}")

        return self.__setitem__(name, value)

if __name__ == "__main__":
    with TreeTester(middle_tree, middle):
        middle(2, 1, 3)


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




if __name__ == "__main__":
    dump_tree(middle_tree)


if __name__ == "__main__":
    with TreeTester(middle_tree, middle):
        middle(2, 1, 3)


# ## Tracking Dependencies

if __name__ == "__main__":
    print('\n## Tracking Dependencies')




class DataTracker(DataStore):
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




class DataTracker(DataTracker):
    def __getitem__(self, name):
        self.check_location()
        self.last_read.append(name)
        return super().__getitem__(name)
    
    def check_location(self):
        pass  # More on that below

if __name__ == "__main__":
    x = 5
    y = 3


if __name__ == "__main__":
    _test_data = DataTracker()
    _test_data['x'] + _test_data['y']


if __name__ == "__main__":
    _test_data.last_read


# #### Checking Locations

if __name__ == "__main__":
    print('\n#### Checking Locations')




class DataTracker(DataTracker):
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

class DataTracker(DataTracker):
    def ignore_next_location_change(self):
        self._ignore_location_change = True

    def ignore_location_change(self):
        self.last_checked_location = self.caller_location()

if __name__ == "__main__":
    _test_data = DataTracker()


if __name__ == "__main__":
    _test_data['x'] + _test_data['y']


if __name__ == "__main__":
    _test_data.last_read


if __name__ == "__main__":
    a = 42
    b = -1
    _test_data['a'] + _test_data['b']


if __name__ == "__main__":
    _test_data.last_read


# #### Setting Variables

if __name__ == "__main__":
    print('\n#### Setting Variables')




import itertools

class DataTracker(DataTracker):
    def __setitem__(self, name, value):

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
        ret = super().__setitem__(name, value)
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
        self.clear_read()

        return ret

    def dependencies(self):
        return Dependencies(self.data_dependencies,
                            self.control_dependencies)

if __name__ == "__main__":
    _test_data = DataTracker()
    _test_data['x'] = 1
    _test_data['y'] = _test_data['x']
    _test_data['z'] = _test_data['x'] + _test_data['y']


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




class DataTracker(DataTracker):
    TEST = '<test>'

    def test(self, value):
        self.__setitem__(self.TEST, value)
        self.__getitem__(self.TEST)
        return super().test(value)

class DataTracker(DataTracker):
    def __enter__(self):
        self.control.append(self.last_read)
        self.clear_read()
        super().__enter__()

class DataTracker(DataTracker):
    def __exit__(self, exc_type, exc_value, traceback):
        self.clear_read()
        self.last_read = self.control.pop()
        self.ignore_next_location_change()
        super().__exit__(exc_type, exc_value, traceback)

if __name__ == "__main__":
    _test_data = DataTracker()
    _test_data['x'] = 1
    _test_data['y'] = _test_data['x']

    if _test_data.test(_test_data['x'] >= _test_data['y']):
        with _test_data:
            _test_data['z'] = _test_data['x'] + _test_data['y']


if __name__ == "__main__":
    _test_data.control_dependencies


if __name__ == "__main__":
    # ignore
    _test_data.dependencies().graph()


# ### Calls and Returns

if __name__ == "__main__":
    print('\n### Calls and Returns')




class DataTracker(DataTracker):
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
            if var.startswith("<"):
                ret_name = var

        self.last_read = self.data.pop()
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




class DataTracker(DataTracker):
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

                if dep_name == DataTracker.TEST:
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

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore sources"""
        self.restore()

    def restore(self):
        for item in self.items_to_instrument:
            globals()[item.__name__] = item

    def instrument(self, item):
        if self.log:
            print("Instrumenting", item)

if __name__ == "__main__":
    with Instrumenter(middle) as ins:
        pass


class Slicer(Instrumenter):
    def __init__(self, *items_to_instrument, 
                 data_tracker=None,
                 log=False):
        super().__init__(*items_to_instrument, log=log)
        if len(items_to_instrument) == 0:
            raise ValueError("Need one or more items to instrument")

        if data_tracker is None:
            data_tracker = DataTracker(log=(log > 1))
        self.data_tracker = data_tracker
        self.data_store = None

    def instrument(self, item):
        source_lines, lineno = inspect.getsourcelines(item)
        tree = ast.parse("".join(source_lines))
        ast.increment_lineno(tree, lineno - 1)

        TrackAccessTransformer().visit(tree)
        TrackCallTransformer().visit(tree)
        TrackControlTransformer().visit(tree)
        TrackReturnTransformer().visit(tree)
        TrackParamsTransformer().visit(tree)

        ast.fix_missing_locations(tree)
        # print_ast_ids(tree)

        if self.log:
            print(f"Instrumenting {item}:")

            if self.log > 1:
                n = lineno
                for line in source_lines:
                    print(f"{n:4} {line.rstrip()}")
                    n += 1
                print()

            print(astor.to_source(tree))

        code = compile(tree, '<string>', 'exec')
        exec(code, globals())
        globals()[DATA_STORE] = self.data_tracker

    def restore(self):
        if DATA_STORE in globals():
            self.data_store = globals()[DATA_STORE]
            del globals()[DATA_STORE]
        super().restore()

    def dependencies(self):
        if self.data_store is None:
            return Dependencies({}, {})
        return self.data_store.dependencies()

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




if __package__ is None or __package__ == "":
    from Assertions import square_root
else:
    from .Assertions import square_root


import math

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



