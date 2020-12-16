#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Fuzzing Book".
# Web site: https://www.fuzzingbook.org/html/Slicer.html
# Last change: 2020-12-16 19:01:58+01:00
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


class Dependencies(object):
    def __init__(self, data, control):
        self.data = data
        self.control = control
        
    def validate(self):
        pass

# ## Drawing Dependencies

if __name__ == "__main__":
    print('\n## Drawing Dependencies')




if __name__ == "__main__":
    from graphviz import Digraph, nohtml


import html

import inspect

class Dependencies(Dependencies):
    def source(self, node):
        (name, location) = node
        code_name, lineno = location
        fun = globals()[code_name]
        source_lines, first_lineno = inspect.getsourcelines(fun)

        try:
            line = source_lines[lineno - first_lineno].strip()
        except IndexError:
            return ''

        return line

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
    def id(self, node):
        id = ""
        for c in repr(node):
            if c.isalnum() or c == '_':
                id += c
            if c == ':' or c == ',':
                id += '_'
        return id

    def label(self, node):
        (name, location) = node
        source = self.source(node)

        title = html.escape(name)
        if name.startswith('<'):
            title = f'<I>{title}</I>'

        return (f'<'
                f'<B>{title}</B>'
                f'<FONT POINT-SIZE="9.0"><BR/><BR/>'
                f'{html.escape(source)}'
                f'</FONT>'
                f'>')

    def tooltip(self, node):
        (name, location) = node
        code_name, lineno = location
        return f"{code_name}:{lineno}"

class Dependencies(Dependencies):
    def graph(self):
        """Draw dependencies."""
        self.validate()

        g = self.make_graph()
        self.draw_dependencies(g)
        self.add_hierarchy(g)
        return g

class Dependencies(Dependencies):
    def draw_dependencies(self, g):
        for node in self.data:
            g.node(self.id(node), 
                   label=self.label(node),
                   tooltip=self.tooltip(node))

            for source in self.data[node]:
                g.edge(self.id(source), self.id(node))

            for source in self.control[node]:
                g.edge(self.id(source), self.id(node),
                       style='dashed', color='grey')

class Dependencies(Dependencies):
    def all_vars(self):
        all_vars = set()
        for node in self.data:
            all_vars.add(node)

            for source in self.data[node]:
                all_vars.add(source)

            for source in self.control[node]:
                all_vars.add(source)

        return all_vars

class Dependencies(Dependencies):
    def all_codes(self):
        code_names = {}
        for node in self.all_vars():
            (name, location) = node
            code_name, lineno = location
            if code_name not in code_names:
                code_names[code_name] = []
            code_names[code_name].append((lineno, node))

        for code_name in code_names:
            code_names[code_name].sort()

        return code_names

class Dependencies(Dependencies):
    def add_hierarchy(self, g):
        """Add invisible edges for a proper hierarchy."""
        code_names = self.all_codes()
        for code_name in code_names:
            last_node = None
            last_lineno = 0
            for (lineno, node) in code_names[code_name]:
                if last_node is not None and lineno > last_lineno:
                    g.edge(self.id(last_node),
                           self.id(node),
                           style='invis')

                last_node = node
                last_lineno = lineno

        return g

class Dependencies(Dependencies):
    def expand_items(self, items):
        all_items = []
        for item in items:
            if isinstance(item, str):
                for node in self.all_vars():
                    (name, location) = node
                    if name == item:
                        all_items.append(node)
            else:
                all_items.append(item)

        return all_items

    def backward_slice(self, *items, mode="cd"):
        data = {}
        control = {}
        queue = self.expand_items(items)
        seen = set()

        while len(queue) > 0:
            var = queue[0]
            queue = queue[1:]
            seen.add(var)

            if 'd' in mode:
                data[var] = self.data[var]
                for next_var in data[var]:
                    if next_var not in seen:
                        queue.append(next_var)
            else:
                data[var] = set()

            if 'c' in mode:
                control[var] = self.control[var]
                for next_var in control[var]:
                    if next_var not in seen:
                        queue.append(next_var)
            else:
                control[var] = set()

        return Dependencies(data, control)

if __name__ == "__main__":
    middle_deps = Dependencies({('z', ('middle', 1)): set(), ('y', ('middle', 1)): set(), ('x', ('middle', 1)): set(), ('<test>', ('middle', 2)): {('y', ('middle', 1)), ('z', ('middle', 1))}, ('<test>', ('middle', 3)): {('y', ('middle', 1)), ('x', ('middle', 1))}, ('<test>', ('middle', 5)): {('z', ('middle', 1)), ('x', ('middle', 1))}, ('<middle() return value>', ('middle', 6)): {('y', ('middle', 1))}}, {('z', ('middle', 1)): set(), ('y', ('middle', 1)): set(), ('x', ('middle', 1)): set(), ('<test>', ('middle', 2)): set(), ('<test>', ('middle', 3)): {('<test>', ('middle', 2))}, ('<test>', ('middle', 5)): {('<test>', ('middle', 3))}, ('<middle() return value>', ('middle', 6)): {('<test>', ('middle', 5))}})


# ### Data Dependencies

if __name__ == "__main__":
    print('\n### Data Dependencies')




if __name__ == "__main__":
    middle_deps.backward_slice('<middle() return value>', mode='d').graph()


# ### Control Dependencies

if __name__ == "__main__":
    print('\n### Control Dependencies')




if __name__ == "__main__":
    middle_deps.backward_slice('<middle() return value>', mode='c').graph()


# ### All Dependencies

if __name__ == "__main__":
    print('\n### All Dependencies')




if __name__ == "__main__":
    middle_deps.graph()


# ### Slices

if __name__ == "__main__":
    print('\n### Slices')




# ## Instrumenting Code

if __name__ == "__main__":
    print('\n## Instrumenting Code')




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


import math

import inspect

if __name__ == "__main__":
    middle_tree = ast.parse(inspect.getsource(middle))
    show_ast(middle_tree)


from ast import NodeTransformer, Subscript, Name, Load, Store, \
    Assign, Attribute, With, withitem, Return, Index, Str, Call, Expr

DATA_STORE = '_data'

if __name__ == "__main__":
    print(ast.dump(ast.parse(f"{DATA_STORE}['x']")))


def make_data_access(id, ctx):
    return Subscript(
        value=Name(id=DATA_STORE, ctx=Load()),
        slice=Index(value=Str(s=id)),
        ctx=ctx
    )

class TrackAccessTransformer(NodeTransformer):
    def visit_Name(self, node):
        if node.id in dir(__builtins__):
            return node  # Do not change built-in names
        return make_data_access(node.id, node.ctx)

def dump_tree(tree):
    ast.fix_missing_locations(tree)
    print(astor.to_source(tree))
    code = compile(tree, '<string>', 'exec')

if __name__ == "__main__":
    TrackAccessTransformer().visit(middle_tree)
    dump_tree(middle_tree)


if __name__ == "__main__":
    print(ast.dump(ast.parse(f"{DATA_STORE}.param('x', x)")))


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


class TrackReturnTransformer(NodeTransformer):
    def __init__(self):
        self.function_name = None
        super().__init__()

    def visit_FunctionDef(self, node):
        self.function_name = node.name
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

    def visit_While(self, node):
        self.generic_visit(node)
        node.test = self.make_test(node.test)
        node.body = self.make_with(node.body)
        node.orelse = self.make_with(node.orelse)
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        node.body = self.make_with(node.body)
        node.orelse = self.make_with(node.orelse)
        return node

if __name__ == "__main__":
    TrackControlTransformer().visit(middle_tree)
    dump_tree(middle_tree)


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
    TrackCallTransformer().visit(call_tree)
    dump_tree(call_tree)


if __name__ == "__main__":
    ast.fix_missing_locations(call_tree)
    compile(call_tree, '<string>', 'exec')


def print_ast_ids(tree):
    for node in ast.walk(tree):
        print(node)
        try:
            print(astor.to_source(node))
        except AttributeError:
            print("(No source)\n")

# print_ast_ids(new_square_root_tree)

# ## Tracking Data

if __name__ == "__main__":
    print('\n## Tracking Data')




class DataStore(dict):
    def __init__(self, *args, log=False):
        super().__init__(*args)
        self.log = log

    def caller_location(self):
        frame = inspect.currentframe()
        while ('self' in frame.f_locals and 
               isinstance(frame.f_locals['self'], self.__class__)):
               frame = frame.f_back
        return frame.f_code.co_name, frame.f_lineno

    def __getitem__(self, name):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: getting {name}")

        if name in self:
            return super().__getitem__(name)
        else:
            return globals()[name]

    def __setitem__(self, name, value):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: setting {name}")

        return super().__setitem__(name, value)

class DataStore(DataStore):
    def test(self, cond):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: testing condition")

        return cond

    def param(self, name, value):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: initializing {name}")

        return self.__setitem__(name, value)

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
    def call(self, fun):
        if self.log:
            code_name, lineno = self.caller_location()
            print(f"{code_name}:{lineno}: calling {fun}")

        return fun

class DataStore(DataStore):
    def __repr__(self):
        return super().__repr__()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

# ## Tracking Dependencies

if __name__ == "__main__":
    print('\n## Tracking Dependencies')




import itertools

class DataTracker(DataStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origins = {}
        self.data_dependencies = {}
        self.control_dependencies = {}

        self.data = [[]]  # Data stack
        self.control = [[]]  # Control stack
        self.last_read = []
        self.last_checked_location = None
        self._ignore_location_change = False

        self.args = []  # Argument stack

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

    def ignore_next_location_change(self):
        self._ignore_location_change = True

    def ignore_location_change(self):
        self.last_checked_location = self.caller_location()

    def __getitem__(self, name):
        self.check_location()
        self.last_read.append(name)
        return super().__getitem__(name)

class DataTracker(DataTracker):
    def __setitem__(self, name, value):
        def add_dependencies(dependencies, vars_read, tp):
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

        if (name, location) not in self.data_dependencies:
            self.data_dependencies[(name, location)] = set()
        if (name, location) not in self.control_dependencies:
            self.control_dependencies[(name, location)] = set()

        add_dependencies(self.data_dependencies[(name, location)],
                         self.last_read, tp="data")
        add_dependencies(self.control_dependencies[(name, location)],
                         itertools.chain.from_iterable(self.control),
                         tp="control")

        self.origins[name] = location

        # Reset read info for next line
        self.clear_read()

        return ret

class DataTracker(DataTracker):
    TEST = '<test>'

    def test(self, value):
        self.__setitem__(self.TEST, value)
        self.__getitem__(self.TEST)
        return super().test(value)

    def __enter__(self):
        self.control.append(self.last_read)
        self.clear_read()
        super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.clear_read()
        self.last_read = self.control.pop()
        self.ignore_next_location_change()
        super().__exit__(exc_type, exc_value, traceback)

    def dependencies(self):
        return Dependencies(self.data_dependencies,
                            self.control_dependencies)

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

# #### Diagnostics

if __name__ == "__main__":
    print('\n#### Diagnostics')




import re
import sys

class Dependencies(Dependencies):
    def format_var(self, var):
        name, location = var
        location_name, lineno = location
        return(f"{name} ({location_name}:{lineno})")

class Dependencies(Dependencies):
    def validate(self):
        for var in self.all_vars():
            source = self.source(var)
            for dep_var in self.data[var] | self.control[var]:
                dep_name, dep_location = dep_var

                if dep_name == DataTracker.TEST:
                    continue

                if dep_name.endswith('return value>'):
                    if source.find('(') < 0:
                        print(f"Warning: {self.format_var(var)} "
                              f"depends on {self.format_var(dep_var)}, "
                              f"but {repr(source)} does not seem to have a call",
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

class Dependencies(Dependencies):
    def __repr__(self):
        self.validate()

        out = ""
        for var in set(self.data.keys() | set(self.control.keys())):
            out += self.format_var(var) + ":\n"
            for data_dep in self.data[var]:
                out += f"    <= {self.format_var(data_dep)}\n"
            for control_dep in self.control[var]:
                out += f"    <- {self.format_var(control_dep)}\n"

        return out

    def dump(self):
        return f"Dependencies({self.data}, {self.control})"

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
            return None
        return self.data_store.dependencies()

if __name__ == "__main__":
    with Slicer(middle) as slicer:
        m = middle(2, 1, 3)
    m


if __name__ == "__main__":
    slicer.dependencies()


if __name__ == "__main__":
    middle(2, 1, 3)


if __name__ == "__main__":
    with Slicer(middle) as middle_slicer:
        y = middle(2, 1, 3)


if __name__ == "__main__":
    middle_slicer.dependencies().graph()


if __name__ == "__main__":
    print(middle_slicer.dependencies().dump())


# ## More Examples

if __name__ == "__main__":
    print('\n## More Examples')




if __package__ is None or __package__ == "":
    from Assertions import square_root
else:
    from .Assertions import square_root


if __name__ == "__main__":
    with Slicer(square_root, log=True) as root_slicer:
        y = square_root(2.0)


if __name__ == "__main__":
    root_slicer.dependencies().graph()


if __package__ is None or __package__ == "":
    from Intro_Debugging import remove_html_markup
else:
    from .Intro_Debugging import remove_html_markup


if __name__ == "__main__":
    with Slicer(remove_html_markup) as rhm_slicer:
        s = remove_html_markup("<foo>bar</foo>")


if __name__ == "__main__":
    rhm_slicer.dependencies().graph()


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
    math_slicer.dependencies().graph()


# ## Things that do not Work

if __name__ == "__main__":
    print('\n## Things that do not Work')




# ### Multiple Assignments

if __name__ == "__main__":
    print('\n### Multiple Assignments')




def test_multiple_assignment():
    x, y = 0, 1
    t = (x * x, y * y)
    return t[x]

with Slicer(test_multiple_assignment) as multi_slicer:
    test_multiple_assignment()

if __name__ == "__main__":
    multi_slicer.dependencies().graph()


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
    attr_slicer.dependencies().graph()


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




import traceback

class tint(int):
    def __new__(cls, value, *args, **kw):
        return int.__new__(cls, value)

    def __init__(self, value, slice=None, **kwargs):
        self.slice = [self.current_location()]
        if slice is not None:
            self.slice += slice

    def current_location(self):
        frame = inspect.currentframe()
        while ('self' in frame.f_locals and 
               isinstance(frame.f_locals['self'], tint)):
            frame = frame.f_back

        return (frame.f_code.co_name, frame.f_lineno)

class tint(tint):
    def __repr__(self):
        return int.__repr__(self)

class tint(tint):
    def __str__(self):
        return int.__str__(self)

if __name__ == "__main__":
    x = tint(2)
    x


if __name__ == "__main__":
    x.slice


if __name__ == "__main__":
    x == 2


if __name__ == "__main__":
    type(x)


class tint(tint):
    def create(self, x):
        return tint(x, slice=self.slice)

class tint(tint):
    def __add__(self, x):
        return self.create(int(self) + x)
    def __radd__(self, x):
        return self.create(x + int(self))

if __name__ == "__main__":
    x = tint(2)
    x = x + 2
    type(x)


if __name__ == "__main__":
    x


class tint(tint):
    def __sub__(self, x):
        return self.create(int(self) - x)
    def __rsub__(self, x):
        return self.create(x - int(self))

class tint(tint):
    def __mul__(self, x):
        return self.create(int(self) * x)
    def __rmul__(self, x):
        return self.create(x * int(self))

class tint(tint):
    def __matmul__(self, x):
        return self.create(int(self) @ x)
    def __rmatmul__(self, x):
        return self.create(x @ int(self))

class tint(tint):
    def __truediv__(self, x):
        return self.create(int(self) / x)
    def __rtruediv__(self, x):
        return self.create(x / int(self))

class tint(tint):
    def __floordiv__(self, x):
        return self.create(int(self) // x)
    def __rfloordiv__(self, x):
        return self.create(x // int(self))

class tint(tint):
    def __mod__(self, x):
        return self.create(int(self) % x)
    def __rmod__(self, x):
        return self.create(x % int(self))

class tint(tint):
    def __divmod__(self, x):
        return self.create(divmod(int(self), x))
    def __rdivmod__(self, x):
        return self.create(divmod(x, int(self)))

class tint(tint):
    def __pow__(self, x):
        return self.create(int(self) ** x)
    def __rpow__(self, x):
        return self.create(x ** int(self))

class tint(tint):
    def __lshift__(self, x):
        return self.create(int(self) << x)
    def __rlshift__(self, x):
        return self.create(x << int(self))

class tint(tint):
    def __rshift__(self, x):
        return self.create(int(self) >> x)
    def __rrshift__(self, x):
        return self.create(x >> int(self))

class tint(tint):
    def __and__(self, x):
        return self.create(int(self) & x)
    def __rand__(self, x):
        return self.create(x & int(self))

class tint(tint):
    def __xor__(self, x):
        return self.create(int(self) ^ x)
    def __rxor__(self, x):
        return self.create(x ^ int(self))

class tint(tint):
    def __or__(self, x):
        return self.create(int(self) | x)
    def __ror__(self, x):
        return self.create(x | int(self))

class tint(tint):
    def __neg__(self):
        return self.create(-int(self))
    def __pos__(self):
        return self.create(+int(self))
    def __abs__(self):
        return self.create(abs(int(self)))
    def __invert__(self):
        return self.create(-int(self))

class tint(tint):
    def __index__(self):
        return int(self)

if __name__ == "__main__":
    x = tint(2)
    y = x + 3 - (3 + x)


if __name__ == "__main__":
    y, type(y), y.slice


if __name__ == "__main__":
    x = tint(2)
    y = tint(1)
    z = tint(3)
    m = middle(x, y, z)
    m, m.slice


if __name__ == "__main__":
    x = tint(4)
    y = square_root(x)


if __name__ == "__main__":
    y.slice


if __package__ is None or __package__ == "":
    from ExpectError import ExpectError
else:
    from .ExpectError import ExpectError


if __name__ == "__main__":
    with ExpectError():
        y = square_root(tint(2))

