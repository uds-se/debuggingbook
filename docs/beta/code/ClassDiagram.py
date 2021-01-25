#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Debugging Book".
# Web site: https://www.debuggingbook.org/html/ClassDiagram.html
# Last change: 2021-01-25 14:14:02+01:00
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


# # Class Diagrams

if __name__ == "__main__":
    print('# Class Diagrams')




if __name__ == "__main__":
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




# ## Getting a Class Hierarchy

if __name__ == "__main__":
    print('\n## Getting a Class Hierarchy')




def class_hierarchy(cls):
    superclasses = cls.mro()
    hierarchy = []
    last_superclass_name = ""

    for superclass in superclasses:
        if superclass.__name__ != last_superclass_name:
            hierarchy.append(superclass)
            last_superclass_name = superclass.__name__

    return hierarchy

class A_Class:
    """A Class which does A thing right.
    Comes with a longer docstring."""

    def foo(self):
        """The Adventures of the glorious Foo"""
        pass

    def quux(self):
        """A method that is not used."""
        pass

class A_Class(A_Class):
    # We define another function in a separate cell.

    def second(self):
        pass

class B_Class(A_Class):
    """A subclass inheriting some methods."""

    def foo(self, fight=False):
        """A WW2 foo fighter."""
        pass

    def bar(self, qux=None, bartender=42):
        """A qux walks into a bar.
        `bartender` is an optional attribute."""
        pass

class C_Class:
    """A class injecting some method"""

    def qux(self):
        pass

class D_Class(B_Class, C_Class):
    """A subclass inheriting from multiple superclasses.
    Comes with a fairly long, but meaningless documentation."""

    def foo(self):
        B_Class.foo(self)

class D_Class(D_Class):
    pass  # An incremental addiiton that should not impact D's semantics

if __name__ == "__main__":
    class_hierarchy(D_Class)


# ## Getting a Class Tree

if __name__ == "__main__":
    print('\n## Getting a Class Tree')




if __name__ == "__main__":
    D_Class.__bases__


def class_tree(cls, lowest=None):
    ret = []
    for base in cls.__bases__:
        if base.__name__ == cls.__name__:
            if not lowest:
                lowest = cls
            ret += class_tree(base, lowest)
        else:
            if lowest:
                cls = lowest
            ret.append((cls, class_tree(base)))

    return ret

if __name__ == "__main__":
    class_tree(D_Class)


if __name__ == "__main__":
    class_tree(D_Class)[0][0]


if __name__ == "__main__":
    assert class_tree(D_Class)[0][0] == D_Class


def class_set(classes):
    if not isinstance(classes, list):
        classes = [classes]

    ret = set()

    def traverse_tree(tree):
        for (cls, subtrees) in tree:
            ret.add(cls)
            for subtree in subtrees:
                traverse_tree(subtrees)

    for cls in classes:
        traverse_tree(class_tree(cls))

    return ret

if __name__ == "__main__":
    class_set(D_Class)


if __name__ == "__main__":
    assert A_Class in class_set(D_Class)


if __name__ == "__main__":
    assert B_Class in class_set(D_Class)


if __name__ == "__main__":
    assert C_Class in class_set(D_Class)


if __name__ == "__main__":
    assert D_Class in class_set(D_Class)


if __name__ == "__main__":
    class_set([B_Class, C_Class])


# ## Getting methods

if __name__ == "__main__":
    print('\n## Getting methods')




import inspect

def class_methods(cls):
    def _class_methods(cls):
        all_methods = inspect.getmembers(cls, lambda m: inspect.isfunction(m))
        for base in cls.__bases__:
            all_methods += _class_methods(base)

        return all_methods

    unique_methods = []
    methods_seen = set()
    for (name, fun) in _class_methods(cls):
        if name not in methods_seen:
            unique_methods.append((name, fun))
            methods_seen.add(name)

    return unique_methods

if __name__ == "__main__":
    class_methods(D_Class)


def public_class_methods(cls):
    return [(name, method) for (name, method) in class_methods(cls) 
            if method.__qualname__.startswith(cls.__name__)]

def doc_class_methods(cls):
    return [(name, method) for (name, method) in public_class_methods(cls) 
            if method.__doc__ is not None]

if __name__ == "__main__":
    public_class_methods(D_Class)


if __name__ == "__main__":
    doc_class_methods(D_Class)


def overloaded_class_methods(classes):
    all_methods = {}
    for cls in class_set(classes):
        for (name, method) in class_methods(cls):
            if method.__qualname__.startswith(cls.__name__):
                all_methods.setdefault(name, set())
                all_methods[name].add(cls)

    return [name for name in all_methods if len(all_methods[name]) >= 2]

if __name__ == "__main__":
    overloaded_class_methods(D_Class)


# ### Getting Docs

if __name__ == "__main__":
    print('\n### Getting Docs')




if __name__ == "__main__":
    A_Class.__doc__


if __name__ == "__main__":
    A_Class.__bases__[0].__doc__


if __name__ == "__main__":
    A_Class.__bases__[0].__name__


def docstring(cls):
    while not cls.__doc__:
        superclass = None
        for base in cls.__bases__:
            if base.__name__ == cls.__name__:
                superclass = base
                break

        if not superclass:
            return None

        cls = superclass

    return cls.__doc__

if __name__ == "__main__":
    docstring(A_Class)


import html

import re

def escape(text):
    text = html.escape(text)
    assert '<' not in text
    assert '>' not in text
    text = text.replace('{', '&#x7b;')
    text = text.replace('|', '&#x7c;')
    text = text.replace('}', '&#x7d;')
    return text

if __name__ == "__main__":
    escape("f(foo={})")


def escape_doc(docstring):
    DOC_INDENT = 4
    docstring = "&#x0a;".join(
        ' ' * DOC_INDENT + escape(line).strip()
        for line in docstring.split('\n')
    )
    return docstring

if __name__ == "__main__":
    print(escape_doc("'Hello\n    {You|Me}'"))


# ## Drawing Class Hierarchy with Method Names

if __name__ == "__main__":
    print('\n## Drawing Class Hierarchy with Method Names')




from inspect import signature

import warnings

def display_class_hierarchy(classes, 
                            public_methods=None,
                            abstract_classes=None,
                            include_methods=True,
                            include_legend=True,
                            project='fuzzingbook',
                            log=False):
    """Visualize a class hierarchy.
`classes` is a Python class (or a list of classes) to be visualized.
`public_methods`, if given, is a list of methods to be shown as "public" (bold).
  (Default: all methods with a docstring)
`include_methods`: if True, include all methods (default)
`include_legend`: if True, include a legend (default)
    """
    from graphviz import Digraph

    if project == 'debuggingbook':
        CLASS_FONT = 'Raleway, Helvetica, Arial, sans-serif'
        CLASS_COLOR = '#6A0DAD'  # HTML 'purple'
    else:
        CLASS_FONT = 'Patua One, Helvetica, sans-serif'
        CLASS_COLOR = '#B03A2E'

    METHOD_FONT = "'Fira Mono', 'Source Code Pro', 'Courier', monospace"
    METHOD_COLOR = 'black'

    if isinstance(classes, list):
        starting_class = classes[0]
    else:
        starting_class = classes
        classes = [starting_class]

    title = starting_class.__name__ + " class hierarchy"

    dot = Digraph(comment=title)
    dot.attr('node', shape='record', fontname=CLASS_FONT)
    dot.attr('graph', rankdir='BT', tooltip=title)
    dot.attr('edge', arrowhead='empty')
    edges = set()
    overloaded_methods = set()

    drawn_classes = set()

    def method_string(method_name, public, overloaded, fontsize=10):
        method_string = f'<font face="{METHOD_FONT}" point-size="{str(fontsize)}">'

        if overloaded:
            name = f'<i>{method_name}()</i>'
        else:
            name = f'{method_name}()'

        if public:
            method_string += f'<b>{name}</b>'
        else:
            method_string += f'<font color="{METHOD_COLOR}">' \
                             f'{name}</font>'

        method_string += '</font>'
        return method_string

    def is_overloaded(method_name, f):
        return (method_name in overloaded_methods or
                (f.__doc__ is not None and "in subclasses" in f.__doc__))

    def is_abstract(cls):
        if not abstract_classes:
            return False

        return (cls in abstract_classes or
                any(c.__name__ == cls.__name__ for c in abstract_classes))

    def is_public(method_name, f):
        if public_methods:
            return (method_name in public_methods or
                    f in public_methods or
                    any(f.__qualname__ == m.__qualname__
                        for m in public_methods))

        return f.__doc__ is not None

    def class_methods_string(cls, url):
        methods = public_class_methods(cls)
        # return "<br/>".join([name + "()" for (name, f) in methods])
        if len(methods) == 0:
            return ""

        methods_string = f'<table border="0" cellpadding="0" ' \
                         f'cellspacing="0" ' \
                         f'align="left" tooltip="{cls.__name__}" href="#">'

        for public in [True, False]:
            for (name, f) in methods:
                if public != is_public(name, f):
                    continue

                if log:
                    print(f"    Drawing {name}()")

                if is_public(name, f) and not f.__doc__:
                    warnings.warn(f"{f.__qualname__}() is listed as public,"
                                  f" but has no docstring")

                overloaded = is_overloaded(name, f)

                method_doc = escape(name + str(inspect.signature(f)))
                if f.__doc__:
                    method_doc += ":&#x0a;" + escape_doc(f.__doc__)

                # Tooltips are only shown if a href is present, too
                tooltip = f' tooltip="{method_doc}"'
                href = f' href="{url}"'
                methods_string += f'<tr><td align="left" border="0"' \
                                  f'{tooltip}{href}>'

                methods_string += method_string(name, public, overloaded)

                methods_string += '</td></tr>'

        methods_string += '</table>'
        return methods_string

    def display_class_node(cls):
        name = cls.__name__

        if name in drawn_classes:
            return
        drawn_classes.add(name)

        if log:
            print(f"Drawing class {name}")

        if cls.__module__ == '__main__':
            url = '#'
        else:
            url = cls.__module__ + '.ipynb'

        if is_abstract(cls):
            formatted_class_name = f'<i>{cls.__name__}</i>'
        else:
            formatted_class_name = cls.__name__

        if include_methods:
            methods = class_methods_string(cls, url)
            spec = '<{<b><font color="' + CLASS_COLOR + '">' + \
                formatted_class_name + '</font></b>|' + methods + '}>'
        else:
            spec = '<' + formatted_class_name + '>'

        class_doc = escape('class ' + cls.__name__)
        if docstring(cls):
            class_doc += ':&#x0a;' + escape_doc(docstring(cls))
        else:
            warnings.warn(f"Class {cls.__name__} has no docstring")

        dot.node(name, spec, tooltip=class_doc, href=url)

    def display_class_trees(trees):
        for tree in trees:
            (cls, subtrees) = tree
            display_class_node(cls)

            for subtree in subtrees:
                (subcls, _) = subtree

                if (cls.__name__, subcls.__name__) not in edges:
                    dot.edge(cls.__name__, subcls.__name__)
                    edges.add((cls.__name__, subcls.__name__))

            display_class_trees(subtrees)

    def display_legend():
        fontsize = 8.0

        label = f'<b><font color="{CLASS_COLOR}">Legend</font></b><br align="left"/>' 

        for item in [
            method_string("public_method",
                          public=True, overloaded=False, fontsize=fontsize),
            method_string("private_method",
                          public=False, overloaded=False, fontsize=fontsize),
            method_string("overloaded_method",
                          public=False, overloaded=True, fontsize=fontsize)
        ]:
            label += '&bull;&nbsp;' + item + '<br align="left"/>'

        label += f'<font face="Helvetica" point-size="{str(fontsize + 1)}">' \
                 'Hover over names to see doc' \
                 '</font><br align="left"/>'

        dot.node('Legend', label=f'<{label}>', shape='plain', fontsize=str(fontsize + 2))

    for cls in classes:
        tree = class_tree(cls)
        overloaded_methods = overloaded_class_methods(cls)
        display_class_trees(tree)

    if include_legend:
        display_legend()

    return dot

if __name__ == "__main__":
    display_class_hierarchy(D_Class, project='debuggingbook')


if __name__ == "__main__":
    display_class_hierarchy(D_Class, project='fuzzingbook')


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




if __name__ == "__main__":
    display_class_hierarchy(D_Class)


if __name__ == "__main__":
    display_class_hierarchy([A_Class, B_Class],
                            abstract_classes=[A_Class],
                            public_methods=[
                                A_Class.quux,
                            ], log=True)

