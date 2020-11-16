#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Fuzzing Book".
# Web site: https://www.fuzzingbook.org/html/ClassDiagram.html
# Last change: 2020-11-16 20:31:00+01:00
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
    def foo(self):
        pass

class B_Class(A_Class):
    def foo(self):
        pass
    def bar(self):
        pass

class C_Class:
    def qux(self):
        pass

class D_Class(B_Class, C_Class):
    def foo(self):
        B_Class.foo(self)


if __name__ == "__main__":
    class_hierarchy(A_Class)


# ## Getting a Class Tree

if __name__ == "__main__":
    print('\n## Getting a Class Tree')




if __name__ == "__main__":
    D_Class.__bases__


def class_tree(cls):
    ret = []
    for base in cls.__bases__:
        if base.__name__ == cls.__name__:
            ret += class_tree(base)
        else:
            ret.append((cls, class_tree(base)))
    return ret

def class_tree(cls):
    
    def base_tree(base):
        while base.__name__ == cls.__name__:
            base = base.__bases__[0]
        return class_tree(base)
    
    ret = []
    for base in cls.__bases__:
        ret.append((cls, base_tree(base)))
    return ret

if __name__ == "__main__":
    class_tree(D_Class)


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
    return [(name, method) for (name, method) in class_methods(cls) if method.__qualname__.startswith(cls.__name__)]

def doc_class_methods(cls):
    return [(name, method) for (name, method) in public_class_methods(cls) if method.__doc__ is not None]

if __name__ == "__main__":
    public_class_methods(D_Class)


if __name__ == "__main__":
    doc_class_methods(D_Class)


# ## Drawing Class Hierarchy with Method Names

if __name__ == "__main__":
    print('\n## Drawing Class Hierarchy with Method Names')




def display_class_hierarchy(classes, include_methods=True,
                            project='fuzzingbook'):
    from graphviz import Digraph

    if project == 'debuggingbook':
        CLASS_FONT = 'Raleway, Helvetica, Arial, sans-serif'
        CLASS_COLOR = 'purple'
    else:
        CLASS_FONT = 'Patua One, Helvetica, sans-serif'
        CLASS_COLOR = '#B03A2E'

    METHOD_FONT = "'Fira Mono', 'Source Code Pro', monospace"
    METHOD_COLOR = 'black'

    if isinstance(classes, list):
        starting_class = classes[0]
    else:
        starting_class = classes
        classes = [starting_class]

    dot = Digraph(comment=starting_class.__name__ + " hierarchy")
    dot.attr('node', shape='record', fontname=CLASS_FONT)
    dot.attr('graph', rankdir='BT')
    dot.attr('edge', arrowhead='empty')
    edges = set()

    def method_string(method_name, f):
        method_string = f'<font face="{METHOD_FONT}" point-size="10">'
        if f.__doc__ is not None:
            method_string += '<b>' + method_name + '()</b>'
        else:
            method_string += f'<font color="{METHOD_COLOR}">{method_name}()</font>'
        method_string += '</font>'
        return method_string

    def class_methods_string(cls):
        methods = public_class_methods(cls)
        # return "<br/>".join([name + "()" for (name, f) in methods])

        methods_string = '<table border="0" cellpadding="0" cellspacing="0" align="left">'
        for doc in [True, False]:
            for (name, f) in methods:
                if (doc and f.__doc__ is not None) or (not doc and f.__doc__ is None):
                    methods_string += '<tr><td align="left" border="0">'
                    methods_string += method_string(name, f)
                    methods_string += '</td></tr>'
        methods_string += '</table>'
        return methods_string

    def display_class_node(cls):
        name = cls.__name__
        url = cls.__module__ + '.ipynb'
        if include_methods:
            methods = class_methods_string(cls)
            spec = '<{<b><font color="' + CLASS_COLOR + '">' + \
                cls.__name__ + '</font></b>|' + methods + '}>'
        else:
            spec = '<' + cls.__name__ + '>'
        dot.node(name, spec, href=url)

    def display_class_tree(trees):
        for tree in trees:
            (cls, subtrees) = tree
            display_class_node(cls)
            for subtree in subtrees:
                (subcls, _) = subtree
                if (cls, subcls) not in edges:
                    dot.edge(cls.__name__, subcls.__name__)
                    edges.add((cls, subcls))
            display_class_tree(subtrees)

    for cls in classes:
        tree = class_tree(cls)
        display_class_tree(tree)

    return dot

if __name__ == "__main__":
    display_class_hierarchy([D_Class, A_Class], project='debuggingbook')


if __name__ == "__main__":
    display_class_hierarchy([D_Class, A_Class], project='fuzzingbook')


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




if __name__ == "__main__":
    display_class_hierarchy(D_Class)


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



