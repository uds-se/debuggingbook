#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Debugging Book".
# Web site: https://www.debuggingbook.org/html/DynamicInvariants.html
# Last change: 2021-01-20 16:41:55+01:00
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


# # Mining Function Specifications

if __name__ == "__main__":
    print('# Mining Function Specifications')




if __name__ == "__main__":
    from bookutils import YouTubeVideo
    YouTubeVideo("HDu1olXFvv0")


if __name__ == "__main__":
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)


if __package__ is None or __package__ == "":
    from Tracer import Tracer
else:
    from .Tracer import Tracer


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




# ## Specifications and Assertions

if __name__ == "__main__":
    print('\n## Specifications and Assertions')




def square_root(x):
    assert x >= 0  # Precondition

    ...

    assert result * result == x  # Postcondition
    return result

# ## Beyond Generic Failures

if __name__ == "__main__":
    print('\n## Beyond Generic Failures')




if __name__ == "__main__":
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)


def square_root(x):
    """Computes the square root of x, using the Newton-Raphson method"""
    approx = None
    guess = x / 2
    while approx != guess:
        approx = guess
        guess = (approx + x / approx) / 2

    return approx

if __package__ is None or __package__ == "":
    from ExpectError import ExpectError, ExpectTimeout
else:
    from .ExpectError import ExpectError, ExpectTimeout


if __name__ == "__main__":
    with ExpectError():
        square_root("foo")


if __name__ == "__main__":
    with ExpectError():
        x = square_root(0.0)


if __name__ == "__main__":
    with ExpectTimeout(1):
        x = square_root(-1.0)


# ## Mining Data Types

if __name__ == "__main__":
    print('\n## Mining Data Types')




def square_root_with_type_annotations(x: float) -> float:
    """Computes the square root of x, using the Newton-Raphson method"""
    return square_root(x)

# ### Runtime Type Checking

if __name__ == "__main__":
    print('\n### Runtime Type Checking')




import enforce

@enforce.runtime_validation
def square_root_with_checked_type_annotations(x: float) -> float:
    """Computes the square root of x, using the Newton-Raphson method"""
    return square_root(x)

if __name__ == "__main__":
    with ExpectError():
        square_root_with_checked_type_annotations(True)


if __name__ == "__main__":
    square_root(True)


if __package__ is None or __package__ == "":
    from bookutils import quiz
else:
    from .bookutils import quiz


if __name__ == "__main__":
    quiz("What happens if we call "
         "`square_root_with_checked_type_annotations(1)`?",
        [
            "`1` is automatically converted to float. It will pass.",
            "`1` is a subtype of float. It will pass.",
            "`1` is an integer, and no float. The type check will fail.",
            "The function will fail for some other reason."
        ],
         1
        )


if __name__ == "__main__":
    with ExpectError(enforce.exceptions.RuntimeTypeError):
        square_root_with_checked_type_annotations(1)


from typing import Union, Optional

@enforce.runtime_validation
def square_root_with_union_type(x: Union[int, float]) -> float:
    """Computes the square root of x, using the Newton-Raphson method"""
    return square_root(x)

if __name__ == "__main__":
    square_root_with_union_type(2)


if __name__ == "__main__":
    square_root_with_union_type(2.0)


if __name__ == "__main__":
    with ExpectError(enforce.exceptions.RuntimeTypeError):
        square_root_with_union_type("Two dot zero")


# ### Static Type Checking

if __name__ == "__main__":
    print('\n### Static Type Checking')




import inspect
import tempfile

if __name__ == "__main__":
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.py')
    f.name


if __name__ == "__main__":
    f.write(inspect.getsource(square_root))
    f.write('\n')
    f.write(inspect.getsource(square_root_with_type_annotations))
    f.write('\n')
    f.write("print(square_root_with_type_annotations('123'))\n")
    f.flush()


if __package__ is None or __package__ == "":
    from bookutils import print_file
else:
    from .bookutils import print_file


if __name__ == "__main__":
    print_file(f.name, start_line_number=1)


import subprocess

if __name__ == "__main__":
    result = subprocess.run(["mypy", "--strict", f.name],
                            universal_newlines=True, stdout=subprocess.PIPE)


if __name__ == "__main__":
    print(result.stdout.replace(f.name + ':', ''))
    del f  # Delete temporary file


# ## Mining Type Specifications

if __name__ == "__main__":
    print('\n## Mining Type Specifications')




if __name__ == "__main__":
    y = square_root(25.0)
    y


if __name__ == "__main__":
    y = square_root(2.0)
    y


# ### Tracing Calls

if __name__ == "__main__":
    print('\n### Tracing Calls')




class CallTracer(Tracer):
    def __init__(self, log=False, **kwargs):
        super().__init__(**kwargs)
        self.log = log
        self.reset()

    def reset(self):
        self._calls = {}
        self._stack = []

class CallTracer(CallTracer):
    def traceit(self, frame, event, arg):
        """Tracking function: Record all calls and all args"""
        if event == "call":
            self.trace_call(frame, event, arg)
        elif event == "return":
            self.trace_return(frame, event, arg)

        return self.traceit

class CallTracer(CallTracer):
    def trace_call(self, frame, event, arg):
        """Save current function name and args on the stack"""
        code = frame.f_code
        function_name = code.co_name
        arguments = get_arguments(frame)
        self._stack.append((function_name, arguments))

        if self.log:
            print(simple_call_string(function_name, arguments))

def get_arguments(frame):
    """Return call arguments in the given frame"""
    # When called, all arguments are local variables
    arguments = [(var, frame.f_locals[var]) for var in frame.f_locals]
    arguments.reverse()  # Want same order as call
    return arguments

class CallTracer(CallTracer):
    def trace_return(self, frame, event, arg):
        """Get return value and store complete call with arguments and return value"""
        code = frame.f_code
        function_name = code.co_name
        return_value = arg
        # TODO: Could call get_arguments() here to also retrieve _final_ values of argument variables

        called_function_name, called_arguments = self._stack.pop()
        assert function_name == called_function_name

        if self.log:
            print(simple_call_string(function_name, called_arguments), "returns", return_value)

        self.add_call(function_name, called_arguments, return_value)

def simple_call_string(function_name, argument_list, return_value=None):
    """Return function_name(arg[0], arg[1], ...) as a string"""
    call = function_name + "(" + \
        ", ".join([var + "=" + repr(value)
                   for (var, value) in argument_list]) + ")"

    if return_value is not None:
        call += " = " + repr(return_value)

    return call

class CallTracer(CallTracer):
    def add_call(self, function_name, arguments, return_value=None):
        """Add given call to list of calls"""
        if function_name not in self._calls:
            self._calls[function_name] = []
        self._calls[function_name].append((arguments, return_value))

class CallTracer(CallTracer):
    def calls(self, function_name=None):
        """Return list of calls for function_name, 
        or a mapping function_name -> calls for all functions tracked"""
        if function_name is None:
            return self._calls

        return self._calls[function_name]

if __name__ == "__main__":
    with CallTracer(log=True) as tracer:
        y = square_root(25)
        y = square_root(2.0)


if __name__ == "__main__":
    calls = tracer.calls('square_root')
    calls


if __name__ == "__main__":
    square_root_argument_list, square_root_return_value = calls[0]
    simple_call_string('square_root', square_root_argument_list, square_root_return_value)


def hello(name):
    print("Hello,", name)

if __name__ == "__main__":
    with CallTracer() as tracer:
        hello("world")


if __name__ == "__main__":
    hello_calls = tracer.calls('hello')
    hello_calls


if __name__ == "__main__":
    hello_argument_list, hello_return_value = hello_calls[0]
    simple_call_string('hello', hello_argument_list, hello_return_value)


# ### Getting Types

if __name__ == "__main__":
    print('\n### Getting Types')




if __name__ == "__main__":
    type(4)


if __name__ == "__main__":
    type(2.0)


if __name__ == "__main__":
    type([4])


if __name__ == "__main__":
    parameter, value = square_root_argument_list[0]
    parameter, type(value)


if __name__ == "__main__":
    type(square_root_return_value)


def square_root_annotated(x: int) -> float:
    return square_root(x)

if __name__ == "__main__":
    square_root_annotated.__annotations__


# ### Annotating Functions with Types

if __name__ == "__main__":
    print('\n### Annotating Functions with Types')




# #### Excursion: Accessing Function Structure

if __name__ == "__main__":
    print('\n#### Excursion: Accessing Function Structure')




import ast
import inspect
import astor

if __name__ == "__main__":
    square_root_source = inspect.getsource(square_root)
    square_root_source


if __package__ is None or __package__ == "":
    from bookutils import print_content
else:
    from .bookutils import print_content


if __name__ == "__main__":
    print_content(square_root_source, '.py')


if __name__ == "__main__":
    square_root_ast = ast.parse(square_root_source)


if __name__ == "__main__":
    print(astor.dump_tree(square_root_ast))


if __package__ is None or __package__ == "":
    from bookutils import show_ast
else:
    from .bookutils import show_ast


if __name__ == "__main__":
    show_ast(square_root_ast)


if __name__ == "__main__":
    print_content(astor.to_source(square_root_ast), '.py')


# #### End of Excursion

if __name__ == "__main__":
    print('\n#### End of Excursion')




# #### Excursion: Annotating Functions with Given Types

if __name__ == "__main__":
    print('\n#### Excursion: Annotating Functions with Given Types')




def parse_type(name):
    class ValueVisitor(ast.NodeVisitor):
        def visit_Expr(self, node):
            self.value_node = node.value

    tree = ast.parse(name)
    name_visitor = ValueVisitor()
    name_visitor.visit(tree)
    return name_visitor.value_node

if __name__ == "__main__":
    print(astor.dump_tree(parse_type('int')))


if __name__ == "__main__":
    print(astor.dump_tree(parse_type('[object]')))


class TypeTransformer(ast.NodeTransformer):
    def __init__(self, argument_types, return_type=None):
        self.argument_types = argument_types
        self.return_type = return_type
        super().__init__()

class TypeTransformer(TypeTransformer):
    def visit_FunctionDef(self, node):
        """Add annotation to function"""
        # Set argument types
        new_args = []
        for arg in node.args.args:
            new_args.append(self.annotate_arg(arg))

        new_arguments = ast.arguments(
            new_args,
            node.args.vararg,
            node.args.kwonlyargs,
            node.args.kw_defaults,
            node.args.kwarg,
            node.args.defaults
        )

        # Set return type
        if self.return_type is not None:
            node.returns = parse_type(self.return_type)

        return ast.copy_location(
            ast.FunctionDef(node.name, new_arguments,
                            node.body, node.decorator_list,
                            node.returns), node)

class TypeTransformer(TypeTransformer):
    def annotate_arg(self, arg):
        """Add annotation to single function argument"""
        arg_name = arg.arg
        if arg_name in self.argument_types:
            arg.annotation = parse_type(self.argument_types[arg_name])
        return arg

if __name__ == "__main__":
    new_ast = TypeTransformer({'x': 'int'}, 'float').visit(square_root_ast)


if __name__ == "__main__":
    print_content(astor.to_source(new_ast), '.py')


if __name__ == "__main__":
    hello_source = inspect.getsource(hello)


if __name__ == "__main__":
    hello_ast = ast.parse(hello_source)


if __name__ == "__main__":
    new_ast = TypeTransformer({'name': 'str'}, 'None').visit(hello_ast)


if __name__ == "__main__":
    print_content(astor.to_source(new_ast), '.py')


# #### End of Excursion

if __name__ == "__main__":
    print('\n#### End of Excursion')




# #### Excursion: Annotating Functions with Mined Types

if __name__ == "__main__":
    print('\n#### Excursion: Annotating Functions with Mined Types')




def type_string(value):
    return type(value).__name__

if __name__ == "__main__":
    type_string(4)


if __name__ == "__main__":
    type_string([])


if __name__ == "__main__":
    type_string([3])


if __name__ == "__main__":
    with CallTracer() as tracer:
        y = square_root(25.0)
        y = square_root(2.0)


if __name__ == "__main__":
    tracer.calls()


def annotate_types(calls):
    annotated_functions = {}
    
    for function_name in calls:
        try:
            annotated_functions[function_name] = annotate_function_with_types(function_name, calls[function_name])
        except KeyError:
            continue

    return annotated_functions

def annotate_function_with_types(function_name, function_calls):
    function = globals()[function_name]  # May raise KeyError for internal functions
    function_code = inspect.getsource(function)
    function_ast = ast.parse(function_code)
    return annotate_function_ast_with_types(function_ast, function_calls)

from typing import Any

def annotate_function_ast_with_types(function_ast, function_calls):
    parameter_types = {}
    return_type = None

    for calls_seen in function_calls:
        args, return_value = calls_seen
        if return_value:
            if return_type and return_type != type_string(return_value):
                return_type = 'Any'
            else:
                return_type = type_string(return_value)

        for parameter, value in args:
            try:
                different_type = (parameter_types[parameter] !=
                                  type_string(value))
            except KeyError:
                different_type = False

            if different_type:
                parameter_types[parameter] = 'Any'
            else:
                parameter_types[parameter] = type_string(value)

    annotated_function_ast = \
        TypeTransformer(parameter_types, return_type).visit(function_ast)

    return annotated_function_ast

if __name__ == "__main__":
    print_content(astor.to_source(annotate_types(tracer.calls())['square_root']), '.py')


# #### End of Excursion

if __name__ == "__main__":
    print('\n#### End of Excursion')




# #### Excursion: A Type Annotator Class

if __name__ == "__main__":
    print('\n#### Excursion: A Type Annotator Class')




class TypeTracer(CallTracer):
    pass

class TypeAnnotator(TypeTracer):
    def typed_functions_ast(self, function_name=None):
        if function_name is None:
            return annotate_types(self.calls())

        return annotate_function_with_types(function_name, 
                                            self.calls(function_name))

    def typed_functions(self, function_name=None):
        if function_name is None:
            functions = ''
            for f_name in self.calls():
                try:
                    f_text = astor.to_source(
                        self.typed_functions_ast(f_name))
                except KeyError:
                    f_text = ''
                functions += f_text
            return functions

        return astor.to_source(self.typed_functions_ast(function_name))

# #### End of Excursion

if __name__ == "__main__":
    print('\n#### End of Excursion')




if __name__ == "__main__":
    with TypeAnnotator() as annotator:
        y = square_root(25.0)
        y = square_root(2.0)


if __name__ == "__main__":
    print_content(annotator.typed_functions(), '.py')


if __name__ == "__main__":
    with TypeAnnotator() as annotator:
        hello('type annotations')
        y = square_root(1.0)


if __name__ == "__main__":
    print_content(annotator.typed_functions(), '.py')


# #### Excursion: Handling Multiple Types

if __name__ == "__main__":
    print('\n#### Excursion: Handling Multiple Types')




if __name__ == "__main__":
    with CallTracer() as tracer:
        y = square_root(25.0)
        y = square_root(4)


if __name__ == "__main__":
    annotated_square_root_ast = annotate_types(tracer.calls())['square_root']
    print_content(astor.to_source(annotated_square_root_ast), '.py')


def sum3(a, b, c):
    return a + b + c

if __name__ == "__main__":
    with TypeAnnotator() as annotator:
        y = sum3(1.0, 2.0, 3.0)
    y


if __name__ == "__main__":
    print_content(annotator.typed_functions(), '.py')


if __name__ == "__main__":
    with TypeAnnotator() as annotator:
        y = sum3(1, 2, 3)
    y


if __name__ == "__main__":
    print_content(annotator.typed_functions(), '.py')


if __name__ == "__main__":
    with TypeAnnotator() as annotator:
        y = sum3("one", "two", "three")
    y


if __name__ == "__main__":
    print_content(annotator.typed_functions(), '.py')


if __name__ == "__main__":
    with TypeAnnotator() as annotator:
        y = sum3(1, 2, 3)
        y = sum3("one", "two", "three")


if __name__ == "__main__":
    typed_sum3_def = annotator.typed_functions('sum3')


if __name__ == "__main__":
    print_content(typed_sum3_def, '.py')


# #### End of Excursion

if __name__ == "__main__":
    print('\n#### End of Excursion')




# ## Mining Invariants

if __name__ == "__main__":
    print('\n## Mining Invariants')




# ### Annotating Functions with Pre- and Postconditions

if __name__ == "__main__":
    print('\n### Annotating Functions with Pre- and Postconditions')




def square_root_with_invariants(x):
    assert x >= 0  # Precondition

    ...

    assert result * result == x  # Postcondition
    return result

import functools

def condition(precondition=None, postcondition=None):
    def decorator(func):
        @functools.wraps(func)  # preserves name, docstring, etc
        def wrapper(*args, **kwargs):
            if precondition is not None:
                assert precondition(*args, **kwargs), \
                    "Precondition violated"

            # Call original function or method
            retval = func(*args, **kwargs)
            if postcondition is not None:
                assert postcondition(retval, *args, **kwargs), \
                    "Postcondition violated"

            return retval
        return wrapper
    return decorator

def precondition(check):
    return condition(precondition=check)

def postcondition(check):
    return condition(postcondition=check)

@precondition(lambda x: x > 0)
def square_root_with_precondition(x):
    return square_root(x)

if __name__ == "__main__":
    with ExpectError():
        square_root_with_precondition(-1.0)


import math

@postcondition(lambda ret, x: math.isclose(ret * ret, x))
def square_root_with_postcondition(x):
    return square_root(x)

if __name__ == "__main__":
    y = square_root_with_postcondition(2.0)
    y


@postcondition(lambda ret, x: math.isclose(ret * ret, x))
def buggy_square_root_with_postcondition(x):
    return square_root(x) + 0.1

if __name__ == "__main__":
    with ExpectError():
        y = buggy_square_root_with_postcondition(2.0)


# ### Mining Invariants

if __name__ == "__main__":
    print('\n### Mining Invariants')




# ### Defining Properties

if __name__ == "__main__":
    print('\n### Defining Properties')




INVARIANT_PROPERTIES = [
    "X < 0",
    "X <= 0",
    "X > 0",
    "X >= 0",
    # "X == 0",  # implied by "X", below
    # "X != 0",  # implied by "not X", below
]

INVARIANT_PROPERTIES += [
    "X == Y",
    "X > Y",
    "X < Y",
    "X >= Y",
    "X <= Y",
]

INVARIANT_PROPERTIES += [
    "isinstance(X, bool)",
    "isinstance(X, int)",
    "isinstance(X, float)",
    "isinstance(X, list)",
    "isinstance(X, dict)",
]

INVARIANT_PROPERTIES += [
    "X == Y + Z",
    "X == Y * Z",
    "X == Y - Z",
    "X == Y / Z",
]

INVARIANT_PROPERTIES += [
    "X < Y < Z",
    "X <= Y <= Z",
    "X > Y > Z",
    "X >= Y >= Z",
]

INVARIANT_PROPERTIES += [
    "X",
    "not X"
]

INVARIANT_PROPERTIES += [
    "X == len(Y)",
    "X == sum(Y)",
    "X in Y",
    "X.startswith(Y)",
    "X.endswith(Y)",
]

# ### Extracting Meta-Variables

if __name__ == "__main__":
    print('\n### Extracting Meta-Variables')




def metavars(prop):
    metavar_list = []

    class ArgVisitor(ast.NodeVisitor):
        def visit_Name(self, node):
            if node.id.isupper():
                metavar_list.append(node.id)

    ArgVisitor().visit(ast.parse(prop))
    return metavar_list

if __name__ == "__main__":
    assert metavars("X < 0") == ['X']


if __name__ == "__main__":
    assert metavars("X.startswith(Y)") == ['X', 'Y']


if __name__ == "__main__":
    assert metavars("isinstance(X, str)") == ['X']


# ### Instantiating Properties

if __name__ == "__main__":
    print('\n### Instantiating Properties')




def instantiate_prop_ast(prop, var_names):
    class NameTransformer(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id not in mapping:
                return node
            return ast.Name(id=mapping[node.id], ctx=ast.Load())

    meta_variables = metavars(prop)
    assert len(meta_variables) == len(var_names)

    mapping = {}
    for i in range(0, len(meta_variables)):
        mapping[meta_variables[i]] = var_names[i]

    prop_ast = ast.parse(prop, mode='eval')
    new_ast = NameTransformer().visit(prop_ast)

    return new_ast

def instantiate_prop(prop, var_names):
    prop_ast = instantiate_prop_ast(prop, var_names)
    prop_text = astor.to_source(prop_ast).strip()
    while prop_text.startswith('(') and prop_text.endswith(')'):
        prop_text = prop_text[1:-1]
    return prop_text

if __name__ == "__main__":
    assert instantiate_prop("X > Y", ['a', 'b']) == 'a > b'


if __name__ == "__main__":
    assert instantiate_prop("X.startswith(Y)", ['x', 'y']) == 'x.startswith(y)'


# ### Evaluating Properties

if __name__ == "__main__":
    print('\n### Evaluating Properties')




def prop_function_text(prop):
    return "lambda " + ", ".join(metavars(prop)) + ": " + prop

if __name__ == "__main__":
    prop_function_text("X > Y")


def prop_function(prop):
    return eval(prop_function_text(prop))

if __name__ == "__main__":
    p = prop_function("X > Y")


if __name__ == "__main__":
    quiz("What is p(100, 1)?", [
        "False",
        "True"
    ], p(100, 1) + 1)


if __name__ == "__main__":
    p(100, 1)


if __name__ == "__main__":
    p(1, 100)


# ### Checking Invariants

if __name__ == "__main__":
    print('\n### Checking Invariants')




import itertools

if __name__ == "__main__":
    for combination in itertools.permutations([1.0, 2.0, 3.0], 2):
        print(combination)


def true_property_instantiations(prop, vars_and_values, log=False):
    instantiations = set()
    p = prop_function(prop)

    len_metavars = len(metavars(prop))
    for combination in itertools.permutations(vars_and_values, len_metavars):
        args = [value for var_name, value in combination]
        var_names = [var_name for var_name, value in combination]

        try:
            result = p(*args)
        except:
            result = None

        if log:
            print(prop, combination, result)
        if result:
            instantiations.add((prop, tuple(var_names)))

    return instantiations

if __name__ == "__main__":
    invs = true_property_instantiations("X < Y", [('x', -1), ('y', 1)], log=True)
    invs


if __name__ == "__main__":
    for prop, var_names in invs:
        print(instantiate_prop(prop, var_names))


if __name__ == "__main__":
    invs = true_property_instantiations("X < 0", [('x', -1), ('y', 1)], log=True)


if __name__ == "__main__":
    for prop, var_names in invs:
        print(instantiate_prop(prop, var_names))


# ### Extracting Invariants

if __name__ == "__main__":
    print('\n### Extracting Invariants')




class InvariantTracer(CallTracer):
    def __init__(self, props=None, **kwargs):
        if props is None:
            props = INVARIANT_PROPERTIES

        self.props = props
        super().__init__(**kwargs)

RETURN_VALUE = 'return_value'

class InvariantTracer(InvariantTracer):
    def invariants(self, function_name=None):
        if function_name is None:
            return {function_name: self.invariants(function_name) 
                    for function_name in self.calls()}

        invariants = None
        for variables, return_value in self.calls(function_name):
            vars_and_values = variables + [(RETURN_VALUE, return_value)]

            s = set()
            for prop in self.props:
                s |= true_property_instantiations(prop, vars_and_values,
                                                  self.log)
            if invariants is None:
                invariants = s
            else:
                invariants &= s

        return invariants

if __name__ == "__main__":
    with InvariantTracer() as tracer:
        y = square_root(25.0)
        y = square_root(10.0)

    tracer.calls()


if __name__ == "__main__":
    invs = tracer.invariants('square_root')
    invs


def pretty_invariants(invariants):
    props = []
    for (prop, var_names) in invariants:
        props.append(instantiate_prop(prop, var_names))
    return sorted(props)

if __name__ == "__main__":
    pretty_invariants(invs)


if __name__ == "__main__":
    square_root(0.01)


if __name__ == "__main__":
    with InvariantTracer() as tracer:
        y = square_root(25.0)
        y = square_root(10.0)
        y = square_root(0.01)

    pretty_invariants(tracer.invariants('square_root'))


if __name__ == "__main__":
    with InvariantTracer() as tracer:
        y = sum3(1, 2, 3)
        y = sum3(-4, -5, -6)

    pretty_invariants(tracer.invariants('sum3'))


if __name__ == "__main__":
    with InvariantTracer() as tracer:
        y = sum3('a', 'b', 'c')
        y = sum3('f', 'e', 'd')

    pretty_invariants(tracer.invariants('sum3'))


if __name__ == "__main__":
    with InvariantTracer() as tracer:
        y = sum3('a', 'b', 'c')
        y = sum3('c', 'b', 'a')
        y = sum3(-4, -5, -6)
        y = sum3(0, 0, 0)

    pretty_invariants(tracer.invariants('sum3'))


# ### Converting Mined Invariants to Annotations

if __name__ == "__main__":
    print('\n### Converting Mined Invariants to Annotations')




class InvariantAnnotator(InvariantTracer):
    def params(self, function_name):
        arguments, return_value = self.calls(function_name)[0]
        return ", ".join(arg_name for (arg_name, arg_value) in arguments)

if __name__ == "__main__":
    with InvariantAnnotator() as annotator:
        y = square_root(25.0)
        y = sum3(1, 2, 3)


if __name__ == "__main__":
    annotator.params('square_root')


if __name__ == "__main__":
    annotator.params('sum3')


class InvariantAnnotator(InvariantAnnotator):
    def preconditions(self, function_name):
        conditions = []

        for inv in pretty_invariants(self.invariants(function_name)):
            if inv.find(RETURN_VALUE) >= 0:
                continue  # Postcondition

            cond = ("@precondition(lambda " + self.params(function_name) +
                    ": " + inv + ")")
            conditions.append(cond)

        return conditions

if __name__ == "__main__":
    with InvariantAnnotator() as annotator:
        y = square_root(25.0)
        y = square_root(0.01)
        y = sum3(1, 2, 3)


if __name__ == "__main__":
    annotator.preconditions('square_root')


class InvariantAnnotator(InvariantAnnotator):
    def postconditions(self, function_name):
        conditions = []

        for inv in pretty_invariants(self.invariants(function_name)):
            if inv.find(RETURN_VALUE) < 0:
                continue  # Precondition

            cond = (f"@postcondition(lambda {RETURN_VALUE},"
                    f" {self.params(function_name)}: {inv})")
            conditions.append(cond)

        return conditions

if __name__ == "__main__":
    with InvariantAnnotator() as annotator:
        y = square_root(25.0)
        y = square_root(0.01)
        y = sum3(1, 2, 3)


if __name__ == "__main__":
    annotator.postconditions('square_root')


class InvariantAnnotator(InvariantAnnotator):
    def functions_with_invariants(self):
        functions = ""
        for function_name in self.invariants():
            try:
                function = self.function_with_invariants(function_name)
            except KeyError:
                continue
            functions += function
        return functions

    def function_with_invariants(self, function_name):
        function = globals()[function_name]  # Can throw KeyError
        source = inspect.getsource(function)
        return '\n'.join(self.preconditions(function_name) +
                         self.postconditions(function_name)) + \
            '\n' + source

if __name__ == "__main__":
    with InvariantAnnotator() as annotator:
        y = square_root(25.0)
        y = square_root(0.01)
        y = sum3(1, 2, 3)


if __name__ == "__main__":
    print_content(annotator.function_with_invariants('square_root'), '.py')


# ## Avoiding Overspecialization

if __name__ == "__main__":
    print('\n## Avoiding Overspecialization')




def sum2(a, b):
    return a + b

if __name__ == "__main__":
    with InvariantAnnotator() as annotator:
        sum2(31, 45)
        sum2(0, 0)
        sum2(-1, -5)


if __name__ == "__main__":
    print_content(annotator.functions_with_invariants(), '.py')


if __name__ == "__main__":
    with InvariantAnnotator() as annotator:
        y = sum2(2, 2)
    print_content(annotator.functions_with_invariants(), '.py')


import random

if __name__ == "__main__":
    with InvariantAnnotator() as annotator:
        for i in range(100):
            a = random.randrange(-10, +10)
            b = random.randrange(-10, +10)
            length = sum2(a, b)


if __name__ == "__main__":
    print_content(annotator.function_with_invariants('sum2'), '.py')


# ## Partial Invariants

if __name__ == "__main__":
    print('\n## Partial Invariants')




if __package__ is None or __package__ == "":
    from StatisticalDebugger import middle  # minor dependency
else:
    from .StatisticalDebugger import middle  # minor dependency


if __name__ == "__main__":
    with InvariantAnnotator() as annotator:
        for i in range(100):
            x = random.randrange(-10, +10)
            y = random.randrange(-10, +10)
            z = random.randrange(-10, +10)
            mid = middle(x, y, z) 


if __name__ == "__main__":
    print_content(annotator.functions_with_invariants(), '.py')


if __package__ is None or __package__ == "":
    from Repairer import MIDDLE_FAILING_TESTCASES  # minor dependency
else:
    from .Repairer import MIDDLE_FAILING_TESTCASES  # minor dependency


if __name__ == "__main__":
    with InvariantAnnotator() as annotator:
        for x, y, z in MIDDLE_FAILING_TESTCASES:
            mid = middle(x, y, z) 


if __name__ == "__main__":
    print_content(annotator.functions_with_invariants(), '.py')


if __name__ == "__main__":
    quiz("Could `InvariantAnnotator` also determine a precondition "
         "that characterizes _passing_ runs?", ["Yes", "No"], int(math.exp(1)))


if __package__ is None or __package__ == "":
    from Repairer import MIDDLE_PASSING_TESTCASES  # minor dependency
else:
    from .Repairer import MIDDLE_PASSING_TESTCASES  # minor dependency


if __name__ == "__main__":
    with InvariantAnnotator() as annotator:
        for x, y, z in MIDDLE_PASSING_TESTCASES:
            mid = middle(x, y, z) 


if __name__ == "__main__":
    print_content(annotator.functions_with_invariants(), '.py')


# ## Some Examples

if __name__ == "__main__":
    print('\n## Some Examples')




# ### Removing HTML Markup

if __name__ == "__main__":
    print('\n### Removing HTML Markup')




if __package__ is None or __package__ == "":
    from Intro_Debugging import remove_html_markup
else:
    from .Intro_Debugging import remove_html_markup


if __name__ == "__main__":
    with InvariantAnnotator() as annotator:
        remove_html_markup("<foo>bar</foo>")
        remove_html_markup("bar")
        remove_html_markup('"bar"')


if __name__ == "__main__":
    print_content(annotator.functions_with_invariants(), '.py')


# ### A Recursive Function

if __name__ == "__main__":
    print('\n### A Recursive Function')




def list_length(elems):
    if elems == []:
        length = 0
    else:
        length = 1 + list_length(elems[1:])
    return length

if __name__ == "__main__":
    with InvariantAnnotator() as annotator:
        length = list_length([1, 2, 3])

    print_content(annotator.functions_with_invariants(), '.py')


# ### Sum of two Numbers

if __name__ == "__main__":
    print('\n### Sum of two Numbers')




def print_sum(a, b):
    print(a + b)

if __name__ == "__main__":
    with InvariantAnnotator() as annotator:
        print_sum(31, 45)
        print_sum(0, 0)
        print_sum(-1, -5)


if __name__ == "__main__":
    print_content(annotator.functions_with_invariants(), '.py')


# ## Checking Specifications

if __name__ == "__main__":
    print('\n## Checking Specifications')




if __name__ == "__main__":
    with InvariantAnnotator() as annotator:
        y = square_root(25.0)
        y = square_root(0.01)


if __name__ == "__main__":
    square_root_def = annotator.functions_with_invariants()
    square_root_def = square_root_def.replace('square_root',
                                              'square_root_annotated')


if __name__ == "__main__":
    print_content(square_root_def, '.py')


if __name__ == "__main__":
    exec(square_root_def)


if __name__ == "__main__":
    with ExpectError():
        square_root_annotated(-1.0)


if __name__ == "__main__":
    with ExpectTimeout(1):
        square_root(-1.0)


if __name__ == "__main__":
    square_root_def = square_root_def.replace('square_root_annotated',
                                              'square_root_negative')
    square_root_def = square_root_def.replace('return approx',
                                              'return -approx')


if __name__ == "__main__":
    print_content(square_root_def, '.py')


if __name__ == "__main__":
    exec(square_root_def)


if __name__ == "__main__":
    with ExpectError():
        square_root_negative(2.0)


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




def sum2(a, b):
    return a + b

if __name__ == "__main__":
    with TypeAnnotator() as type_annotator:
        sum2(1, 2)
        sum2(-4, -5)
        sum2(0, 0)


if __name__ == "__main__":
    print(type_annotator.typed_functions())


if __name__ == "__main__":
    with InvariantAnnotator() as inv_annotator:
        sum2(1, 2)
        sum2(-4, -5)
        sum2(0, 0)


if __name__ == "__main__":
    print(inv_annotator.functions_with_invariants())


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




# ### Exercise 1: Union Types

if __name__ == "__main__":
    print('\n### Exercise 1: Union Types')




def square_root_with_union_type(x: Union[int, float]) -> float:
    ...

# ### Exercise 2: Types for Local Variables

if __name__ == "__main__":
    print('\n### Exercise 2: Types for Local Variables')




def square_root_with_local_types(x: Union[int, float]) -> float:
    """Computes the square root of x, using the Newton-Raphson method"""
    approx: Optional[float] = None
    guess: float = x / 2
    while approx != guess:
        approx: float = guess
        guess: float = (approx + x / approx) / 2
    return approx

# ### Exercise 3: Verbose Invariant Checkers

if __name__ == "__main__":
    print('\n### Exercise 3: Verbose Invariant Checkers')




@precondition(lambda s: len(s) > 0)
def remove_first_char(s):
    return s[1:]

if __name__ == "__main__":
    with ExpectError():
        remove_first_char('')


def condition(precondition=None, postcondition=None, doc='Unknown'):
   def decorator(func):
       @functools.wraps(func) # preserves name, docstring, etc
       def wrapper(*args, **kwargs):
           if precondition is not None:
               assert precondition(*args, **kwargs), "Precondition violated: " + doc

           retval = func(*args, **kwargs) # call original function or method
           if postcondition is not None:
               assert postcondition(retval, *args, **kwargs), "Postcondition violated: " + doc

           return retval
       return wrapper
   return decorator

def precondition(check, **kwargs):
   return condition(precondition=check, doc=kwargs.get('doc', 'Unknown'))

def postcondition(check, **kwargs):
   return condition(postcondition=check, doc=kwargs.get('doc', 'Unknown'))

@precondition(lambda s: len(s) > 0, doc="len(s) > 0")
def remove_first_char(s):
    return s[1:]

remove_first_char('abc')

if __name__ == "__main__":
    with ExpectError():
        remove_first_char('')


class InvariantAnnotator(InvariantAnnotator):
   def preconditions(self, function_name):
       conditions = []

       for inv in pretty_invariants(self.invariants(function_name)):
           if inv.find(RETURN_VALUE) >= 0:
               continue  # Postcondition

           cond = "@precondition(lambda " + self.params(function_name) + ": " + inv + ', doc=' + repr(inv) + ")"
           conditions.append(cond)

       return conditions

class InvariantAnnotator(InvariantAnnotator):
   def postconditions(self, function_name):
       conditions = []

       for inv in pretty_invariants(self.invariants(function_name)):
           if inv.find(RETURN_VALUE) < 0:
               continue  # Precondition

           cond = ("@postcondition(lambda " + 
               RETURN_VALUE + ", " + self.params(function_name) + ": " + inv + ', doc=' + repr(inv) + ")")
           conditions.append(cond)

       return conditions

if __name__ == "__main__":
    with InvariantAnnotator() as annotator:
        y = sum2(2, 2)
    print_content(annotator.functions_with_invariants(), '.py')


# ### Exercise 4: Save Initial Values

if __name__ == "__main__":
    print('\n### Exercise 4: Save Initial Values')




# ### Exercise 5: Implications

if __name__ == "__main__":
    print('\n### Exercise 5: Implications')




# ### Exercise 6: Local Variables

if __name__ == "__main__":
    print('\n### Exercise 6: Local Variables')




# ### Exercise 7: Embedding Invariants as Assertions

if __name__ == "__main__":
    print('\n### Exercise 7: Embedding Invariants as Assertions')




# ### Exercise 8: Grammar-Generated Properties

if __name__ == "__main__":
    print('\n### Exercise 8: Grammar-Generated Properties')




class EmbeddedInvariantAnnotator(InvariantTracer):
    def functions_with_invariants_ast(self, function_name=None):
        if function_name is None:
            return annotate_functions_with_invariants(self.invariants())

        return annotate_function_with_invariants(function_name, self.invariants(function_name))

    def functions_with_invariants(self, function_name=None):
        if function_name is None:
            functions = ''
            for f_name in self.invariants():
                try:
                    f_text = astor.to_source(self.functions_with_invariants_ast(f_name))
                except KeyError:
                    f_text = ''
                functions += f_text
            return functions

        return astor.to_source(self.functions_with_invariants_ast(function_name))
    
    def function_with_invariants(self, function_name):
        return self.functions_with_invariants(function_name)
    def function_with_invariants_ast(self, function_name):
        return self.functions_with_invariants_ast(function_name)

def annotate_invariants(invariants):
    annotated_functions = {}
    
    for function_name in invariants:
        try:
            annotated_functions[function_name] = annotate_function_with_invariants(function_name, invariants[function_name])
        except KeyError:
            continue

    return annotated_functions

def annotate_function_with_invariants(function_name, function_invariants):
    function = globals()[function_name]
    function_code = inspect.getsource(function)
    function_ast = ast.parse(function_code)
    return annotate_function_ast_with_invariants(function_ast, function_invariants)

def annotate_function_ast_with_invariants(function_ast, function_invariants):
    annotated_function_ast = EmbeddedInvariantTransformer(function_invariants).visit(function_ast)
    return annotated_function_ast

class PreconditionTransformer(ast.NodeTransformer):
    def __init__(self, invariants):
        self.invariants = invariants
        super().__init__()

    def preconditions(self):
        preconditions = []
        for (prop, var_names) in self.invariants:
            assertion = "assert " + instantiate_prop(prop, var_names) + ', "violated precondition"'
            assertion_ast = ast.parse(assertion)

            if assertion.find(RETURN_VALUE) < 0:
                preconditions += assertion_ast.body

        return preconditions

    def insert_assertions(self, body):
        preconditions = self.preconditions()
        try:
            docstring = body[0].value.s
        except:
            docstring = None

        if docstring:
            return [body[0]] + preconditions + body[1:]
        else:
            return preconditions + body

    def visit_FunctionDef(self, node):
        """Add invariants to function"""
        # print(ast.dump(node))
        node.body = self.insert_assertions(node.body)
        return node    

class EmbeddedInvariantTransformer(PreconditionTransformer):
    pass

if __name__ == "__main__":
    with EmbeddedInvariantAnnotator() as annotator:
        square_root(5)


if __name__ == "__main__":
    print_content(annotator.functions_with_invariants(), '.py')


if __name__ == "__main__":
    with EmbeddedInvariantAnnotator() as annotator:
        y = sum3(3, 4, 5)
        y = sum3(-3, -4, -5)
        y = sum3(0, 0, 0)


if __name__ == "__main__":
    print_content(annotator.functions_with_invariants(), '.py')


class EmbeddedInvariantTransformer(PreconditionTransformer):
    def postconditions(self):
        postconditions = []

        for (prop, var_names) in self.invariants:
            assertion = ("assert " + instantiate_prop(prop, var_names) +
                         ', "violated postcondition"')
            assertion_ast = ast.parse(assertion)

            if assertion.find(RETURN_VALUE) >= 0:
                postconditions += assertion_ast.body

        return postconditions

    def insert_assertions(self, body):
        new_body = super().insert_assertions(body)
        postconditions = self.postconditions()

        body_ends_with_return = isinstance(new_body[-1], ast.Return)
        if body_ends_with_return:
            saver = RETURN_VALUE + " = " + astor.to_source(new_body[-1].value)
        else:
            saver = RETURN_VALUE + " = None"

        saver_ast = ast.parse(saver)
        postconditions = [saver_ast] + postconditions

        if body_ends_with_return:
            return new_body[:-1] + postconditions + [new_body[-1]]
        else:
            return new_body + postconditions

if __name__ == "__main__":
    with EmbeddedInvariantAnnotator() as annotator:
        square_root(5)


if __name__ == "__main__":
    square_root_def = annotator.functions_with_invariants()


if __name__ == "__main__":
    print_content(square_root_def, '.py')


if __name__ == "__main__":
    exec(square_root_def.replace('square_root', 'square_root_annotated'))


if __name__ == "__main__":
    with ExpectError():
        square_root_annotated(-1)


if __name__ == "__main__":
    with EmbeddedInvariantAnnotator() as annotator:
        y = sum3(3, 4, 5)
        y = sum3(-3, -4, -5)
        y = sum3(0, 0, 0)


if __name__ == "__main__":
    print_content(annotator.functions_with_invariants(), '.py')


if __name__ == "__main__":
    with EmbeddedInvariantAnnotator() as annotator:
        length = list_length([1, 2, 3])

    print_content(annotator.functions_with_invariants(), '.py')


if __name__ == "__main__":
    with EmbeddedInvariantAnnotator() as annotator:
        print_sum(31, 45)


if __name__ == "__main__":
    print_content(annotator.functions_with_invariants(), '.py')


# ### Exercise 9: Loop Invariants

if __name__ == "__main__":
    print('\n### Exercise 9: Loop Invariants')




# ### Exercise 10: Path Invariants

if __name__ == "__main__":
    print('\n### Exercise 10: Path Invariants')



