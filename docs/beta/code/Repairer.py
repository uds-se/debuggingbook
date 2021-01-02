#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Fuzzing Book".
# Web site: https://www.fuzzingbook.org/html/Repairer.html
# Last change: 2021-01-02 01:12:18+01:00
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


# # Repairing Code Automatically

if __name__ == "__main__":
    print('# Repairing Code Automatically')




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




# ## Automatic Code Repairs

if __name__ == "__main__":
    print('\n## Automatic Code Repairs')




if __package__ is None or __package__ == "":
    from StatisticalDebugger import middle
else:
    from .StatisticalDebugger import middle


if __name__ == "__main__":
    # ignore
    from bookutils import print_content


if __name__ == "__main__":
    # ignore
    import inspect


if __name__ == "__main__":
    # ignore
    _, first_lineno = inspect.getsourcelines(middle)
    middle_source = inspect.getsource(middle)
    print_content(middle_source, '.py', start_line_number=first_lineno)


if __name__ == "__main__":
    middle(4, 5, 6)


if __name__ == "__main__":
    middle(2, 1, 3)


def middle_sort_of_fixed(x, y, z):
    return x

if __name__ == "__main__":
    middle_sort_of_fixed(2, 1, 3)


# ## A Test Suite

if __name__ == "__main__":
    print('\n## A Test Suite')




import random

def middle_testcase():
    x = random.randrange(10)
    y = random.randrange(10)
    z = random.randrange(10)
    return x, y, z

if __name__ == "__main__":
    [middle_testcase() for i in range(5)]


def middle_test(x, y, z):
    m = middle(x, y, z)
    assert m == sorted([x, y, z])[1]

if __name__ == "__main__":
    middle_test(4, 5, 6)


if __package__ is None or __package__ == "":
    from ExpectError import ExpectError
else:
    from .ExpectError import ExpectError


if __name__ == "__main__":
    with ExpectError():
        middle_test(2, 1, 3)


def middle_passing_testcase():
    while True:
        try:
            x, y, z = middle_testcase()
            _ = middle_test(x, y, z)
            return x, y, z
        except AssertionError:
            pass

if __name__ == "__main__":
    (x, y, z) = middle_passing_testcase()
    m = middle(x, y, z)
    print(f"middle({x}, {y}, {z}) = {m}")


def middle_failing_testcase():
    while True:
        try:
            x, y, z = middle_testcase()
            _ = middle_test(x, y, z)
        except AssertionError:
            return x, y, z

if __name__ == "__main__":
    (x, y, z) = middle_failing_testcase()
    m = middle(x, y, z)
    print(f"middle({x}, {y}, {z}) = {m}")


MIDDLE_TESTS = 100

MIDDLE_PASSING_TESTCASES = [middle_passing_testcase()
                            for i in range(MIDDLE_TESTS)]

MIDDLE_FAILING_TESTCASES = [middle_failing_testcase()
                            for i in range(MIDDLE_TESTS)]

# ## Locating the Defect

if __name__ == "__main__":
    print('\n## Locating the Defect')




if __package__ is None or __package__ == "":
    from StatisticalDebugger import OchiaiDebugger, CoverageCollector
else:
    from .StatisticalDebugger import OchiaiDebugger, CoverageCollector


if __name__ == "__main__":
    debugger = OchiaiDebugger(CoverageCollector)

    for x, y, z in MIDDLE_PASSING_TESTCASES:
        with debugger.collect_pass():
            m = middle(x, y, z)

    for x, y, z in MIDDLE_FAILING_TESTCASES:
        with debugger.collect_fail():
            m = middle(x, y, z)


if __name__ == "__main__":
    debugger


if __name__ == "__main__":
    # ignore
    location = debugger.rank()[0]
    (func_name, lineno) = location
    lines, first_lineno = inspect.getsourcelines(middle)
    print(lineno, end="")
    print_content(lines[lineno - first_lineno], '.py')


if __name__ == "__main__":
    # ignore
    debugger.suspiciousness(location)


# ## Random Code Mutations

if __name__ == "__main__":
    print('\n## Random Code Mutations')




import string

if __name__ == "__main__":
    string.ascii_letters


if __name__ == "__main__":
    len(string.ascii_letters + '_') * \
      len(string.ascii_letters + '_' + string.digits) * \
      len(string.ascii_letters + '_' + string.digits)


import ast
import astor
import inspect

if __package__ is None or __package__ == "":
    from bookutils import print_content, show_ast
else:
    from .bookutils import print_content, show_ast


def middle_tree():
    return ast.parse(inspect.getsource(middle))

if __name__ == "__main__":
    show_ast(middle_tree())


if __name__ == "__main__":
    print(ast.dump(middle_tree()))


if __name__ == "__main__":
    ast.dump(middle_tree().body[0].body[0].body[0].body[0])


# ### Picking Statements

if __name__ == "__main__":
    print('\n### Picking Statements')




from ast import NodeVisitor

class StatementVisitor(NodeVisitor):
    """Visit all statements within function defs in an AST"""
    def __init__(self):
        self.statements = []
        self.func_name = None
        self.statements_seen = set()
        super().__init__()

    def add_statements(self, node, attr):
        elems = getattr(node, attr, [])
        if not isinstance(elems, list):
            elems = [elems]
            
        for elem in elems:
            stmt = (elem, self.func_name)
            if stmt in self.statements_seen:
                continue
                
            self.statements.append(stmt)
            self.statements_seen.add(stmt)

    def visit_node(self, node):
        # Any node other than the ones listed below
        self.add_statements(node, 'body')
        self.add_statements(node, 'orelse')

    def visit_Module(self, node):
        # Module children are defs, classes and globals - don't add
        super().generic_visit(node)

    def visit_ClassDef(self, node):
        # Class children are defs and globals - don't add
        super().generic_visit(node)

    def generic_visit(self, node):
        self.visit_node(node)
        super().generic_visit(node)
        
    def visit_FunctionDef(self, node):
        if self.func_name is None:
            self.func_name = node.name
        
        self.visit_node(node)
        super().generic_visit(node)
        self.func_name = None

    def visit_AsyncFunctionDef(self, node):
        return visit_FunctionDef(self, node)

def all_statements_and_functions(tree, tp=None):
    visitor = StatementVisitor()
    visitor.visit(tree)
    statements = visitor.statements
    if tp is not None:
        statements = [s for s in statements if isinstance(s[0], tp)]

    return statements

def all_statements(tree, tp=None):
    return [stmt for stmt, func_name in all_statements_and_functions(tree, tp)]

if __name__ == "__main__":
    all_statements(middle_tree(), ast.Return)


if __name__ == "__main__":
    all_statements_and_functions(middle_tree(), ast.If)


import random

if __name__ == "__main__":
    random_node = random.choice(all_statements(middle_tree()))
    astor.to_source(random_node)


# ### Mutating Statements

if __name__ == "__main__":
    print('\n### Mutating Statements')




from ast import NodeTransformer

import copy

class StatementMutator(NodeTransformer):
    """Mutate statements in an AST for automated repair."""
    def __init__(self, suspiciousness_func=None, statements=None, log=False):
        """Constructor.
        `suspiciousness_func` is a function that takes a location 
        (function, line_number) and returns a suspiciousness value
        between 0 and 1.0. If not given, all locations get the 
        same suspiciousness.
        `statements` is a list of statements to choose from.
        """
        super().__init__()

        if suspiciousness_func is None:
            suspiciousness_func = lambda location: 1.0
        self.suspiciousness_func = suspiciousness_func

        if statements is None:
            statements = []
        self.statements = statements

        self.log = log

        self.mutations = 0

# #### Choosing Nodes to Mutate

if __name__ == "__main__":
    print('\n#### Choosing Nodes to Mutate')




import warnings

class StatementMutator(StatementMutator):
    def format_node(self, node):
        if node is None:
            return None
        if isinstance(node, list):
            return "; ".join(self.format_node(elem) for elem in node)

        s = RE_SPACE.sub(' ', astor.to_source(node)).strip()
        if len(s) > 20:
            s = s[:20] + "..."
        return repr(s)

    def node_suspiciousness(self, stmt, func_name):
        if not hasattr(stmt, 'lineno'):
            warnings.warn(f"{self.format_node(stmt)}: Expected line number")
            return 0.0

        suspiciousness = self.suspiciousness_func((func_name, stmt.lineno))
        if suspiciousness is None:  # not executed
            return 0.0

        return suspiciousness

class StatementMutator(StatementMutator):
    def node_to_be_mutated(self, tree):
        statements = all_statements_and_functions(tree)
        assert len(statements) > 0, "No statements"

        weights = [self.node_suspiciousness(stmt, func_name) 
                   for stmt, func_name in statements]
        assert sum(weights) > 0, "No suspicious location"

        if self.log > 1:
            print("Weights:")
            for i, stmt in enumerate(statements):
                node, func_name = stmt
                print(f"{weights[i]:.2} {self.format_node(node)}")

        stmts = [stmt for stmt, func_name in statements]

        return random.choices(stmts, weights=weights)[0]

# #### Choosing an Operator

if __name__ == "__main__":
    print('\n#### Choosing an Operator')




import re

RE_SPACE = re.compile(r'[ \t\n]+')

class StatementMutator(StatementMutator):
    def choose_op(self):
        return random.choice([self.insert, self.swap, self.delete])

    def visit(self, node):
        super().visit(node)  # Visits (and transforms?) children

        if not node.mutate_me:
            return node
        
        op = self.choose_op()
        new_node = op(node)
        self.mutations += 1

        if self.log:
            print(f"{node.lineno:4}:{op.__name__:6}: "
                  f"{self.format_node(node)} "
                  f"becomes {self.format_node(new_node)}")

        return new_node

# #### Swapping Statements

if __name__ == "__main__":
    print('\n#### Swapping Statements')




class StatementMutator(StatementMutator):
    def choose_statement(self):
        return copy.deepcopy(random.choice(self.statements))

class StatementMutator(StatementMutator):
    def swap(self, node):
        # Replace with a random node from statements
        new_node = self.choose_statement()

        if isinstance(new_node, ast.stmt):
            # The source `if P: X` is added as `if P: pass`
            if hasattr(new_node, 'body'):
                new_node.body = [ast.Pass()]
            if hasattr(new_node, 'orelse'):
                new_node.orelse = []
            if hasattr(new_node, 'finalbody'):
                new_node.finalbody = []

        # ast.copy_location(new_node, node)
        return new_node

# #### Inserting Statements

if __name__ == "__main__":
    print('\n#### Inserting Statements')




class StatementMutator(StatementMutator):
    def insert(self, node):
        # Insert a random node from statements
        new_node = self.choose_statement()

        if isinstance(new_node, ast.stmt) and hasattr(new_node, 'body'):
            # Inserting `if P: X` as `if P:`
            new_node.body = [node]
            if hasattr(new_node, 'orelse'):
                new_node.orelse = []
            if hasattr(new_node, 'finalbody'):
                new_node.finalbody = []
            # ast.copy_location(new_node, node)
            return new_node

        # Only insert before `return`, not after it
        if isinstance(node, ast.Return):
            if isinstance(new_node, ast.Return):
                return new_node
            else:
                return [new_node, node]

        return [node, new_node]

# #### Deleting Statements

if __name__ == "__main__":
    print('\n#### Deleting Statements')




class StatementMutator(StatementMutator):
    def delete(self, node):
        # Delete this node

        branches = [attr for attr in ['body', 'orelse', 'finalbody']
                    if hasattr(node, attr) and getattr(node, attr)]
        if branches:
            # Replace `if P: S` by `S`
            branch = random.choice(branches)
            new_node = getattr(node, branch)
            return new_node

        if isinstance(node, ast.stmt):
            # Avoid empty bodies; make this a `pass` statement
            new_node = ast.Pass()
            ast.copy_location(new_node, node)
            return new_node

        return None  # Just delete

# #### All Together

if __name__ == "__main__":
    print('\n#### All Together')




class StatementMutator(StatementMutator):
    def visit(self, node):
        if not self.statements:
            self.statements = all_statements(node)

        return super().visit(node)

class StatementMutator(StatementMutator):
    def mutate(self, tree):
        tree = copy.deepcopy(tree)
        for node in ast.walk(tree):
            node.mutate_me = False

        node = self.node_to_be_mutated(tree)
        node.mutate_me = True

        self.mutations = 0
        tree = self.visit(tree)

        if self.mutations == 0:
            warnings.warn("No mutations found")

        ast.fix_missing_locations(tree)
        return tree

if __name__ == "__main__":
    mutator = StatementMutator(log=True)
    for i in range(10):
        new_tree = mutator.mutate(middle_tree())


if __name__ == "__main__":
    print_content(astor.to_source(new_tree), '.py')


# ## Fitness

if __name__ == "__main__":
    print('\n## Fitness')




WEIGHT_PASSING = 0.99
WEIGHT_FAILING = 0.01

def middle_fitness(tree):
    original_middle = middle

    try:
        code = compile(tree, '<fitness>', 'exec')
    except ValueError:
        return 0  # Compilation error

    exec(code, globals())

    passing_passed = 0
    failing_passed = 0

    # Test how many of the passing runs pass
    for x, y, z in MIDDLE_PASSING_TESTCASES:
        try:
            middle_test(x, y, z)
            passing_passed += 1
        except AssertionError:
            pass

    passing_ratio = passing_passed / len(MIDDLE_PASSING_TESTCASES)

    # Test how many of the failing runs pass
    for x, y, z in MIDDLE_FAILING_TESTCASES:
        try:
            middle_test(x, y, z)
            failing_passed += 1
        except AssertionError:
            pass

    failing_ratio = failing_passed / len(MIDDLE_FAILING_TESTCASES)

    fitness = (WEIGHT_PASSING * passing_ratio +
               WEIGHT_FAILING * failing_ratio)

    globals()['middle'] = original_middle
    return fitness

if __name__ == "__main__":
    middle_fitness(middle_tree())


if __name__ == "__main__":
    middle_fitness(ast.parse("def middle(x, y, z): return z"))


if __package__ is None or __package__ == "":
    from StatisticalDebugger import middle_fixed
else:
    from .StatisticalDebugger import middle_fixed


if __name__ == "__main__":
    middle_fixed_source = \
        inspect.getsource(middle_fixed).replace('middle_fixed', 'middle').strip()


if __name__ == "__main__":
    middle_fitness(ast.parse(middle_fixed_source))


# ## Population

if __name__ == "__main__":
    print('\n## Population')




POPULATION_SIZE = 50
mutator = StatementMutator()

MIDDLE_POPULATION = [middle_tree()] + \
    [mutator.mutate(middle_tree()) for i in range(POPULATION_SIZE - 1)]

if __name__ == "__main__":
    MIDDLE_POPULATION.sort(key=middle_fitness, reverse=True)


if __name__ == "__main__":
    print(astor.to_source(MIDDLE_POPULATION[0]), 
          middle_fitness(MIDDLE_POPULATION[0]))


if __name__ == "__main__":
    print(astor.to_source(MIDDLE_POPULATION[-1]), 
          middle_fitness(MIDDLE_POPULATION[-1]))


# ## Evolution

if __name__ == "__main__":
    print('\n## Evolution')




def evolve_middle():
    global MIDDLE_POPULATION

    statements = all_statements(middle_tree())
    mutator = StatementMutator(statements=statements)

    n = len(MIDDLE_POPULATION)

    MIDDLE_POPULATION.sort(key=middle_fitness, reverse=True)
    MIDDLE_POPULATION = MIDDLE_POPULATION[:n // 2]

    offspring = [mutator.mutate(tree) for tree in MIDDLE_POPULATION]

    MIDDLE_POPULATION += offspring

if __name__ == "__main__":
    evolve_middle()


if __name__ == "__main__":
    tree = MIDDLE_POPULATION[0]
    print(astor.to_source(tree), middle_fitness(tree))


if __name__ == "__main__":
    for i in range(50):
        evolve_middle()
        best_tree = MIDDLE_POPULATION[0]
        fitness = middle_fitness(best_tree)
        print(f"\rIteration {i:2}: fitness = {fitness}  ", end="")
        if fitness >= 1.0:
            break


if __name__ == "__main__":
    print_content(astor.to_source(best_tree), '.py')


# ## Simplifying

if __name__ == "__main__":
    print('\n## Simplifying')




# ### Textual Reduction

if __name__ == "__main__":
    print('\n### Textual Reduction')




if __package__ is None or __package__ == "":
    from DeltaDebugger import DeltaDebugger
else:
    from .DeltaDebugger import DeltaDebugger


if __name__ == "__main__":
    middle_lines = astor.to_source(best_tree).split('\n')


def test_middle_lines(lines):
    source = "\n".join(lines)
    tree = ast.parse(source)
    assert middle_fitness(tree) < 1.0  # "Fail" only while fitness is 1.0

if __name__ == "__main__":
    with DeltaDebugger() as dd:
        test_middle_lines(middle_lines)


if __name__ == "__main__":
    reduced_source = "\n".join(dd.min_args()['lines'])
    repaired_source = astor.to_source(ast.parse(reduced_source))  # normalize
    print_content(repaired_source, '.py')


if __name__ == "__main__":
    original_source = astor.to_source(middle_tree())


if __package__ is None or __package__ == "":
    from ChangeDebugger import diff
else:
    from .ChangeDebugger import diff


import urllib

def print_patch(p):
    print_content(urllib.parse.unquote(str(p)), '.py')

if __name__ == "__main__":
    for patch in diff(original_source, repaired_source):
        print_patch(patch)


# ## Crossover

if __name__ == "__main__":
    print('\n## Crossover')




def p1():
    def inner():
        print(a)
        print(b)
        print(c)

    a = 1
    b = 2
    c = 3

def p2():
    def inner():
        print(x)
        print(y)
        print(z)

    x = 1
    y = 2
    z = 3

class BodyCrossover:
    def __init__(self, log=False):
        self.log = log

    def cross_bodies(self, body_1, body_2):
        """Crossover the statement lists `body_1` x `body_2`.
        Return new lists.
        """

        assert isinstance(body_1, list)
        assert isinstance(body_2, list)

        split_1 = len(body_1) // 2
        split_2 = len(body_2) // 2
        return body_1[:split_1] + body_2[split_2:], body_2[:split_2] + body_1[split_1:]

if __name__ == "__main__":
    tree_p1 = ast.parse(inspect.getsource(p1))
    tree_p2 = ast.parse(inspect.getsource(p2))


if __name__ == "__main__":
    body_p1 = tree_p1.body[0].body
    body_p2 = tree_p2.body[0].body
    body_p1


if __name__ == "__main__":
    crosser = BodyCrossover()
    tree_p1.body[0].body, tree_p2.body[0].body = crosser.cross_bodies(body_p1, body_p2)


if __name__ == "__main__":
    print_content(astor.to_source(tree_p1), '.py')


if __name__ == "__main__":
    print_content(astor.to_source(tree_p2), '.py')


class BodyCrossover(BodyCrossover):
    # In modules and class defs, the ordering of elements does not matter (much)
    SKIP_LIST = {ast.Module, ast.ClassDef}

    def can_cross(self, tree, body_attr='body'):
        if any(isinstance(tree, cls) for cls in self.SKIP_LIST):
            return False

        body = getattr(tree, body_attr, [])
        return body and len(body) >= 2

class BodyCrossover(BodyCrossover):
    def crossover_attr(self, t1, t2, body_attr):
        """Crossover the bodies `body_attr` of two trees `t1` and `t2`.
        Return True if successful."""
        assert isinstance(t1, ast.AST)
        assert isinstance(t2, ast.AST)
        assert isinstance(body_attr, str)

        if not getattr(t1, body_attr, None) or not getattr(t2, body_attr, None):
            return False

        body_1 = getattr(t1, body_attr)
        body_2 = getattr(t2, body_attr)

        # print(f"t1.{body_attr} = {body_1}")
        # print(f"t2.{body_attr} = {body_2}")

        # If both trees have the attribute, we can cross their bodies
        if self.can_cross(t1, body_attr) and self.can_cross(t2, body_attr):
            if self.log:
                print(f"Crossing {t1}.{body_attr} and {t2}.{body_attr}")

            new_body_1, new_body_2 = self.cross_bodies(body_1, body_2)
            setattr(t1, body_attr, new_body_1)
            setattr(t2, body_attr, new_body_2)
            return True

        # Strategy 1: Find matches in class/function of same name
        for child_1 in body_1:
            if hasattr(child_1, 'name'):
                for child_2 in body_2:
                    if (hasattr(child_2, 'name') and
                           child_1.name == child_2.name):
                        if self.crossover(child_1, child_2):
                            return True

        # Strategy 2: Find matches anywhere
        for child_1 in random.sample(body_1, len(body_1)):
            for child_2 in random.sample(body_2, len(body_2)):
                if self.crossover(child_1, child_2):
                    return True

        return False

class BodyCrossover(BodyCrossover):
    def crossover_branches(self, t1, t2):
        """Special case:
        `t1` = `if P: S1 else: S2` x `t2` = `if P': S1' else: S2'`
        becomes
        `t1` = `if P: S2' else: S1'` and `t2` = `if P': S2 else: S1`
        Returns True if successful.
        """
        assert isinstance(t1, ast.AST)
        assert isinstance(t2, ast.AST)
        
        if (hasattr(t1, 'body') and hasattr(t1, 'orelse') and
            hasattr(t2, 'body') and hasattr(t2, 'orelse')):
            t1.body, t1.orelse, t2.body, t2.orelse = \
                t2.orelse, t2.body, t2.orelse, t1.body
            return True

        return False

class CrossoverError(ValueError):
    pass

class BodyCrossover(BodyCrossover):
    def crossover(self, t1, t2):
        """Do a crossover of ASTs `t1` and `t2`.
        Raises `CrossoverError` if no crossover is found."""
        assert isinstance(t1, ast.AST)
        assert isinstance(t2, ast.AST)

        if self.crossover_branches(t1, t2):
            return t1, t2

        for body_attr in ['body', 'orelse', 'finalbody']:
            if self.crossover_attr(t1, t2, body_attr):
                return t1, t2

        raise CrossoverError("No crossover found")

def p1():
    if True:
        print(1)
        print(2)
        print(3)

def p2():
    if True:
        print(a)
        print(b)
    if False:
        print(c)
        print(d)

if __name__ == "__main__":
    crossover = BodyCrossover(log=True)
    tree_p1 = ast.parse(inspect.getsource(p1))
    tree_p2 = ast.parse(inspect.getsource(p2))
    crossover.crossover(tree_p1, tree_p2)


if __name__ == "__main__":
    print_content(astor.to_source(tree_p1), '.py')


if __name__ == "__main__":
    print_content(astor.to_source(tree_p2), '.py')


# ## A Repairer Class

if __name__ == "__main__":
    print('\n## A Repairer Class')




class RobustCoverageCollector(CoverageCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exception_type = None

    def __exit__(self, tp, value, traceback):
        """Called at end of `with` block. Turn tracing off."""
        super().__exit__(tp, value, traceback)
        self.exception_type = tp
        return True  # ignore exception

class Repairer(OchiaiDebugger):
    def __init__(self, *args, **kwargs):
        self.fitness_cache = {}
        return super().__init__(RobustCoverageCollector, *args, **kwargs)

# ### Running Tests

if __name__ == "__main__":
    print('\n### Running Tests')




class FailureNotReproducedError(ValueError):
    pass

class Repairer(Repairer):
    def run_test_set(self, test_set, validate=False):
        passed = 0
        collectors = self.collectors[test_set]
        function = globals()[self.function().__name__]  # may be redefined

        for c in collectors:
            if self.log >= 4:
                print(f"Testing {c.id()}...", end="")

            try:
                function(**c.args())
            except Exception as err:
                if self.log >= 4:
                    print(f"failed ({err.__class__.__name__})")

                if validate and test_set == self.PASS:
                    raise err(
                        f"{c.id()} should have passed, but failed")
                continue

            passed += 1
            if self.log >= 4:
                print("passed")

            if validate and test_set == self.FAIL:
                raise FailureNotReproducedError(
                    f"{c.id()} should have failed, but passed")

        return passed

class Repairer(Repairer):
    def weight(self, test_set):
        return {
            self.PASS: WEIGHT_PASSING,
            self.FAIL: WEIGHT_FAILING
        }[test_set]

    def run_tests(self, validate=False):
        # Run tests
        fitness = 0

        for test_set in [self.PASS, self.FAIL]:
            passed = self.run_test_set(test_set, validate=validate)
            ratio = passed / len(self.collectors[test_set])
            fitness += self.weight(test_set) * ratio

        return fitness

class DefinitionVisitor(NodeVisitor):
    def __init__(self):
        self.definitions = []

    def add_definition(self, node):
        self.definitions.append(node.name)

    def visit_FunctionDef(self, node):
        self.add_definition(node)

    def visit_AsyncFunctionDef(self, node):
        self.add_definition(node)

    def visit_Class(self, node):
        self.add_definition(node)

class Repairer(Repairer):
    def toplevel_defs(self, tree):
        visitor = DefinitionVisitor()
        visitor.visit(tree)
        return visitor.definitions

# ### (Re)defining Functions

if __name__ == "__main__":
    print('\n### (Re)defining Functions')




class Repairer(Repairer):
    def fitness(self, tree):
        key = ast.dump(tree)
        if key in self.fitness_cache:
            return self.fitness_cache[key]

        # Save defs
        original_defs = {}
        for name in self.toplevel_defs(tree):
            original_defs[name] = globals()[name]

        if self.log >= 3:
            print("Repair candidate:")
            print_content(astor.to_source(tree), '.py')
            print()

        # Create new definition
        try:
            code = compile(tree, '<Repairer>', 'exec')
        except ValueError:  # Compilation error
            if self.log >= 3:
                print(f"Fitness = 0 (compilation error)")

            fitness = 0
            return fitness

        exec(code, globals())

        fitness = self.run_tests(validate=False)

        for name in original_defs:
            globals()[name] = original_defs[name]

        if self.log >= 3:
            print(f"Fitness = {fitness}")

        self.fitness_cache[key] = fitness
        return fitness

class Repairer(Repairer):
    def validate(self):
        fitness = self.run_tests(validate=True)
        assert fitness == self.weight(self.PASS)

# ### Repairing

if __name__ == "__main__":
    print('\n### Repairing')




class Repairer(Repairer):
    def getsource(self, item):
        if isinstance(item, str):
            item = globals()[item]
        return inspect.getsource(item)

class Repairer(Repairer):
    def repair(self, targets=None, sources=None, 
               population_size=POPULATION_SIZE,
               iterations=100, 
               mutator_cls=StatementMutator, mutator=None, 
               crossover_cls=BodyCrossover, crossover=None):
        """Repair the function test runs collected from.
        `targets`: a list of functions/modules to be repaired.
        (default: covered functions)
        `sources`: a list of functions/modules to take repairs from.
        (default: `targets`)
        """
        if targets is None:
            targets = self.default_functions()
        if not targets:
            raise ValueError("No targets to repair")

        if sources is None:
            sources = self.default_functions()
        if not sources:
            raise ValueError("No sources to take repairs from")

        if self.function() is None:
            raise ValueError("Multiple entry points observed")
            
        target_tree = self.parse(targets)
        source_tree = self.parse(sources)

        self.log_tree("Target code to be repaired:", target_tree)
        if ast.dump(target_tree) != ast.dump(source_tree):
            self.log_tree("Source code to take repairs from:", source_tree)

        self.validate()

        if mutator is None:
            statements = all_statements(source_tree)
            mutator = mutator_cls(statements=statements, 
                                  suspiciousness_func=self.suspiciousness,
                                  log=(self.log >= 3))
        if crossover is None:
            crossover = crossover_cls()

        population = [target_tree] + \
            [mutator.mutate(copy.deepcopy(target_tree))
                for i in range(population_size)]

        last_key = ast.dump(target_tree)

        for iteration in range(iterations):
            population = self.evolve(population, mutator, crossover)

            best_tree = population[0]
            fitness = self.fitness(best_tree)

            if self.log:
                print(f"Evolving population: "
                      f"iteration{iteration:4}/{iterations} "
                      f"fitness = {fitness:.5}   \r", end="")

            if self.log >= 2:
                best_key = ast.dump(best_tree)
                if best_key != last_key:
                    print()
                    print()
                    self.log_tree(f"New best code (fitness = {fitness}):",
                                  best_tree)
                    last_key = best_key

            if fitness >= 1.0:
                break

        if self.log:
            print()

        if self.log and self.log < 2:
            self.log_tree(f"Best code (fitness = {fitness}):", best_tree)

        best_tree = self.reduce(best_tree)
        fitness = self.fitness(best_tree)

        self.log_tree(f"Reduced code (fitness = {fitness}):", best_tree)

        return best_tree, fitness

# ### Evolving

if __name__ == "__main__":
    print('\n### Evolving')




class Repairer(Repairer):
    def evolve(self, population, mutator, crossover):
        n = len(population)

        population.sort(key=self.fitness, reverse=True)
        population = population[:n // 2]

        offspring = [mutator.mutate(tree) for tree in population]
        population += offspring

        return population

class Repairer(Repairer):
    def fitness_key(self, tree):
        tree_size = len([node for node in ast.walk(tree)])
        return (self.fitness(tree), -tree_size)

class Repairer(Repairer):
    def evolve(self, population, mutator, crossover):
        n = len(population)

        # Keep the fitter half of the population
        population.sort(key=self.fitness_key, reverse=True)
        population = population[:n // 2]

        # Create offspring as crossover of parents
        offspring = []
        while len(offspring) < n // 2:
            parent_1 = copy.deepcopy(random.choice(population))
            parent_2 = copy.deepcopy(random.choice(population))
            try:
                crossover.crossover(parent_1, parent_2)
                offspring += [parent_1, parent_2]
            except CrossoverError:
                # Try different parents
                pass

        # Mutate offspring
        offspring = [mutator.mutate(tree) for tree in offspring]
        population += offspring

        return population

# ### Reducing

if __name__ == "__main__":
    print('\n### Reducing')




class Repairer(Repairer):
    def test_reduce(self, lines, original_fitness):
        source = "\n".join(lines)
        tree = ast.parse(source)
        assert self.fitness(tree) < original_fitness

    def reduce(self, tree):
        original_fitness = self.fitness(tree)
        source_lines = astor.to_source(tree).split('\n')

        with DeltaDebugger() as dd:
            self.test_reduce(source_lines, original_fitness)

        reduced_source = "\n".join(dd.min_args()['lines'])
        return ast.parse(reduced_source)

# ### Helper Functions

if __name__ == "__main__":
    print('\n### Helper Functions')




class Repairer(Repairer):
    def default_functions(self):
        def is_test(name):
            return name.startswith('test') or name.endswith('test')

        return [name for name in self.covered_functions()
                if not is_test(name)]

    def log_tree(self, description, tree):
        if self.log:
            print(description)
            print_content(astor.to_source(tree), '.py')
            print()
            print()
            
    def parse(self, names):
        tree = ast.parse("")
        for name in names:
            item = globals()[name]
            item_lines, item_first_lineno = inspect.getsourcelines(item)
            item_tree = ast.parse("".join(item_lines))
            ast.increment_lineno(item_tree, item_first_lineno - 1)
            tree.body += item_tree.body

        return tree

# ### Repairer in Action

if __name__ == "__main__":
    print('\n### Repairer in Action')




if __name__ == "__main__":
    repairer = Repairer(log=2)


if __name__ == "__main__":
    for x, y, z in MIDDLE_PASSING_TESTCASES:
        with repairer.collect_pass():
            m = middle_test(x, y, z)


if __name__ == "__main__":
    for x, y, z in MIDDLE_FAILING_TESTCASES:
        with repairer.collect_fail():
            m = middle_test(x, y, z)


if __name__ == "__main__":
    print(repairer.code(function=middle, suspiciousness=True))


# repairer  # FIXME

if __name__ == "__main__":
    repairer.suspiciousness(('middle', 508))


if __name__ == "__main__":
    best_tree, fitness = repairer.repair()


if __name__ == "__main__":
    print_content(astor.to_source(best_tree), '.py')


if __name__ == "__main__":
    fitness


# ## More Examples

if __name__ == "__main__":
    print('\n## More Examples')




# ### Removing HTML Markup

if __name__ == "__main__":
    print('\n### Removing HTML Markup')




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

def remove_html_markup_tree():
    return ast.parse(inspect.getsource(remove_html_markup))

def remove_html_markup_test(html, plain):
    outcome = remove_html_markup(html)
    assert outcome == plain, \
        f"Got {repr(outcome)}, expected {repr(plain)}"

if __name__ == "__main__":
    globals()['remove_html_markup_test']


if __name__ == "__main__":
    remove_html_markup_test('<foo>bar</foo>', 'bar')


if __package__ is None or __package__ == "":
    from ExpectError import ExpectError
else:
    from .ExpectError import ExpectError


if __name__ == "__main__":
    with ExpectError():
        remove_html_markup_test('<foo>"bar"</foo>', '"bar"')


def random_string(length=5, start=ord(' '), end=ord('~')):
    return "".join(chr(random.randrange(start, end + 1)) for i in range(length))

if __name__ == "__main__":
    random_string()


def random_id(length=2):
    return random_string(start=ord('a'), end=ord('z'))

if __name__ == "__main__":
    random_id()


def random_plain():
    return random_string().replace('<', '').replace('>', '')

def random_string_noquotes():
    return random_string().replace('"', '').replace("'", '')

def random_html(depth=0):
    prefix = random_plain()
    tag = random_id()

    if depth > 0:
        html, plain = random_html(depth - 1)
    else:
        html = plain = random_plain()

    attr = random_id()
    value = '"' + random_string_noquotes() + '"'
    postfix = random_plain()

    return f'{prefix}<{tag} {attr}={value}>{html}</{tag}>{postfix}', \
        prefix + plain + postfix

if __name__ == "__main__":
    random_html()


def remove_html_testcase(expected=True):
    while True:
        html, plain = random_html()
        outcome = (remove_html_markup(html) == plain)
        if outcome == expected:
            return html, plain

REMOVE_HTML_PASSING_TESTCASES = \
    [remove_html_testcase(True) for i in range(100)]
REMOVE_HTML_FAILING_TESTCASES = \
    [remove_html_testcase(False) for i in range(100)]

if __name__ == "__main__":
    REMOVE_HTML_PASSING_TESTCASES[0]


if __name__ == "__main__":
    REMOVE_HTML_FAILING_TESTCASES[0]


if __name__ == "__main__":
    for html, plain in REMOVE_HTML_FAILING_TESTCASES:
        try:
            remove_html_markup_test(html, plain)
            assert False
        except:
            pass


if __name__ == "__main__":
    for html, plain in REMOVE_HTML_PASSING_TESTCASES:
        remove_html_markup_test(html, plain)


if __name__ == "__main__":
    repairer = Repairer(log=2)


if __name__ == "__main__":
    for html, plain in REMOVE_HTML_PASSING_TESTCASES:
        with repairer.collect_pass():
            remove_html_markup_test(html, plain)


if __name__ == "__main__":
    for html, plain in REMOVE_HTML_FAILING_TESTCASES:
        with repairer.collect_fail():
            remove_html_markup_test(html, plain)


if __name__ == "__main__":
    from IPython.display import HTML


if __name__ == "__main__":
    with ExpectError():
        repairer.code()  # FIXME


if __name__ == "__main__":
    HTML(repairer.code(function=remove_html_markup, color=True))


if __name__ == "__main__":
    best_tree, fitness = repairer.repair()


# ## Mutate Conditions

if __name__ == "__main__":
    print('\n## Mutate Conditions')




class ConditionVisitor(NodeVisitor):
    def __init__(self):
        self.conditions = []
        self.conditions_seen = set()
        super().__init__()

    def add_conditions(self, node, attr):
        elems = getattr(node, attr, [])
        if not isinstance(elems, list):
            elems = [elems]

        for elem in elems:
            elem_str = astor.to_source(elem)
            if elem_str not in self.conditions_seen:
                self.conditions.append(elem)
                self.conditions_seen.add(elem_str)

    def visit_BoolOp(self, node):
        self.add_conditions(node, 'values')
        return super().generic_visit(node)

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            self.add_conditions(node, 'operand')
        return super().generic_visit(node)

    def generic_visit(self, node):
        if hasattr(node, 'test'):
            self.add_conditions(node, 'test')
        return super().generic_visit(node)

def all_conditions(trees, tp=None):
    if not isinstance(trees, list):
        assert isinstance(trees, ast.AST)
        trees = [trees]

    visitor = ConditionVisitor()
    for tree in trees:
        visitor.visit(tree)
    conditions = visitor.conditions
    if tp is not None:
        conditions = [c for c in conditions if isinstance(c, tp)]

    return conditions

if __name__ == "__main__":
    [astor.to_source(cond).strip() for cond in all_conditions(remove_html_markup_tree())]


class ConditionMutator(StatementMutator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditions = all_conditions(self.statements)
        if self.log:
            print("Found conditions",
                  [astor.to_source(cond).strip() for cond in self.conditions])

    def choose_condition(self):
        return copy.deepcopy(random.choice(self.conditions))

class ConditionMutator(ConditionMutator):
    def choose_bool_op(self):
        return random.choice(['set', 'not', 'and', 'or'])

    def swap(self, node):
        if not hasattr(node, 'test'):
            return super().swap(node)

        c1 = self.choose_condition()
        c2 = self.choose_condition()
        new_test = None

        choice = self.choose_bool_op()

        if choice == 'set':
            new_test = c1
        elif choice == 'not':
            new_test = ast.UnaryOp(op=ast.Not(), operand=c1)
        elif choice == 'and':
            new_test = ast.BoolOp(op=ast.And(), values=[c1, c2])
        elif choice == 'or':
            new_test = ast.BoolOp(op=ast.Or(), values=[c1, c2])
        else:
            raise ValueError("Unknown boolean operand")

        if new_test:
            # ast.copy_location(new_test, node)
            node.test = new_test

        return node

if __name__ == "__main__":
    mutator = ConditionMutator(
        statements=all_statements(remove_html_markup_tree()), log=True)


if __name__ == "__main__":
    new_tree = mutator.mutate(remove_html_markup_tree())


if __name__ == "__main__":
    repairer = Repairer(log=2)


if __name__ == "__main__":
    for html, plain in REMOVE_HTML_PASSING_TESTCASES:
        with repairer.collect_pass():
            remove_html_markup_test(html, plain)


if __name__ == "__main__":
    for html, plain in REMOVE_HTML_FAILING_TESTCASES:
        with repairer.collect_fail():
            remove_html_markup_test(html, plain)


if __name__ == "__main__":
    repairer.repair(mutator_cls=ConditionMutator)


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




class PrunePassTransformer(NodeTransformer):
    def prune_pass(self, body):
        if not body:
            return body

        # Get rid of `pass` statements
        new_body = list(filter(lambda stmt: not isinstance(stmt, ast.Pass), body))
        if len(new_body) == 0:
            # Oops – we need at least one `pass`
            new_node = ast.Pass()
            ast.copy_location(new_node, body[0])
            new_body = [new_node]

        return new_body
    
    def visit_node(self, node):
        if hasattr(node, 'body'):
            node.body = self.prune_pass(node.body)
        if hasattr(node, 'orelse'):
            node.orelse = self.prune_pass(node.orelse)
        return node

    def generic_visit(self, node):
        node = super().generic_visit(node)
        return self.visit_node(node)

if __name__ == "__main__":
    prune_passes = PrunePassTransformer()
    prune_passes.visit(best_tree)
    print_content(astor.to_source(best_tree), '.py')


class PruneReturnTransformer(NodeTransformer):
    def ends_in_return(self, body):
        return len(body) > 0 and isinstance(body[-1], ast.Return)

    def prune_returns(self, body):
        if body is None:
            return None

        # Get rid of statements after `return`
        for i, stmt in enumerate(body):
            if isinstance(stmt, ast.Return):
                body = body[:i + 1]
                break
            if isinstance(stmt, ast.If):
                body_returns = self.ends_in_return(stmt.body)
                orelse_returns = self.ends_in_return(stmt.orelse)
                if body_returns and orelse_returns:
                    body = body[:i + 1]
                    break

        return body

    def visit_node(self, node):
        if hasattr(node, 'body'):
            node.body = self.prune_returns(node.body)
        if hasattr(node, 'orelse'):
            node.orelse = self.prune_returns(node.orelse)
        return node

    def generic_visit(self, node):
        node = super().generic_visit(node)
        return self.visit_node(node)

if __name__ == "__main__":
    prune_returns = PruneReturnTransformer()
    prune_returns.visit(best_tree)
    print_content(astor.to_source(best_tree), '.py')

