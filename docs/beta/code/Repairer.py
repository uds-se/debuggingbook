#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Debugging Book".
# Web site: https://www.debuggingbook.org/html/Repairer.html
# Last change: 2021-01-23 13:42:45+01:00
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


# # Repairing Code Automatically

if __name__ == "__main__":
    print('# Repairing Code Automatically')




if __name__ == "__main__":
    from bookutils import YouTubeVideo
    YouTubeVideo("UJTf7cW0idI")


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




# ### The middle() Function

if __name__ == "__main__":
    print('\n### The middle() Function')




if __package__ is None or __package__ == "":
    from StatisticalDebugger import middle
else:
    from .StatisticalDebugger import middle


if __package__ is None or __package__ == "":
    from bookutils import print_content
else:
    from .bookutils import print_content


import inspect

if __name__ == "__main__":
    _, first_lineno = inspect.getsourcelines(middle)
    middle_source = inspect.getsource(middle)
    print_content(middle_source, '.py', start_line_number=first_lineno)


if __name__ == "__main__":
    middle(4, 5, 6)


if __name__ == "__main__":
    middle(2, 1, 3)


# ### Validated Repairs

if __name__ == "__main__":
    print('\n### Validated Repairs')




def middle_sort_of_fixed(x, y, z):
    return x

if __name__ == "__main__":
    middle_sort_of_fixed(2, 1, 3)


# ### Genetic Optimization

if __name__ == "__main__":
    print('\n### Genetic Optimization')




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
    from StatisticalDebugger import OchiaiDebugger, DifferenceDebugger
else:
    from .StatisticalDebugger import OchiaiDebugger, DifferenceDebugger


if __name__ == "__main__":
    middle_debugger = OchiaiDebugger()

    for x, y, z in MIDDLE_PASSING_TESTCASES + MIDDLE_FAILING_TESTCASES:
        with middle_debugger:
            m = middle_test(x, y, z)


if __name__ == "__main__":
    middle_debugger


if __name__ == "__main__":
    location = middle_debugger.rank()[0]
    (func_name, lineno) = location
    lines, first_lineno = inspect.getsourcelines(middle)
    print(lineno, end="")
    print_content(lines[lineno - first_lineno], '.py')


if __name__ == "__main__":
    middle_debugger.suspiciousness(location)


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
        return self.visit_FunctionDef(self, node)

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

    def __init__(self, suspiciousness_func=None, source=None, log=False):
        """Constructor.
`suspiciousness_func` is a function that takes a location
  (function, line_number) and returns a suspiciousness value
  between 0 and 1.0. If not given, all locations get the
  same suspiciousness of 1.0.
`source` is a list of statements to choose from."""

        super().__init__()
        self.log = log

        if suspiciousness_func is None:
            def suspiciousness_func(location):
                return 1.0

        self.suspiciousness_func = suspiciousness_func

        if source is None:
            source = []
        self.source = source

        if self.log > 1:
            for i, node in enumerate(self.source):
                print(f"Source for repairs #{i}:")
                print_content(astor.to_source(node), '.py')
                print()
                print()

        self.mutations = 0

# #### Choosing Suspicious Statements to Mutate

if __name__ == "__main__":
    print('\n#### Choosing Suspicious Statements to Mutate')




import warnings

class StatementMutator(StatementMutator):
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
        stmts = [stmt for stmt, func_name in statements]

        if self.log > 1:
            print("Weights:")
            for i, stmt in enumerate(statements):
                node, func_name = stmt
                print(f"{weights[i]:.2} {self.format_node(node)}")

        if sum(weights) == 0.0:
            # No suspicious line
            return random.choice(stmts)
        else:
            return random.choices(stmts, weights=weights)[0]

# #### Choosing a Mutation Method

if __name__ == "__main__":
    print('\n#### Choosing a Mutation Method')




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
            print(f"{node.lineno:4}:{op.__name__ + ':':7} "
                  f"{self.format_node(node)} "
                  f"becomes {self.format_node(new_node)}")

        return new_node

# #### Swapping Statements

if __name__ == "__main__":
    print('\n#### Swapping Statements')




class StatementMutator(StatementMutator):
    def choose_statement(self):
        return copy.deepcopy(random.choice(self.source))

class StatementMutator(StatementMutator):
    def swap(self, node):
        """Replace `node` with a random node from `source`"""
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
        """Insert a random node from `source` after `node`"""
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
        """Delete `node`."""

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

if __package__ is None or __package__ == "":
    from bookutils import quiz
else:
    from .bookutils import quiz


if __name__ == "__main__":
    quiz("Why are statements replaced by `pass` rather than deleted?",
         [
             "Because `if P: pass` is valid Python, while `if P:` is not",
             "Because in Python, bodies for `if`, `while`, etc. cannot be empty",
             "Because a `pass` node makes a target for future mutations",
             "Because it causes the tests to pass"
         ],
         [3 ^ n for n in range(3)]
        )


# #### Helpers

if __name__ == "__main__":
    print('\n#### Helpers')




class StatementMutator(StatementMutator):
    NODE_MAX_LENGTH = 20

    def format_node(self, node):
        """Return a string representation for `node`."""
        if node is None:
            return "None"

        if isinstance(node, list):
            return "; ".join(self.format_node(elem) for elem in node)

        s = RE_SPACE.sub(' ', astor.to_source(node)).strip()
        if len(s) > self.NODE_MAX_LENGTH - len("..."):
            s = s[:self.NODE_MAX_LENGTH] + "..."
        return repr(s)

# #### All Together

if __name__ == "__main__":
    print('\n#### All Together')




class StatementMutator(StatementMutator):
    def mutate(self, tree):
        """Mutate the given AST `tree` in place. Return mutated tree."""

        assert isinstance(tree, ast.AST)

        tree = copy.deepcopy(tree)

        if not self.source:
            self.source = all_statements(tree)

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
    """Compute fitness of a `middle()` candidate given in `tree`"""
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
    middle_fitness(ast.parse("def middle(x, y, z): return x"))


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




POPULATION_SIZE = 40
middle_mutator = StatementMutator()

MIDDLE_POPULATION = [middle_tree()] + \
    [middle_mutator.mutate(middle_tree()) for i in range(POPULATION_SIZE - 1)]

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

    source = all_statements(middle_tree())
    mutator = StatementMutator(source=source)

    n = len(MIDDLE_POPULATION)

    offspring = []
    while len(offspring) < n:
        parent = random.choice(MIDDLE_POPULATION)
        offspring.append(mutator.mutate(parent))

    MIDDLE_POPULATION += offspring
    MIDDLE_POPULATION.sort(key=middle_fitness, reverse=True)
    MIDDLE_POPULATION = MIDDLE_POPULATION[:n]

if __name__ == "__main__":
    evolve_middle()


if __name__ == "__main__":
    tree = MIDDLE_POPULATION[0]
    print(astor.to_source(tree), middle_fitness(tree))


if __name__ == "__main__":
    for i in range(50):
        evolve_middle()
        best_middle_tree = MIDDLE_POPULATION[0]
        fitness = middle_fitness(best_middle_tree)
        print(f"\rIteration {i:2}: fitness = {fitness}  ", end="")
        if fitness >= 1.0:
            break


if __name__ == "__main__":
    print_content(astor.to_source(best_middle_tree), '.py', start_line_number=1)


if __name__ == "__main__":
    original_middle = middle
    code = compile(best_middle_tree, '<string>', 'exec')
    exec(code, globals())

    for x, y, z in MIDDLE_PASSING_TESTCASES + MIDDLE_FAILING_TESTCASES:
        middle_test(x, y, z)

    middle = original_middle


if __name__ == "__main__":
    quiz("Some of the lines in our fix candidate are redundant. Which are these?",
        [
            "Line 3: `if x < y`",
            "Line 4: `if x > z`",
            "Line 5: `return x`",
            "Line 13: `return z`"
        ],
        [eval(chr(100 - x)) for x in [49, 50]]
        )


# ## Simplifying

if __name__ == "__main__":
    print('\n## Simplifying')




if __package__ is None or __package__ == "":
    from DeltaDebugger import DeltaDebugger
else:
    from .DeltaDebugger import DeltaDebugger


if __name__ == "__main__":
    middle_lines = astor.to_source(best_middle_tree).strip().split('\n')


def test_middle_lines(lines):
    source = "\n".join(lines)
    tree = ast.parse(source)
    assert middle_fitness(tree) < 1.0  # "Fail" only while fitness is 1.0

if __name__ == "__main__":
    with DeltaDebugger() as dd:
        test_middle_lines(middle_lines)


if __name__ == "__main__":
    reduced_lines = dd.min_args()['lines']


if __name__ == "__main__":
    reduced_source = "\n".join(reduced_lines)


if __name__ == "__main__":
    repaired_source = astor.to_source(ast.parse(reduced_source))  # normalize
    print_content(repaired_source, '.py')


if __name__ == "__main__":
    original_source = astor.to_source(ast.parse(middle_source))  # normalize


if __package__ is None or __package__ == "":
    from ChangeDebugger import diff, print_patch  # minor dependency
else:
    from .ChangeDebugger import diff, print_patch  # minor dependency


if __name__ == "__main__":
    for patch in diff(original_source, repaired_source):
        print_patch(patch)


# ## Crossover

if __name__ == "__main__":
    print('\n## Crossover')




# ### Excursion: Implementing Crossover

if __name__ == "__main__":
    print('\n### Excursion: Implementing Crossover')




# #### Crossing Statement Lists

if __name__ == "__main__":
    print('\n#### Crossing Statement Lists')




def p1():
    a = 1
    b = 2
    c = 3

def p2():
    x = 1
    y = 2
    z = 3

class CrossoverOperator:
    def __init__(self, log=False):
        """Constructor. If `log` is set, turn on logging."""
        self.log = log

    def cross_bodies(self, body_1, body_2):
        """CrossoverOperator the statement lists `body_1` x `body_2`.
        Return new lists.
        """

        assert isinstance(body_1, list)
        assert isinstance(body_2, list)

        crossover_point_1 = len(body_1) // 2
        crossover_point_2 = len(body_2) // 2
        return (body_1[:crossover_point_1] + body_2[crossover_point_2:],
                body_2[:crossover_point_2] + body_1[crossover_point_1:])

if __name__ == "__main__":
    tree_p1 = ast.parse(inspect.getsource(p1))
    tree_p2 = ast.parse(inspect.getsource(p2))


if __name__ == "__main__":
    body_p1 = tree_p1.body[0].body
    body_p2 = tree_p2.body[0].body
    body_p1


if __name__ == "__main__":
    crosser = CrossoverOperator()
    tree_p1.body[0].body, tree_p2.body[0].body = crosser.cross_bodies(body_p1, body_p2)


if __name__ == "__main__":
    print_content(astor.to_source(tree_p1), '.py')


if __name__ == "__main__":
    print_content(astor.to_source(tree_p2), '.py')


# #### Applying Crossover on Programs

if __name__ == "__main__":
    print('\n#### Applying Crossover on Programs')




class CrossoverOperator(CrossoverOperator):
    # In modules and class defs, the ordering of elements does not matter (much)
    SKIP_LIST = {ast.Module, ast.ClassDef}

    def can_cross(self, tree, body_attr='body'):
        if any(isinstance(tree, cls) for cls in self.SKIP_LIST):
            return False

        body = getattr(tree, body_attr, [])
        return body and len(body) >= 2

class CrossoverOperator(CrossoverOperator):
    def crossover_attr(self, t1, t2, body_attr):
        """CrossoverOperator the bodies `body_attr` of two trees `t1` and `t2`.
        Return True if successful."""
        assert isinstance(t1, ast.AST)
        assert isinstance(t2, ast.AST)
        assert isinstance(body_attr, str)

        if not getattr(t1, body_attr, None) or not getattr(t2, body_attr, None):
            return False

        if self.crossover_branches(t1, t2):
            return t1, t2

        if self.log > 1:
            print(f"Checking {t1}.{body_attr} x {t2}.{body_attr}")

        body_1 = getattr(t1, body_attr)
        body_2 = getattr(t2, body_attr)

        # If both trees have the attribute, we can cross their bodies
        if self.can_cross(t1, body_attr) and self.can_cross(t2, body_attr):
            if self.log:
                print(f"Crossing {t1}.{body_attr} x {t2}.{body_attr}")

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
                        if self.crossover_attr(child_1, child_2, body_attr):
                            return True

        # Strategy 2: Find matches anywhere
        for child_1 in random.sample(body_1, len(body_1)):
            for child_2 in random.sample(body_2, len(body_2)):
                if self.crossover_attr(child_1, child_2, body_attr):
                    return True

        return False

class CrossoverOperator(CrossoverOperator):
    def crossover_branches(self, t1, t2):
        """Special case:
        `t1` = `if P: S1 else: S2` x `t2` = `if P': S1' else: S2'`
        becomes
        `t1` = `if P: S2' else: S1'` and `t2` = `if P': S2 else: S1`
        Returns True if successful.
        """
        assert isinstance(t1, ast.AST)
        assert isinstance(t2, ast.AST)

        if (getattr(t1, 'body', None) and getattr(t1, 'orelse', None) and
            getattr(t2, 'body', None) and getattr(t2, 'orelse', None)):

            if self.log:
                print(f"Crossing branches {t1} x {t2}")

            t1.body, t1.orelse, t2.body, t2.orelse = \
                t2.orelse, t2.body, t1.orelse, t1.body
            return True

        return False

class CrossoverOperator(CrossoverOperator):
    def crossover(self, t1, t2):
        """Do a crossover of ASTs `t1` and `t2`.
        Raises `CrossoverError` if no crossover is found."""
        assert isinstance(t1, ast.AST)
        assert isinstance(t2, ast.AST)

        for body_attr in ['body', 'orelse', 'finalbody']:
            if self.crossover_attr(t1, t2, body_attr):
                return t1, t2

        raise CrossoverError("No crossover found")

class CrossoverError(ValueError):
    pass

# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ### Crossover in Action

if __name__ == "__main__":
    print('\n### Crossover in Action')




def p1():
    if True:
        print(1)
        print(2)
        print(3)

def p2():
    if True:
        print(a)
        print(b)
    else:
        print(c)
        print(d)

if __name__ == "__main__":
    crossover = CrossoverOperator()
    tree_p1 = ast.parse(inspect.getsource(p1))
    tree_p2 = ast.parse(inspect.getsource(p2))
    crossover.crossover(tree_p1, tree_p2);


if __name__ == "__main__":
    print_content(astor.to_source(tree_p1), '.py')


if __name__ == "__main__":
    print_content(astor.to_source(tree_p2), '.py')


if __name__ == "__main__":
    middle_t1, middle_t2 = crossover.crossover(middle_tree(),
                                              ast.parse(inspect.getsource(p2)))


if __name__ == "__main__":
    print_content(astor.to_source(middle_t1), '.py')


if __name__ == "__main__":
    print_content(astor.to_source(middle_t2), '.py')


# ## A Repairer Class

if __name__ == "__main__":
    print('\n## A Repairer Class')




# ### Excursion: Implementing Repairer

if __name__ == "__main__":
    print('\n### Excursion: Implementing Repairer')




class Repairer():
    def __init__(self, debugger, targets=None, sources=None, log=False,
                 mutator_class=StatementMutator,
                 crossover_class=CrossoverOperator,
                 reducer_class=DeltaDebugger,
                 globals=None):
        """Constructor.
`debugger`: a `DifferenceDebugger` to take tests and coverage from.
`targets`: a list of functions/modules to be repaired.
    (default: the covered functions in `debugger`, except tests)
`sources`: a list of functions/modules to take repairs from.
    (default: same as `targets`)
`globals`: if given, a `globals()` dict for executing targets
    (default: `globals()` of caller)"""

        assert isinstance(debugger, DifferenceDebugger)
        self.debugger = debugger
        self.log = log

        if targets is None:
            targets = self.default_functions()
        if not targets:
            raise ValueError("No targets to repair")

        if sources is None:
            sources = self.default_functions()
        if not sources:
            raise ValueError("No sources to take repairs from")

        if self.debugger.function() is None:
            raise ValueError("Multiple entry points observed")

        self.target_tree = self.parse(targets)
        self.source_tree = self.parse(sources)

        self.log_tree("Target code to be repaired:", self.target_tree)
        if ast.dump(self.target_tree) != ast.dump(self.source_tree):
            self.log_tree("Source code to take repairs from:", 
                          self.source_tree)

        self.fitness_cache = {}

        self.mutator = \
            mutator_class(
                source=all_statements(self.source_tree),
                suspiciousness_func=self.debugger.suspiciousness,
                log=(self.log >= 3))
        self.crossover = crossover_class(log=(self.log >= 3))
        self.reducer = reducer_class(log=(self.log >= 3))

        if globals is None:
            globals = self.caller_globals()
        self.globals = globals

# #### Helper Functions

if __name__ == "__main__":
    print('\n#### Helper Functions')




if __package__ is None or __package__ == "":
    from Slicer import StackInspector  # minor dependency
else:
    from .Slicer import StackInspector  # minor dependency


class Repairer(Repairer, StackInspector):
    pass

class Repairer(Repairer):
    def getsource(self, item):
        """Get the source for `item`. Can also be a string."""
        if isinstance(item, str):
            item = self.globals[item]
        return inspect.getsource(item)

class Repairer(Repairer):
    def default_functions(self):
        """Return the set of functions to be repaired.
        Functions whose names start or end in `test` are excluded."""
        def is_test(name):
            return name.startswith('test') or name.endswith('test')

        return [func for func in self.debugger.covered_functions()
                if not is_test(func.__name__)]

class Repairer(Repairer):
    def log_tree(self, description, tree):
        """Print out `tree` as source code prefixed by `description`."""
        if self.log:
            print(description)
            print_content(astor.to_source(tree), '.py')
            print()
            print()

class Repairer(Repairer):
    def parse(self, items):
        """Read in a set of items into a single tree"""
        tree = ast.parse("")
        for item in items:
            if isinstance(item, str):
                item = self.globals[item]

            item_lines, item_first_lineno = inspect.getsourcelines(item)

            try:
                item_tree = ast.parse("".join(item_lines))
            except IndentationError:
                # inner function or likewise
                warnings.warn(f"Can't parse {item.__name__}")
                continue

            ast.increment_lineno(item_tree, item_first_lineno - 1)
            tree.body += item_tree.body

        return tree

# #### Running Tests

if __name__ == "__main__":
    print('\n#### Running Tests')




class Repairer(Repairer):
    def run_test_set(self, test_set, validate=False):
        """Run given `test_set`
        (`DifferenceDebugger.PASS` or `DifferenceDebugger.FAIL`).
        If `validate` is set, check expectations.
        Return number of passed tests."""
        passed = 0
        collectors = self.debugger.collectors[test_set]
        function = self.debugger.function()
        # FIXME: function may have been redefined

        for c in collectors:
            if self.log >= 4:
                print(f"Testing {c.id()}...", end="")

            try:
                function(**c.args())
            except Exception as err:
                if self.log >= 4:
                    print(f"failed ({err.__class__.__name__})")

                if validate and test_set == self.debugger.PASS:
                    raise err.__class__(
                        f"{c.id()} should have passed, but failed")
                continue

            passed += 1
            if self.log >= 4:
                print("passed")

            if validate and test_set == self.debugger.FAIL:
                raise FailureNotReproducedError(
                    f"{c.id()} should have failed, but passed")

        return passed

class FailureNotReproducedError(ValueError):
    pass

if __name__ == "__main__":
    repairer = Repairer(middle_debugger)
    assert repairer.run_test_set(middle_debugger.PASS) == \
        len(MIDDLE_PASSING_TESTCASES)
    assert repairer.run_test_set(middle_debugger.FAIL) == 0


class Repairer(Repairer):
    def weight(self, test_set):
        """Return the weight of `test_set`
        (`DifferenceDebugger.PASS` or `DifferenceDebugger.FAIL`).
        """
        return {
            self.debugger.PASS: WEIGHT_PASSING,
            self.debugger.FAIL: WEIGHT_FAILING
        }[test_set]

    def run_tests(self, validate=False):
        """Run passing and failing tests, returning weighted fitness."""
        fitness = 0

        for test_set in [self.debugger.PASS, self.debugger.FAIL]:
            passed = self.run_test_set(test_set, validate=validate)
            ratio = passed / len(self.debugger.collectors[test_set])
            fitness += self.weight(test_set) * ratio

        return fitness

class Repairer(Repairer):
    def validate(self):
        fitness = self.run_tests(validate=True)
        assert fitness == self.weight(self.debugger.PASS)

if __name__ == "__main__":
    repairer = Repairer(middle_debugger)
    repairer.validate()


# #### (Re)defining Functions

if __name__ == "__main__":
    print('\n#### (Re)defining Functions')




class Repairer(Repairer):
    def fitness(self, tree):
        """Test `tree`, returning its fitness"""
        key = ast.dump(tree)
        if key in self.fitness_cache:
            return self.fitness_cache[key]

        # Save defs
        original_defs = {}
        for name in self.toplevel_defs(tree):
            if name in self.globals:
                original_defs[name] = self.globals[name]
            else:
                warnings.warn(f"Couldn't find definition of {repr(name)}")

        assert original_defs, f"Couldn't find any definition"

        if self.log >= 3:
            print("Repair candidate:")
            print_content(astor.to_source(tree), '.py')
            print()

        # Create new definition
        try:
            code = compile(tree, '<Repairer>', 'exec')
        except ValueError:  # Compilation error
            code = None

        if code is None:
            if self.log >= 3:
                print(f"Fitness = 0.0 (compilation error)")

            fitness = 0.0
            return fitness

        # Execute new code, defining new functions in `self.globals`
        exec(code, self.globals)

        # Set new definitions in the namespace (`__globals__`)
        # of the function we will be calling.
        function = self.debugger.function()
        for name in original_defs:
            function.__globals__[name] = self.globals[name]

        fitness = self.run_tests(validate=False)

        # Restore definitions
        for name in original_defs:
            function.__globals__[name] = original_defs[name]
            self.globals[name] = original_defs[name]

        if self.log >= 3:
            print(f"Fitness = {fitness}")

        self.fitness_cache[key] = fitness
        return fitness

class Repairer(Repairer):
    def toplevel_defs(self, tree):
        """Return a list of defined functions and classes in `tree`"""
        visitor = DefinitionVisitor()
        visitor.visit(tree)
        return visitor.definitions

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

if __name__ == "__main__":
    repairer = Repairer(middle_debugger, log=4)


if __name__ == "__main__":
    good_fitness = repairer.fitness(middle_tree())
    good_fitness


if __name__ == "__main__":
    assert good_fitness >= 0.99, "fitness() failed"


if __name__ == "__main__":
    bad_middle_tree = ast.parse("def middle(x, y, z): return x")
    bad_fitness = repairer.fitness(bad_middle_tree)
    bad_fitness


if __name__ == "__main__":
    assert bad_fitness < 0.5, "fitness() failed"


# #### Repairing

if __name__ == "__main__":
    print('\n#### Repairing')




class Repairer(Repairer):
    def initial_population(self, size):
        """Return an initial population of size `size`"""
        return [self.target_tree] + \
            [self.mutator.mutate(copy.deepcopy(self.target_tree))
                for i in range(size - 1)]

    def repair(self, population_size=POPULATION_SIZE, iterations=100):
        """Repair the function we collected test runs from.
        Use a population size of `population_size` and
        at most `iterations` iterations.
        Returns the AST of the repaired function."""
        self.validate()

        population = self.initial_population(population_size)

        last_key = ast.dump(self.target_tree)

        for iteration in range(iterations):
            population = self.evolve(population)

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

# #### Evolving

if __name__ == "__main__":
    print('\n#### Evolving')




class Repairer(Repairer):
    def evolve(self, population):
        """Evolve the candidate population by mutating and crossover."""
        n = len(population)

        # Create offspring as crossover of parents
        offspring = []
        while len(offspring) < n:
            parent_1 = copy.deepcopy(random.choice(population))
            parent_2 = copy.deepcopy(random.choice(population))
            try:
                self.crossover.crossover(parent_1, parent_2)
            except CrossoverError:
                pass  # Just keep parents
            offspring += [parent_1, parent_2]

        # Mutate offspring
        offspring = [self.mutator.mutate(tree) for tree in offspring]

        # Add it to population
        population += offspring

        # Keep the fitter part of the population
        population.sort(key=self.fitness_key, reverse=True)
        population = population[:n]

        return population

class Repairer(Repairer):
    def fitness_key(self, tree):
        """Key to be used for sorting the population"""
        tree_size = len([node for node in ast.walk(tree)])
        return (self.fitness(tree), -tree_size)

# #### Simplifying

if __name__ == "__main__":
    print('\n#### Simplifying')




class Repairer(Repairer):
    def reduce(self, tree):
        """Simplify `tree` using delta debugging."""

        original_fitness = self.fitness(tree)
        source_lines = astor.to_source(tree).split('\n')

        with self.reducer:
            self.test_reduce(source_lines, original_fitness)

        reduced_lines = self.reducer.min_args()['source_lines']
        reduced_source = "\n".join(reduced_lines)

        return ast.parse(reduced_source)

import traceback

class Repairer(Repairer):
    def test_reduce(self, source_lines, original_fitness):
        """Test function for delta debugging."""

        try:
            source = "\n".join(source_lines)
            tree = ast.parse(source)
            fitness = self.fitness(tree)
            assert fitness < original_fitness

        except AssertionError:
            raise
        except SyntaxError:
            raise
        except IndentationError:
            raise
        except Exception:
            # traceback.print_exc()  # Uncomment to see internal errors
            raise

# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ### Repairer in Action

if __name__ == "__main__":
    print('\n### Repairer in Action')




if __name__ == "__main__":
    repairer = Repairer(middle_debugger, log=True)


if __name__ == "__main__":
    best_tree, fitness = repairer.repair()


if __name__ == "__main__":
    print_content(astor.to_source(best_tree), '.py')


if __name__ == "__main__":
    fitness


# ## Removing HTML Markup

if __name__ == "__main__":
    print('\n## Removing HTML Markup')




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

# ### Excursion: Creating HTML Test Cases

if __name__ == "__main__":
    print('\n### Excursion: Creating HTML Test Cases')




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

# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




if __name__ == "__main__":
    REMOVE_HTML_PASSING_TESTCASES[0]


if __name__ == "__main__":
    html, plain = REMOVE_HTML_PASSING_TESTCASES[0]
    remove_html_markup_test(html, plain)


if __name__ == "__main__":
    REMOVE_HTML_FAILING_TESTCASES[0]


if __name__ == "__main__":
    with ExpectError():
        html, plain = REMOVE_HTML_FAILING_TESTCASES[0]
        remove_html_markup_test(html, plain)


if __name__ == "__main__":
    html_debugger = OchiaiDebugger()


if __name__ == "__main__":
    for html, plain in (REMOVE_HTML_PASSING_TESTCASES + 
                        REMOVE_HTML_FAILING_TESTCASES):
        with html_debugger:
            remove_html_markup_test(html, plain)


if __name__ == "__main__":
    html_debugger


if __name__ == "__main__":
    html_repairer = Repairer(html_debugger, log=True)


if __name__ == "__main__":
    best_tree, fitness = html_repairer.repair()


if __name__ == "__main__":
    quiz("Why couldn't `Repairer()` repair `remove_html_markup()`?",
         [
             "The population is too small!",
             "The suspiciousness is too evenly distributed!",
             "We need more test cases!",
             "We need more iterations!",
             "There is no statement in the source with a correct condition!",
             "The population is too big!",
         ], 5242880 >> 20)


# ## Mutating Conditions

if __name__ == "__main__":
    print('\n## Mutating Conditions')




# ### Collecting Conditions

if __name__ == "__main__":
    print('\n### Collecting Conditions')




def all_conditions(trees, tp=None):
    """Return all conditions from the AST (or AST list) `trees`.
    If `tp` is given, return only elements of that type."""
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

if __name__ == "__main__":
    [astor.to_source(cond).strip()
        for cond in all_conditions(remove_html_markup_tree())]


# ### Mutating Conditions

if __name__ == "__main__":
    print('\n### Mutating Conditions')




class ConditionMutator(StatementMutator):
    """Mutate conditions in an AST"""

    def __init__(self, *args, **kwargs):
        """Constructor. Arguments are as with `StatementMutator` constructor."""
        super().__init__(*args, **kwargs)
        self.conditions = all_conditions(self.source)
        if self.log:
            print("Found conditions",
                  [astor.to_source(cond).strip() 
                   for cond in self.conditions])

    def choose_condition(self):
        """Return a random condition from source."""
        return copy.deepcopy(random.choice(self.conditions))

class ConditionMutator(ConditionMutator):
    def choose_bool_op(self):
        return random.choice(['set', 'not', 'and', 'or'])

    def swap(self, node):
        """Replace `node` condition by a condition from `source`"""
        if not hasattr(node, 'test'):
            return super().swap(node)

        cond = self.choose_condition()
        new_test = None

        choice = self.choose_bool_op()

        if choice == 'set':
            new_test = cond
        elif choice == 'not':
            new_test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        elif choice == 'and':
            new_test = ast.BoolOp(op=ast.And(), values=[cond, node.test])
        elif choice == 'or':
            new_test = ast.BoolOp(op=ast.Or(), values=[cond, node.test])
        else:
            raise ValueError("Unknown boolean operand")

        if new_test:
            # ast.copy_location(new_test, node)
            node.test = new_test

        return node

if __name__ == "__main__":
    mutator = ConditionMutator(source=all_statements(remove_html_markup_tree()),
                               log=True)


if __name__ == "__main__":
    for i in range(10):
        new_tree = mutator.mutate(remove_html_markup_tree())


if __name__ == "__main__":
    condition_repairer = Repairer(html_debugger,
                                  mutator_class=ConditionMutator,
                                  log=2)


if __name__ == "__main__":
    best_tree, fitness = condition_repairer.repair(iterations=200)


if __name__ == "__main__":
    repaired_source = astor.to_source(best_tree)


if __name__ == "__main__":
    print_content(repaired_source, '.py')


if __name__ == "__main__":
    original_source = astor.to_source(remove_html_markup_tree())


if __name__ == "__main__":
    for patch in diff(original_source, repaired_source):
        print_patch(patch)


if __name__ == "__main__":
    quiz("Is this actually the best solution?",
        [
            "Yes, sure, of course. Why?",
            "Err - what happened to single quotes?"
        ], 1 << 1)


if __name__ == "__main__":
    quiz("Why aren't single quotes handled in the solution?",
        [
            "Because they're not important. I mean, who uses 'em anyway?",
            "Because they are not part of our tests? "
                "Let me look up how they are constructed..."
        ], 1 << 1)


if __name__ == "__main__":
    remove_html_markup("<foo quote='>abc'>\"me\"</foo>")


if __name__ == "__main__":
    REMOVE_HTML_PASSING_TESTCASES.append(("<foo quote='>abc'>\"me\"</foo>", '"me"'))


if __name__ == "__main__":
    best_tree, fitness = condition_repairer.repair(iterations=200)


if __name__ == "__main__":
    print_content(astor.to_source(best_tree), '.py')


if __name__ == "__main__":
    fitness


# ## Limitations

if __name__ == "__main__":
    print('\n## Limitations')




# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




if __name__ == "__main__":
    print_content(middle_source, '.py')


if __name__ == "__main__":
    middle_debugger = OchiaiDebugger()


if __name__ == "__main__":
    for x, y, z in MIDDLE_PASSING_TESTCASES + MIDDLE_FAILING_TESTCASES:
        with middle_debugger:
            m = middle_test(x, y, z)


if __name__ == "__main__":
    middle_repairer = Repairer(middle_debugger)
    tree, fitness = middle_repairer.repair()
    print(astor.to_source(tree), fitness)


if __package__ is None or __package__ == "":
    from ClassDiagram import display_class_hierarchy
else:
    from .ClassDiagram import display_class_hierarchy


if __name__ == "__main__":
    display_class_hierarchy([Repairer, ConditionMutator, CrossoverOperator],
                            public_methods=[
                                Repairer.__init__,
                                Repairer.repair,
                                StatementMutator.__init__,
                                StatementMutator.mutate,
                                ConditionMutator.__init__,
                                CrossoverOperator.__init__,
                                CrossoverOperator.crossover,
                            ],
                            project='debuggingbook')


# ## Lessons Learned

if __name__ == "__main__":
    print('\n## Lessons Learned')




# ## Background

if __name__ == "__main__":
    print('\n## Background')




# ## Exercises

if __name__ == "__main__":
    print('\n## Exercises')




# ### Exercise 1: Automated Repair Parameters

if __name__ == "__main__":
    print('\n### Exercise 1: Automated Repair Parameters')




# ### Exercise 2: Elitism

if __name__ == "__main__":
    print('\n### Exercise 2: Elitism')




# ### Exercise 3: Evolving Values

if __name__ == "__main__":
    print('\n### Exercise 3: Evolving Values')




if __package__ is None or __package__ == "":
    from Assertions import square_root  # minor dependency
else:
    from .Assertions import square_root  # minor dependency


if __name__ == "__main__":
    with ExpectError():
        square_root_of_zero = square_root(0)


import math

def square_root_fixed(x):
    assert x >= 0  # precondition

    approx = 0  # <-- FIX: Change `None` to 0
    guess = x / 2
    while approx != guess:
        approx = guess
        guess = (approx + x / approx) / 2

    assert math.isclose(approx * approx, x)
    return approx

if __name__ == "__main__":
    square_root_fixed(0)


# ### Exercise 4: Evolving Variable Names

if __name__ == "__main__":
    print('\n### Exercise 4: Evolving Variable Names')




# ### Exercise 5: Parallel Repair

if __name__ == "__main__":
    print('\n### Exercise 5: Parallel Repair')



