#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Debugging Book".
# Web site: https://www.debuggingbook.org/html/DDSet.html
# Last change: 2021-01-30 18:21:45+01:00
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


# # The Language of Failure

if __name__ == "__main__":
    print('# The Language of Failure')




if __name__ == "__main__":
    from bookutils import YouTubeVideo
    YouTubeVideo("w4u5gCgPlmg")


if __name__ == "__main__":
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)


if __package__ is None or __package__ == "":
    import DeltaDebugger
else:
    from . import DeltaDebugger


# ## A Failing Program

if __name__ == "__main__":
    print('\n## A Failing Program')




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

    # postcondition
    assert '<' not in out and '>' not in out

    return out

if __package__ is None or __package__ == "":
    from ExpectError import ExpectError
else:
    from .ExpectError import ExpectError


BAD_INPUT = '<foo>"bar</foo>'

if __name__ == "__main__":
    with ExpectError(AssertionError):
        remove_html_markup(BAD_INPUT)


# ## Grammars

if __name__ == "__main__":
    print('\n## Grammars')




import string

SIMPLE_HTML_GRAMMAR = {
    "<start>":
        ["<html>"],

    "<html>":
        ["<plain-text>", "<tagged-text>"],

    "<plain-text>":
        ["", "<plain-char><plain-text>"],

    "<plain-char>":
        ["<letter>", "<digit>", "<other>"],

    "<tagged-text>":
        ["<opening-tag><html><closing-tag>",
         "<combined-tag>",
         "<opening-tag>"],

    "<opening-tag>":
        ["<lt><id><gt>",
         "<lt><id><attrs><gt>"],

    "<closing-tag>":
        ["<lt>/<id><gt>"],

    "<combined-tag>":
        ["<lt><id><attrs>/<gt>"],

    "<lt>": [ "<" ],
    "<gt>": [ ">" ],

    "<attrs>":
        ["<attr>", "<attr> <attrs>" ],

    "<attr>":
        ["<id>='<plain-text>'",
         '<id>="<plain-text>"'],

    "<id>":
        ["<letter>", "<id><letter>", "<id><digit>"],

    "<letter>": list(string.ascii_letters),
    "<digit>": list(string.digits),
    "<other>": list(string.punctuation.replace('<', '').replace('>', ''))
}

# ## Fuzzing

if __name__ == "__main__":
    print('\n## Fuzzing')




from fuzzingbook.GrammarFuzzer import GrammarFuzzer

if __name__ == "__main__":
    simple_html_fuzzer = GrammarFuzzer(SIMPLE_HTML_GRAMMAR)


if __name__ == "__main__":
    for i in range(20):
        fuzz_html = simple_html_fuzzer.fuzz()
        print(repr(fuzz_html))


# ## Derivation Trees

if __name__ == "__main__":
    print('\n## Derivation Trees')




from fuzzingbook.Parser import display_tree

if __name__ == "__main__":
    fuzz_html


if __name__ == "__main__":
    display_tree(simple_html_fuzzer.derivation_tree)


# ## Parsing

if __name__ == "__main__":
    print('\n## Parsing')




from fuzzingbook.Parser import EarleyParser

if __name__ == "__main__":
    simple_html_parser = EarleyParser(SIMPLE_HTML_GRAMMAR)


if __name__ == "__main__":
    bad_input_tree = list(simple_html_parser.parse(BAD_INPUT))[0]


if __name__ == "__main__":
    display_tree(bad_input_tree)


from fuzzingbook.GrammarFuzzer import tree_to_string, all_terminals

if __name__ == "__main__":
    tree_to_string(bad_input_tree)


# ## Mutating the Tree

if __name__ == "__main__":
    print('\n## Mutating the Tree')




from fuzzingbook.Grammars import is_valid_grammar

class TreeMutator:
    def __init__(self, grammar, tree, log=False):
        assert is_valid_grammar(grammar)
        self.grammar = grammar
        self.tree = tree
        self.log = log

class TreeMutator(TreeMutator):
    def get_subtree(self, path, tree=None):
        if tree is None:
            tree = self.tree

        node, children = tree

        if not path:
            return tree

        return self.get_subtree(path[1:], children[path[0]])

def bad_input_tree_mutator():
    return TreeMutator(SIMPLE_HTML_GRAMMAR, bad_input_tree, log=2)    

if __name__ == "__main__":
    bad_input_tree_mutator().get_subtree([0, 0, 1, 0])


class TreeMutator(TreeMutator):
    def new_tree(self, start_symbol):
        if self.log >= 2:
            print(f"Creating new tree for {start_symbol}")

        fuzzer = GrammarFuzzer(self.grammar, start_symbol=start_symbol)
        fuzzer.fuzz()
        return fuzzer.derivation_tree

if __name__ == "__main__":
    bad_input_tree_mutator().new_tree('<plain-text>')


class TreeMutator(TreeMutator):
    def mutate(self, path, tree=None):
        if tree is None:
            tree = self.tree

        node, children = tree

        if not path:
            return self.new_tree(node)

        head = path[0]
        new_children = (children[:head] +
                        [self.mutate(path[1:], children[head])] +
                        children[head + 1:])
        return node, new_children

if __name__ == "__main__":
    bad_input_tree_mutator().mutate([0])


if __name__ == "__main__":
    tree_to_string(bad_input_tree_mutator().mutate([0, 0, 1, 0]))


# ## Generalizing Trees

if __name__ == "__main__":
    print('\n## Generalizing Trees')




class TreeGeneralizer(TreeMutator):
    def __init__(self, grammar, tree, test,
                 max_tries_for_abstraction=10,
                 **kwargs):
        super().__init__(grammar, tree, **kwargs)
        self.test = test
        self.max_tries_for_abstraction = max_tries_for_abstraction

class TreeGeneralizer(TreeGeneralizer):
    def test_tree(self, tree):
        s = tree_to_string(tree)
        if self.log:
            print(f"Testing {repr(s)}...", end="")
        try:
            self.test(s)
        except Exception as exc:
            if self.log:
                print(f"FAIL ({type(exc).__name__})")
            ret = False
        else:
            if self.log:
                print(f"PASS")
            ret = True

        return ret

class TreeGeneralizer(TreeGeneralizer):
    def can_generalize(self, path, tree=None):
        for i in range(self.max_tries_for_abstraction):
            mutated_tree = self.mutate(path, tree)
            if self.test_tree(mutated_tree):
                # Failure no longer occurs; cannot abstract
                return False
            
        return True

def bad_input_tree_generalizer():
    return TreeGeneralizer(SIMPLE_HTML_GRAMMAR, bad_input_tree,
                           remove_html_markup, log=True)    

if __name__ == "__main__":
    bad_input_tree_generalizer().can_generalize([0])


if __name__ == "__main__":
    bad_input_tree_generalizer().can_generalize([0, 0, 1, 0])


if __name__ == "__main__":
    bad_input_tree_generalizer().can_generalize([0, 0, 0])


class TreeGeneralizer(TreeGeneralizer):
    def find_paths(self, predicate, path=None, tree=None):
        if path is None:
            path = []
        if tree is None:
            tree = self.tree
            
        node, children = self.get_subtree(path)

        if predicate(path, tree):
            if self.log:
                node, children = self.get_subtree(path)
            return [path]

        paths = []
        for i, child in enumerate(children):
            child_node, _ = child
            if child_node in self.grammar:
                paths += self.find_paths(predicate, path + [i])

        return paths        
    
    def generalizable_paths(self):
        return self.find_paths(self.can_generalize)

if __name__ == "__main__":
    bad_input_tree_generalizer().generalizable_paths()


class TreeGeneralizer(TreeGeneralizer):
    def generalize_path(self, path, tree=None):
        if tree is None:
            tree = self.tree

        node, children = tree

        if not path:
            return node, []

        head = path[0]
        new_children = (children[:head] +
                        [self.generalize_path(path[1:], children[head])] +
                        children[head + 1:])
        return node, new_children

if __name__ == "__main__":
    all_terminals(bad_input_tree_generalizer().generalize_path([0, 0, 0]))


class TreeGeneralizer(TreeGeneralizer):
    def generalize(self):
        tree = self.tree
        for path in self.generalizable_paths():
            tree = self.generalize_path(path, tree)
            
        return tree

if __name__ == "__main__":
    all_terminals(bad_input_tree_generalizer().generalize())


# ## Putting it all Together

if __name__ == "__main__":
    print('\n## Putting it all Together')




if __package__ is None or __package__ == "":
    from DeltaDebugger import CallCollector, is_reducible
else:
    from .DeltaDebugger import CallCollector, is_reducible


import copy

class DDSetDebugger(CallCollector):
    def __init__(self, grammar, 
                 generalizer=TreeGeneralizer,
                 parser=EarleyParser,
                 **kwargs):
        super().__init__(**kwargs)
        self.grammar = grammar
        self.parser = parser(grammar)
        self.generalizer = generalizer

    def generalized_args(self, **kwargs):
        generalized_args = copy.deepcopy(self.args())

        for arg in self.args():
            def test(value):
                return self.call({arg: value})

            value = self.args()[arg]
            if is_reducible(value):
                tree = list(self.parser.parse(value))[0]
                gen = self.generalizer(self.grammar, tree, test, **kwargs)
                generalized_args[arg] = all_terminals(gen.generalize())

        return generalized_args

if __name__ == "__main__":
    with DDSetDebugger(SIMPLE_HTML_GRAMMAR) as dd:
        remove_html_markup(BAD_INPUT)


if __name__ == "__main__":
    dd.generalized_args()['s']


# ## More Examples

if __name__ == "__main__":
    print('\n## More Examples')




# ### Square Root

if __name__ == "__main__":
    print('\n### Square Root')




if __package__ is None or __package__ == "":
    from Assertions import square_root
else:
    from .Assertions import square_root


INT_GRAMMAR = {
    "<start>":
        ["<int>"],

    "<int>":
        ["<positive-int>", "-<positive-int>"],

    "<positive-int>":
        ["<digit>", "<nonzero-digit><positive-int>"],

    "<nonzero-digit>": list("123456789"),
    
    "<digit>": list(string.digits),
}

def square_root_test(s):
    return square_root(int(s))

if __name__ == "__main__":
    with DDSetDebugger(INT_GRAMMAR) as dd:
        square_root_test("-1")


if __name__ == "__main__":
    dd.generalized_args(log=True)['s']


# ### Middle

if __name__ == "__main__":
    print('\n### Middle')




if __package__ is None or __package__ == "":
    from StatisticalDebugger import middle
else:
    from .StatisticalDebugger import middle


XYZ_GRAMMAR = {
    "<start>":
        ["<int>, <int>, <int>"],

    "<int>":
        ["<positive-int>", "-<positive-int>"],

    "<positive-int>":
        ["<digit>", "<nonzero-digit><positive-int>"],

    "<nonzero-digit>": list("123456789"),
    
    "<digit>": list(string.digits),
}

def middle_test(s):
    x, y, z = eval(s)
    assert middle(x, y, z) == sorted([x, y, z])[1]

if __name__ == "__main__":
    with ExpectError(AssertionError):
        middle_test("2, 1, 3")


if __name__ == "__main__":
    with DDSetDebugger(XYZ_GRAMMAR) as dd:
        middle_test("2, 1, 3")


if __name__ == "__main__":
    dd.generalized_args(log=True)['s']


# # Synopsis

if __name__ == "__main__":
    print('\n# Synopsis')




# ## _Section 1_

if __name__ == "__main__":
    print('\n## _Section 1_')




# ## _Section 2_

if __name__ == "__main__":
    print('\n## _Section 2_')




# ### Excursion: All the Details

if __name__ == "__main__":
    print('\n### Excursion: All the Details')




# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ## _Section 3_

if __name__ == "__main__":
    print('\n## _Section 3_')




import random

def int_fuzzer():
    """A simple function that returns a random integer"""
    return random.randrange(1, 100) + 0.5

if __name__ == "__main__":
    pass


# ## _Section 4_

if __name__ == "__main__":
    print('\n## _Section 4_')




# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




if __name__ == "__main__":
    print(int_fuzzer())


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
    pass


if __name__ == "__main__":
    2 + 2


# ### Exercise 2: _Title_

if __name__ == "__main__":
    print('\n### Exercise 2: _Title_')




# ## Reducing Trees

if __name__ == "__main__":
    print('\n## Reducing Trees')




if __package__ is None or __package__ == "":
    from DeltaDebugger import DeltaDebugger
else:
    from .DeltaDebugger import DeltaDebugger


import copy

if __name__ == "__main__":
    from IPython.display import display


class TreeHDDReducer(TreeGeneralizer):
    def _reduce(self, path, tree):
        """This is HDD"""

        node, children = self.get_subtree(path, tree)
            
        if len(path) >= 1:
            parent, parent_children = self.get_subtree(path[:-1], tree)
 
            assert parent_children[path[-1]] == (node, children)

            def test_children(children):
                parent_children[path[-1]] = (node, children)
                s = tree_to_string(tree)
                self.test(s)

            with DeltaDebugger() as dd:
                test_children(children)
            
            # display(display_tree(tree))

            children = dd.min_args()['children']
            parent_children[path[-1]] = (node, children)
        
        for i, child in enumerate(children):
            self._reduce(path + [i], tree)
            
        return tree

    def reduce(self):
        return self._reduce([], self.tree)

def bad_input_tree_hdd_reducer():
    return TreeHDDReducer(SIMPLE_HTML_GRAMMAR, copy.deepcopy(bad_input_tree),
                       remove_html_markup, log=True)    

if __name__ == "__main__":
    all_terminals(bad_input_tree_hdd_reducer().reduce())


# ## More Reducing Trees

if __name__ == "__main__":
    print('\n## More Reducing Trees')




class TreeReducer(TreeGeneralizer):
    def new_min_tree(self, start_symbol):
        if self.log >= 2:
            print(f"Creating new minimal tree for {start_symbol}")

        fuzzer = GrammarFuzzer(self.grammar, start_symbol=start_symbol,
                               min_nonterminals=0,
                               max_nonterminals=0)
        fuzzer.fuzz()
        return fuzzer.derivation_tree

def bad_input_tree_reducer():
    return TreeReducer(SIMPLE_HTML_GRAMMAR, bad_input_tree,
                       remove_html_markup, log=2)    

if __name__ == "__main__":
    tree_to_string(bad_input_tree_reducer().new_min_tree('<start>'))


class TreeReducer(TreeReducer):
    def reduce_path(self, path, tree=None):
        if tree is None:
            tree = self.tree

        node, children = tree

        if not path:
            return self.new_min_tree(node)

        head = path[0]
        new_children = (children[:head] +
                        [self.reduce_path(path[1:], children[head])] +
                        children[head + 1:])
        return node, new_children

if __name__ == "__main__":
    tree_to_string(bad_input_tree_reducer().reduce_path([0, 0, 1, 0]))


class TreeReducer(TreeReducer):
    def can_reduce(self, path, tree=None):
        reduced_tree = self.reduce_path(path, tree)
        if self.test_tree(reduced_tree):
            # Failure no longer occurs; cannot reduce
            return False

        return True

class TreeReducer(TreeReducer):
    def reducible_paths(self):
        return self.find_paths(self.can_reduce)

if __name__ == "__main__":
    bad_input_tree_reducer().reducible_paths()


class TreeReducer(TreeReducer):
    def reduce(self):
        tree = self.tree
        for path in self.reducible_paths():
            tree = self.reduce_path(path, tree)
            
        return tree

if __name__ == "__main__":
    all_terminals(bad_input_tree_reducer().reduce())

