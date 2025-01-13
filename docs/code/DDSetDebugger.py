#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# "Generalizing Failure Circumstances" - a chapter of "The Debugging Book"
# Web site: https://www.debuggingbook.org/html/DDSetDebugger.html
# Last change: 2025-01-13 15:55:24+01:00
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
The Debugging Book - Generalizing Failure Circumstances

This file can be _executed_ as a script, running all experiments:

    $ python DDSetDebugger.py

or _imported_ as a package, providing classes, functions, and constants:

    >>> from debuggingbook.DDSetDebugger import <identifier>

but before you do so, _read_ it and _interact_ with it at:

    https://www.debuggingbook.org/html/DDSetDebugger.html

This chapter provides a class `DDSetDebugger`, implementing the DDSET algorithm for generalizing failure-inducing inputs. The `DDSetDebugger` is used as follows:

with DDSetDebugger(grammar) as dd:
    function(args...)
dd


Here, `function(args...)` is a failing function call (= raises an execption) that takes at least one string argument; `grammar` is an [input grammar in fuzzingbook format](https://www.fuzzingbook.org/html/Grammars.html) that matches the format of this argument.

The result is a call of `function()` with an _abstract failure-inducing input_ – a variant of the conrete input in which parts are replaced by placeholders in the form ``, where `` is a nonterminal in the grammar. The failure has been verified to occur for a number of instantiations of ``.

Here is an example of how `DDSetDebugger` works. The concrete failing input `"bar` is generalized to an _abstract failure-inducing input_:

>>> with DDSetDebugger(SIMPLE_HTML_GRAMMAR) as dd:
>>>     remove_html_markup('"bar')
>>> dd
remove_html_markup(s='"')

The abstract input tells us that the failure occurs for whatever opening and closing HTML tags as long as there is a double quote between them.

A programmatic interface is available as well. `generalize()` returns a mapping of argument names to (generalized) values:

>>> dd.generalize()
{'s': '"'}

Using `fuzz()`, the abstract input can be instantiated to further concrete inputs, all set to produce the failure again:

>>> for i in range(10):
>>>     print(dd.fuzz())
remove_html_markup(s='"1')
remove_html_markup(s='"c*C')
remove_html_markup(s='"')
remove_html_markup(s='")')
remove_html_markup(s='"')
remove_html_markup(s='"')
remove_html_markup(s='"\t7')
remove_html_markup(s='"')
remove_html_markup(s='"2')
remove_html_markup(s='"\r~\t\r')


`DDSetDebugger` can be customized by passing a subclass of `TreeGeneralizer`, which does the gist of the work; for details, see its constructor.
The full class hierarchy is shown below.

For more details, source, and documentation, see
"The Debugging Book - Generalizing Failure Circumstances"
at https://www.debuggingbook.org/html/DDSetDebugger.html
'''


# Allow to use 'from . import <module>' when run as script (cf. PEP 366)
if __name__ == '__main__' and __package__ is None:
    __package__ = 'debuggingbook'


# Generalizing Failure Circumstances
# ==================================

if __name__ == '__main__':
    print('# Generalizing Failure Circumstances')



if __name__ == '__main__':
    from .bookutils import YouTubeVideo
    YouTubeVideo("PV22XtIQU1s")

if __name__ == '__main__':
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)

from . import DeltaDebugger

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



## A Failing Program
## -----------------

if __name__ == '__main__':
    print('\n## A Failing Program')



def remove_html_markup(s):  # type: ignore
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

if __name__ == '__main__':
    remove_html_markup("Be <em>quiet</em>, he said")

BAD_INPUT = '<foo>"bar</foo>'

from .ExpectError import ExpectError

if __name__ == '__main__':
    with ExpectError(AssertionError):
        remove_html_markup(BAD_INPUT)

from .bookutils import quiz

if __name__ == '__main__':
    quiz("If `s = '<foo>\"bar</foo>'` (i.e., `BAD_INPUT`), "
         "what is the value of `out` such that the assertion fails?",
        [
            '`bar`',
            '`bar</foo>`',
            '`"bar</foo>`',
            '`<foo>"bar</foo>`',
        ], '9999999 // 4999999')

## Grammars
## --------

if __name__ == '__main__':
    print('\n## Grammars')



import fuzzingbook

from typing import Any, Callable, Optional, Type, Tuple
from typing import Dict, Union, List, cast, Generator

Grammar = Dict[str,  # A grammar maps strings...
               List[
                   Union[str,  # to list of strings...
                         Tuple[str, Dict[str, Any]]  # or to pairs of strings and attributes.
                        ]
               ]
              ]

DIGIT_GRAMMAR: Grammar = {
    "<start>":
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
}

EXPR_GRAMMAR: Grammar = {
    "<start>":
        ["<expr>"],

    "<expr>":
        ["<term> + <expr>", "<term> - <expr>", "<term>"],

    "<term>":
        ["<factor> * <term>", "<factor> / <term>", "<factor>"],

    "<factor>":
        ["+<factor>",
         "-<factor>",
         "(<expr>)",
         "<integer>.<integer>",
         "<integer>"],

    "<integer>":
        ["<digit><integer>", "<digit>"],

    "<digit>":
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
}

from fuzzingbook.GrammarFuzzer import GrammarFuzzer

if __name__ == '__main__':
    simple_expr_fuzzer = GrammarFuzzer(EXPR_GRAMMAR)

if __name__ == '__main__':
    for i in range(10):
        fuzz_expr = simple_expr_fuzzer.fuzz()
        print(fuzz_expr)

SIMPLE_HTML_GRAMMAR: Grammar = {
    "<start>":
        ["<html>"],

    "<html>":
        ["<plain-text>", "<tagged-text>"],
}

import string

SIMPLE_HTML_GRAMMAR.update({
    "<plain-text>":
        ["", "<plain-char><plain-text>"],

    "<plain-char>":
        ["<letter>", "<digit>", "<other>", "<whitespace>"],

    "<letter>": list(string.ascii_letters),
    "<digit>": list(string.digits),
    "<other>": list(string.punctuation.replace('<', '').replace('>', '')),
    "<whitespace>": list(string.whitespace)
})

SIMPLE_HTML_GRAMMAR.update({
    "<tagged-text>":
        ["<opening-tag><html><closing-tag>",
         "<self-closing-tag>",
         "<opening-tag>"],
})

SIMPLE_HTML_GRAMMAR.update({
    "<opening-tag>":
        ["<lt><id><gt>",
         "<lt><id><attrs><gt>"],

    "<lt>": ["<"],
    "<gt>": [">"],

    "<id>":
        ["<letter>", "<id><letter>", "<id><digit>"],

    "<closing-tag>":
        ["<lt>/<id><gt>"],

    "<self-closing-tag>":
        ["<lt><id><attrs>/<gt>"],
})

SIMPLE_HTML_GRAMMAR.update({
    "<attrs>":
        ["<attr>", "<attr><attrs>" ],

    "<attr>":
        [" <id>='<plain-text>'",
         ' <id>="<plain-text>"'],
})

if __name__ == '__main__':
    simple_html_fuzzer = GrammarFuzzer(SIMPLE_HTML_GRAMMAR)

if __name__ == '__main__':
    for i in range(10):
        fuzz_html = simple_html_fuzzer.fuzz()
        print(repr(fuzz_html))

## Derivation Trees
## ----------------

if __name__ == '__main__':
    print('\n## Derivation Trees')



DerivationTree = Tuple[str, Optional[List[Any]]]

if __name__ == '__main__':
    fuzz_html

from graphviz import Digraph

def display_tree(tree: DerivationTree) -> Digraph:
    def graph_attr(dot: Digraph) -> None:
        dot.attr('node', shape='box', color='white', margin='0.0,0.0')
        dot.attr('node',
                 fontname="'Fira Mono', 'Source Code Pro', 'Courier', monospace")

    def node_attr(dot: Digraph, nid: str, symbol: str, ann: str) -> None:
        fuzzingbook.GrammarFuzzer.default_node_attr(dot, nid, symbol, ann)
        if symbol.startswith('<'):
            dot.node(repr(nid), fontcolor='#0060a0')
        else:
            dot.node(repr(nid), fontcolor='#00a060')
        dot.node(repr(nid), scale='2')

    return fuzzingbook.GrammarFuzzer.display_tree(tree,
        node_attr=node_attr,
        graph_attr=graph_attr)

if __name__ == '__main__':
    display_tree(simple_html_fuzzer.derivation_tree)

import pprint

if __name__ == '__main__':
    pp = pprint.PrettyPrinter(depth=7)
    pp.pprint(simple_html_fuzzer.derivation_tree)

## Parsing
## -------

if __name__ == '__main__':
    print('\n## Parsing')



from fuzzingbook.Parser import Parser, EarleyParser  # minor dependency

if __name__ == '__main__':
    simple_html_parser = EarleyParser(SIMPLE_HTML_GRAMMAR)

if __name__ == '__main__':
    bad_input_tree = list(simple_html_parser.parse(BAD_INPUT))[0]

if __name__ == '__main__':
    display_tree(bad_input_tree)

from fuzzingbook.GrammarFuzzer import tree_to_string, all_terminals

if __name__ == '__main__':
    tree_to_string(bad_input_tree)

if __name__ == '__main__':
    assert tree_to_string(bad_input_tree) == BAD_INPUT

## Mutating the Tree
## -----------------

if __name__ == '__main__':
    print('\n## Mutating the Tree')



from fuzzingbook.Grammars import is_valid_grammar

class TreeMutator:
    """Grammar-based mutations of derivation trees."""

    def __init__(self, grammar: Grammar, tree: DerivationTree,
                 fuzzer: Optional[GrammarFuzzer] = None, log: Union[bool, int] = False):
        """
        Constructor. 
        `grammar` is the underlying grammar; 
        `tree` is the tree to work on.
        `fuzzer` is the grammar fuzzer to use (default: `GrammarFuzzer`)
        """

        assert is_valid_grammar(grammar)
        self.grammar = grammar
        self.tree = tree
        self.log = log

        if fuzzer is None:
            fuzzer = GrammarFuzzer(grammar)

        self.fuzzer = fuzzer

### Referencing Subtrees

if __name__ == '__main__':
    print('\n### Referencing Subtrees')



TreePath = List[int]

class TreeMutator(TreeMutator):
    def get_subtree(self, path: TreePath, tree: Optional[DerivationTree] = None) -> DerivationTree:
        """Access a subtree based on `path` (a list of children numbers)"""
        if tree is None:
            tree = self.tree

        symbol, children = tree

        if not path or children is None:
            return tree

        return self.get_subtree(path[1:], children[path[0]])

def bad_input_tree_mutator() -> TreeMutator:
    return TreeMutator(SIMPLE_HTML_GRAMMAR, bad_input_tree, log=2)    

if __name__ == '__main__':
    plain_text_subtree = bad_input_tree_mutator().get_subtree([0, 0, 1, 0])
    pp.pprint(plain_text_subtree)

if __name__ == '__main__':
    tree_to_string(plain_text_subtree)

def primes_generator() -> Generator[int, None, None]:
    # Adapted from https://www.python.org/ftp/python/doc/nluug-paper.ps
    primes = [2]
    yield 2
    i = 3
    while True:
        for p in primes:
            if i % p == 0 or p * p > i:
                break

        if i % p != 0:
            primes.append(i)
            yield i

        i += 2

if __name__ == '__main__':
    prime_numbers = primes_generator()

if __name__ == '__main__':
    quiz("In `bad_input_tree`, what is "
         " the subtree at the path `[0, 0, 2, 1]` as string?", 
        [
            f"`{tree_to_string(bad_input_tree_mutator().get_subtree([0, 0, 2, 0]))}`",
            f"`{tree_to_string(bad_input_tree_mutator().get_subtree([0, 0, 2, 1]))}`",
            f"`{tree_to_string(bad_input_tree_mutator().get_subtree([0, 0, 2]))}`",
            f"`{tree_to_string(bad_input_tree_mutator().get_subtree([0, 0, 0]))}`",
        ], 'next(prime_numbers)', globals()
        )

### Creating new Subtrees

if __name__ == '__main__':
    print('\n### Creating new Subtrees')



class TreeMutator(TreeMutator):
    def new_tree(self, start_symbol: str) -> DerivationTree:
        """Create a new subtree for <start_symbol>."""

        if self.log >= 2:
            print(f"Creating new tree for {start_symbol}")

        tree = (start_symbol, None)
        return self.fuzzer.expand_tree(tree)

if __name__ == '__main__':
    plain_text_tree = cast(TreeMutator, bad_input_tree_mutator()).new_tree('<plain-text>')
    display_tree(plain_text_tree)

if __name__ == '__main__':
    tree_to_string(plain_text_tree)

### Mutating the Tree

if __name__ == '__main__':
    print('\n### Mutating the Tree')



class TreeMutator(TreeMutator):
    def mutate(self, path: TreePath, tree: Optional[DerivationTree] = None) -> DerivationTree:
        """Return a new tree mutated at `path`"""
        if tree is None:
            tree = self.tree
        assert tree is not None

        symbol, children = tree

        if not path or children is None:
            return self.new_tree(symbol)

        head = path[0]
        new_children = (children[:head] +
                        [self.mutate(path[1:], children[head])] +
                        children[head + 1:])
        return symbol, new_children

if __name__ == '__main__':
    mutated_tree = cast(TreeMutator, bad_input_tree_mutator()).mutate([0, 0, 1, 0])
    display_tree(mutated_tree)

if __name__ == '__main__':
    tree_to_string(mutated_tree)

## Generalizing Trees
## ------------------

if __name__ == '__main__':
    print('\n## Generalizing Trees')



class TreeGeneralizer(TreeMutator):
    """Determine which parts of a derivation tree can be generalized."""

    def __init__(self, grammar: Grammar, tree: DerivationTree, test: Callable,
                 max_tries_for_generalization: int = 10, **kwargs: Any) -> None:
        """
        Constructor. `grammar` and `tree` are as in `TreeMutator`.
        `test` is a function taking a string that either
          * raises an exception, indicating test failure;
          * or not, indicating test success.
        `max_tries_for_generalization` is the number of times
        an instantiation has to fail before it is generalized.
        """

        super().__init__(grammar, tree, **kwargs)
        self.test = test
        self.max_tries_for_generalization = max_tries_for_generalization

class TreeGeneralizer(TreeGeneralizer):
    def test_tree(self, tree: DerivationTree) -> bool:
        """Return True if testing `tree` passes, else False"""
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

### Testing for Generalization

if __name__ == '__main__':
    print('\n### Testing for Generalization')



class TreeGeneralizer(TreeGeneralizer):
    def can_generalize(self, path: TreePath, tree: Optional[DerivationTree] = None) -> bool:
        """Return True if the subtree at `path` can be generalized."""
        for i in range(self.max_tries_for_generalization):
            mutated_tree = self.mutate(path, tree)
            if self.test_tree(mutated_tree):
                # Failure no longer occurs; cannot abstract
                return False

        return True

def bad_input_tree_generalizer(**kwargs: Any) -> TreeGeneralizer:
    return TreeGeneralizer(SIMPLE_HTML_GRAMMAR, bad_input_tree,
                           remove_html_markup, **kwargs)

if __name__ == '__main__':
    bad_input_tree_generalizer(log=True).can_generalize([0])

if __name__ == '__main__':
    bad_input_tree_generalizer(log=True).can_generalize([0, 0, 1, 0])

if __name__ == '__main__':
    bad_input_tree_generalizer(log=True).can_generalize([0, 0, 2])

if __name__ == '__main__':
    quiz("Is this also true for `<opening-tag>`?",
         [
             "Yes",
             "No"
         ], '("Yes" == "Yes") + ("No" == "No")')

BAD_ATTR_INPUT = '<foo attr="\'">bar</foo>'

if __name__ == '__main__':
    remove_html_markup(BAD_ATTR_INPUT)

if __name__ == '__main__':
    bad_input_tree_generalizer().can_generalize([0, 0, 0])

if __name__ == '__main__':
    bad_input_tree_generalizer(max_tries_for_generalization=100, log=True).can_generalize([0, 0, 0])

### Generalizable Paths

if __name__ == '__main__':
    print('\n### Generalizable Paths')



class TreeGeneralizer(TreeGeneralizer):
    def find_paths(self, 
                   predicate: Callable[[TreePath, DerivationTree], bool], 
                   path: Optional[TreePath] = None, 
                   tree: Optional[DerivationTree] = None) -> List[TreePath]:
        """
        Return a list of all paths for which `predicate` holds.
        `predicate` is a function `predicate`(`path`, `tree`), where
        `path` denotes a subtree in `tree`. If `predicate()` returns
        True, `path` is included in the returned list.
        """

        if path is None:
            path = []
        assert path is not None

        if tree is None:
            tree = self.tree
        assert tree is not None

        symbol, children = self.get_subtree(path)

        if predicate(path, tree):
            return [path]

        paths = []
        if children is not None:
            for i, child in enumerate(children):
                child_symbol, _ = child
                if child_symbol in self.grammar:
                    paths += self.find_paths(predicate, path + [i])

        return paths

    def generalizable_paths(self) -> List[TreePath]:
        """Return a list of all paths whose subtrees can be generalized."""
        return self.find_paths(self.can_generalize)

if __name__ == '__main__':
    bad_input_generalizable_paths = \
        cast(TreeGeneralizer, bad_input_tree_generalizer()).generalizable_paths()
    bad_input_generalizable_paths

class TreeGeneralizer(TreeGeneralizer):
    def generalize_path(self, path: TreePath, 
                        tree: Optional[DerivationTree] = None) -> DerivationTree:
        """Return a copy of the tree in which the subtree at `path`
        is generalized (= replaced by a nonterminal without children)"""

        if tree is None:
            tree = self.tree
        assert tree is not None

        symbol, children = tree

        if not path or children is None:
            return symbol, None  # Nonterminal without children

        head = path[0]
        new_children = (children[:head] +
                        [self.generalize_path(path[1:], children[head])] +
                        children[head + 1:])
        return symbol, new_children

if __name__ == '__main__':
    all_terminals(cast(TreeGeneralizer, bad_input_tree_generalizer()).generalize_path([0, 0, 0]))

class TreeGeneralizer(TreeGeneralizer):
    def generalize(self) -> DerivationTree:
        """Returns a copy of the tree in which all generalizable subtrees
        are generalized (= replaced by nonterminals without children)"""
        tree = self.tree
        assert tree is not None

        for path in self.generalizable_paths():
            tree = self.generalize_path(path, tree)

        return tree

if __name__ == '__main__':
    abstract_failure_inducing_input = cast(TreeGeneralizer, bad_input_tree_generalizer()).generalize()

if __name__ == '__main__':
    all_terminals(abstract_failure_inducing_input)

## Fuzzing with Patterns
## ---------------------

if __name__ == '__main__':
    print('\n## Fuzzing with Patterns')



import copy

class TreeGeneralizer(TreeGeneralizer):
    def fuzz_tree(self, tree: DerivationTree) -> DerivationTree:
        """Return an instantiated copy of `tree`."""
        tree = copy.deepcopy(tree)
        return self.fuzzer.expand_tree(tree)

if __name__ == '__main__':
    bitg = cast(TreeGeneralizer, bad_input_tree_generalizer())
    for i in range(10):
        print(all_terminals(bitg.fuzz_tree(abstract_failure_inducing_input)))

if __name__ == '__main__':
    successes = 0
    failures = 0
    trials = 1000

    for i in range(trials):
        test_input = all_terminals(
            bitg.fuzz_tree(abstract_failure_inducing_input))
        try:
            remove_html_markup(test_input)
        except AssertionError:
            successes += 1
        else:
            failures += 1

if __name__ == '__main__':
    successes, failures

if __name__ == '__main__':
    failures / 1000

## Putting it all Together
## -----------------------

if __name__ == '__main__':
    print('\n## Putting it all Together')



### Constructor

if __name__ == '__main__':
    print('\n### Constructor')



from .DeltaDebugger import CallCollector

class DDSetDebugger(CallCollector):
    """
    Debugger implementing the DDSET algorithm for abstracting failure-inducing inputs.
    """

    def __init__(self, grammar: Grammar, 
                 generalizer_class: Type = TreeGeneralizer,
                 parser: Optional[Parser] = None,
                 **kwargs: Any) -> None:
        """Constructor.
        `grammar` is an input grammar in fuzzingbook format.
        `generalizer_class` is the tree generalizer class to use
        (default: `TreeGeneralizer`)
        `parser` is the parser to use (default: `EarleyParser(grammar)`).
        All other keyword args are passed to the tree generalizer, notably:
        `fuzzer` - the fuzzer to use (default: `GrammarFuzzer`), and
        `log` - enables debugging output if True.
        """
        super().__init__()
        self.grammar = grammar
        assert is_valid_grammar(grammar)

        self.generalizer_class = generalizer_class

        if parser is None:
            parser = EarleyParser(grammar)
        self.parser = parser
        self.kwargs = kwargs

        # These save state for further fuzz() calls
        self.generalized_args: Dict[str, Any] = {}
        self.generalized_trees: Dict[str, DerivationTree] = {}
        self.generalizers: Dict[str, TreeGeneralizer] = {}

### Generalizing Arguments

if __name__ == '__main__':
    print('\n### Generalizing Arguments')



class DDSetDebugger(DDSetDebugger):
    def generalize(self) -> Dict[str, Any]:
        """
        Generalize arguments seen. For each function argument,
        produce an abstract failure-inducing input that characterizes
        the set of inputs for which the function fails.
        """
        if self.generalized_args:
            return self.generalized_args

        self.generalized_args = copy.deepcopy(self.args())
        self.generalized_trees = {}
        self.generalizers = {}

        for arg in self.args():
            def test(value: Any) -> Any:
                return self.call({arg: value})

            value = self.args()[arg]
            if isinstance(value, str):
                tree = list(self.parser.parse(value))[0]
                gen = self.generalizer_class(self.grammar, tree, test, 
                                             **self.kwargs)
                generalized_tree = gen.generalize()

                self.generalizers[arg] = gen
                self.generalized_trees[arg] = generalized_tree
                self.generalized_args[arg] = all_terminals(generalized_tree)

        return self.generalized_args

class DDSetDebugger(DDSetDebugger):
    def __repr__(self) -> str:
        """Return a string representation of the generalized call."""
        return self.format_call(self.generalize())

if __name__ == '__main__':
    with DDSetDebugger(SIMPLE_HTML_GRAMMAR) as dd:
        remove_html_markup(BAD_INPUT)
    dd

### Fuzzing

if __name__ == '__main__':
    print('\n### Fuzzing')



class DDSetDebugger(DDSetDebugger):
    def fuzz_args(self) -> Dict[str, Any]:
        """
        Return arguments randomly instantiated
        from the abstract failure-inducing pattern.
        """
        if not self.generalized_trees:
            self.generalize()

        args = copy.deepcopy(self.generalized_args)
        for arg in args:
            if arg not in self.generalized_trees:
                continue

            tree = self.generalized_trees[arg]
            gen = self.generalizers[arg]
            instantiated_tree = gen.fuzz_tree(tree)
            args[arg] = all_terminals(instantiated_tree)

        return args

    def fuzz(self) -> str:
        """
        Return a call with arguments randomly instantiated
        from the abstract failure-inducing pattern.
        """
        return self.format_call(self.fuzz_args())

if __name__ == '__main__':
    with DDSetDebugger(SIMPLE_HTML_GRAMMAR) as dd:
        remove_html_markup(BAD_INPUT)

if __name__ == '__main__':
    dd.fuzz()

if __name__ == '__main__':
    dd.fuzz()

if __name__ == '__main__':
    dd.fuzz()

if __name__ == '__main__':
    with ExpectError(AssertionError):
        eval(dd.fuzz())

## More Examples
## -------------

if __name__ == '__main__':
    print('\n## More Examples')



### Square Root

if __name__ == '__main__':
    print('\n### Square Root')



from .Assertions import square_root  # minor dependency

if __name__ == '__main__':
    with ExpectError(AssertionError):
        square_root(-1)

INT_GRAMMAR: Grammar = {
    "<start>":
        ["<int>"],

    "<int>":
        ["<positive-int>", "-<positive-int>"],

    "<positive-int>":
        ["<digit>", "<nonzero-digit><positive-int>"],

    "<nonzero-digit>": list("123456789"),
    
    "<digit>": list(string.digits),
}

def square_root_test(s: str) -> None:
    return square_root(int(s))

if __name__ == '__main__':
    with DDSetDebugger(INT_GRAMMAR, log=True) as dd_square_root:
        square_root_test("-1")

if __name__ == '__main__':
    dd_square_root

### Middle

if __name__ == '__main__':
    print('\n### Middle')



from .StatisticalDebugger import middle  # minor dependency

def middle_test(s: str) -> None:
    x, y, z = eval(s)
    assert middle(x, y, z) == sorted([x, y, z])[1]

XYZ_GRAMMAR: Grammar = {
    "<start>":
        ["<int>, <int>, <int>"],

    "<int>":
        ["<positive-int>", "-<positive-int>"],

    "<positive-int>":
        ["<digit>", "<nonzero-digit><positive-int>"],

    "<nonzero-digit>": list("123456789"),
    
    "<digit>": list(string.digits),
}

if __name__ == '__main__':
    with ExpectError(AssertionError):
        middle_test("2, 1, 3")

if __name__ == '__main__':
    with DDSetDebugger(XYZ_GRAMMAR, log=True) as dd_middle:
        middle_test("2, 1, 3")

if __name__ == '__main__':
    dd_middle

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



if __name__ == '__main__':
    with DDSetDebugger(SIMPLE_HTML_GRAMMAR) as dd:
        remove_html_markup('<foo>"bar</foo>')
    dd

if __name__ == '__main__':
    dd.generalize()

if __name__ == '__main__':
    for i in range(10):
        print(dd.fuzz())

from .ClassDiagram import display_class_hierarchy

if __name__ == '__main__':
    display_class_hierarchy([DDSetDebugger, TreeGeneralizer],
                            public_methods=[
                                CallCollector.__init__,
                                CallCollector.__enter__,
                                CallCollector.__exit__,
                                CallCollector.function,
                                CallCollector.args,
                                CallCollector.exception,
                                CallCollector.call,  # type: ignore
                                DDSetDebugger.__init__,
                                DDSetDebugger.__repr__,
                                DDSetDebugger.fuzz,
                                DDSetDebugger.fuzz_args,
                                DDSetDebugger.generalize,
                            ], project='debuggingbook')

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



### Exercise 1: Generalization and Specialization

if __name__ == '__main__':
    print('\n### Exercise 1: Generalization and Specialization')



if __name__ == '__main__':
    all_terminals(abstract_failure_inducing_input)

if __name__ == '__main__':
    more_precise_bitg = \
        cast(TreeGeneralizer, bad_input_tree_generalizer(max_tries_for_generalization=100))

    more_precise_abstract_failure_inducing_input = \
        more_precise_bitg.generalize()

if __name__ == '__main__':
    all_terminals(more_precise_abstract_failure_inducing_input)

if __name__ == '__main__':
    successes = 0
    failures = 0
    trials = 1000

    for i in range(trials):
        test_input = all_terminals(
            more_precise_bitg.fuzz_tree(
                more_precise_abstract_failure_inducing_input))
        try:
            remove_html_markup(test_input)
        except AssertionError:
            successes += 1
        else:
            failures += 1

if __name__ == '__main__':
    successes, failures

if __name__ == '__main__':
    failures / 1000
