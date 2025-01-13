#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# "Learning from Failures" - a chapter of "The Debugging Book"
# Web site: https://www.debuggingbook.org/html/Alhazen.html
# Last change: 2025-01-13 15:57:33+01:00
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
The Debugging Book - Learning from Failures

This file can be _executed_ as a script, running all experiments:

    $ python Alhazen.py

or _imported_ as a package, providing classes, functions, and constants:

    >>> from debuggingbook.Alhazen import <identifier>

but before you do so, _read_ it and _interact_ with it at:

    https://www.debuggingbook.org/html/Alhazen.html

This chapter provides an implementation of the _Alhazen_ approach \cite{Kampmann2020}, which trains machine learning _classifiers_ from input features.
Given a test function, a grammar, and a set of inputs, the `Alhazen` class produces a decision tree that _characterizes failure circumstances_:

>>> alhazen = Alhazen(sample_runner, CALC_GRAMMAR, initial_sample_list,
>>>                   max_iterations=20)
>>> alhazen.run()

The final decision tree can be accessed using `last_tree()`:

>>> # alhazen.last_tree()

We can visualize the resulting decision tree using `Alhazen.show_decision_tree()`:

>>> alhazen.show_decision_tree()
A decision tree is read from top to bottom.
Decision nodes (with two children) come with a _predicate_ on top.
This predicate is either

* _numeric_, such as ` > 20`, indicating the numeric value of the given symbol, or
* _existential_, such as ` == '1'`, which has a _negative_ value when False, and a _positive_ value when True.

If the predicate evaluates to `True`, follow the left path; if it evaluates to `False`, follow the right path.
A leaf node (no children) will give you the final decision `class = BUG` or `class = NO_BUG`.

So if the predicate states ` == 'sqrt' <= 0.5`, this means that

* If the function is _not_ `sqrt` (the predicate ` == 'sqrt'` is negative, see above, and hence less than 0.5), follow the left (`True`) path.
* If the function _is_ `sqrt` (the predicate ` == 'sqrt'` is positive), follow the right (`False`) path.

The `samples` field shows the number of sample inputs that contributed to this decision.
The `gini` field (aka Gini impurity) indicates how many samples fall into the displayed class (`BUG` or `NO_BUG`).
A `gini` value of `0.0` means _purity_ - all samples fall into the displayed class.
The _saturation_ of nodes also indicates purity – the higher the saturation, the higher the purity.

There is also a text version available, with much fewer (but hopefully still essential) details:

>>> print(alhazen.friendly_decision_tree())
if  <= 4.5000:
  if  == 'sqrt':
    if  <= 42.1600:
      if  == '-':
        BUG
      else:
        NO_BUG
    else:
      NO_BUG
  else:
    NO_BUG
else:
  NO_BUG



In both representations, we see that the present failure is associated with a negative value for the `sqrt` function and precise boundaries for its value.
In fact, the error conditions are given in the source code:

>>> import inspect
>>> print(inspect.getsource(task_sqrt))
def task_sqrt(x):
    """Computes the square root of x, using the Newton-Raphson method"""
    if x <= -12 and x >= -42:
        x = 0  # Guess where the bug is :-)
    else:
        x = 1
    x = max(x, 0)
    approx = None
    guess = x / 2
    while approx != guess:
        approx = guess
        guess = (approx + x / approx) / 2
    return approx



Try out Alhazen on your own code and your own examples!


For more details, source, and documentation, see
"The Debugging Book - Learning from Failures"
at https://www.debuggingbook.org/html/Alhazen.html
'''


# Allow to use 'from . import <module>' when run as script (cf. PEP 366)
if __name__ == '__main__' and __package__ is None:
    __package__ = 'debuggingbook'


# Learning from Failures
# ======================

if __name__ == '__main__':
    print('# Learning from Failures')



if __name__ == '__main__':
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



## Machine Learning for Automated Debugging
## ----------------------------------------

if __name__ == '__main__':
    print('\n## Machine Learning for Automated Debugging')



### The Alhazen Approach

if __name__ == '__main__':
    print('\n### The Alhazen Approach')



### Structure of this Chapter

if __name__ == '__main__':
    print('\n### Structure of this Chapter')



## Inputs and Grammars
## -------------------

if __name__ == '__main__':
    print('\n## Inputs and Grammars')



from typing import List, Tuple, Dict, Any, Optional

from fuzzingbook.Grammars import Grammar, EXPR_GRAMMAR, reachable_nonterminals, is_valid_grammar
from fuzzingbook.GrammarFuzzer import GrammarFuzzer, expansion_to_children, DerivationTree, tree_to_string, display_tree, is_nonterminal
from fuzzingbook.Parser import EarleyParser

from math import tan as rtan
from math import cos as rcos
from math import sin as rsin

if __name__ == '__main__':
    """
This file contains the code under test for the example bug.
The sqrt() method fails on x <= 0.
"""
    def task_sqrt(x):
        """Computes the square root of x, using the Newton-Raphson method"""
        if x <= -12 and x >= -42:
            x = 0  # Guess where the bug is :-)
        else:
            x = 1
        x = max(x, 0)
        approx = None
        guess = x / 2
        while approx != guess:
            approx = guess
            guess = (approx + x / approx) / 2
        return approx


    def task_tan(x):
        return rtan(x)


    def task_cos(x):
        return rcos(x)


    def task_sin(x):
        return rsin(x)

CALC_GRAMMAR: Grammar = {
    "<start>":
        ["<function>(<term>)"],

    "<function>":
        ["sqrt", "tan", "cos", "sin"],

    "<term>": ["-<value>", "<value>"],

    "<value>":
        ["<integer>.<digits>",
         "<integer>"],

    "<integer>":
        ["<lead-digit><digits>", "<digit>"],

    "<digits>":
        ["<digit><digits>", "<digit>"],

    "<lead-digit>":  # First digit cannot be zero
        ["1", "2", "3", "4", "5", "6", "7", "8", "9"],

    "<digit>":
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
}

if __name__ == '__main__':
    initial_sample_list = ['sqrt(-16)', 'sqrt(4)']

from enum import Enum

class OracleResult(Enum):
    BUG = "BUG"
    NO_BUG = "NO_BUG"
    UNDEF = "UNDEF"

    def __str__(self):
        return self.value

import sys

SUBJECT = "calculator"

def sample_runner(sample):
    testcode = sample

    try:
        # Simply execute the calculator code, with the functions replaced
        exec(testcode, {"sqrt": task_sqrt, "tan": task_tan, "sin": task_sin, "cos": task_cos}, {})
        return OracleResult.NO_BUG
    except ZeroDivisionError:
        return OracleResult.BUG
    except Exception as e:
        print(e, file=sys.stderr)
        return OracleResult.UNDEF

if __name__ == '__main__':
    sample = "sqrt(-16)"
    sample_runner(sample)

if __name__ == '__main__':
    assert sample_runner("sqrt(-23)") == OracleResult.BUG
    assert sample_runner("sqrt(44)") == OracleResult.NO_BUG
    assert sample_runner("cos(-9)") == OracleResult.NO_BUG

if __name__ == '__main__':
    sample_runner("undef_function(QUERY)")

import pandas
import numpy
import matplotlib

def execute_samples(sample_list):
    data = []
    for sample in sample_list:
        result = sample_runner(sample)
        data.append({"oracle": result })

    return pandas.DataFrame.from_records(data)

if __name__ == '__main__':
    sample_list = ["sqrt(-20)", "cos(2)", "sqrt(-100)", "undef_function(foo)"]

if __name__ == '__main__':
    labels = execute_samples(sample_list)
    labels

if __name__ == '__main__':
    for i, row in enumerate(labels['oracle']): print(sample_list[i].ljust(30) + str(row))

if __name__ == '__main__':
    clean_data = labels.drop(labels[labels.oracle.astype(str) == "UNDEF"].index)
    clean_data

if __name__ == '__main__':
    oracle = execute_samples(sample_list)
    for i, row in enumerate(oracle['oracle']):
        print(sample_list[i].ljust(30) + str(row))

if __name__ == '__main__':
    guess_samples = ['cos(-16)', 'tan(-16)', 'sqrt(-100)', 'sqrt(-20.23412431234123)']

if __name__ == '__main__':
    guess_oracle = execute_samples(guess_samples)

if __name__ == '__main__':
    for i, row in enumerate(guess_oracle['oracle']):
        print(guess_samples[i].ljust(30) + str(row))

## Step 1: Extracting Features
## ---------------------------

if __name__ == '__main__':
    print('\n## Step 1: Extracting Features')



### Internal and "Friendly" Feature Names

if __name__ == '__main__':
    print('\n### Internal and "Friendly" Feature Names')



### Implementing Feature Classes

if __name__ == '__main__':
    print('\n### Implementing Feature Classes')



from abc import ABC, abstractmethod

class Feature(ABC):
    '''
    The abstract base class for grammar features.

    Args:
        name : A unique identifier name for this feature. Should not contain Whitespaces.
               e.g., 'type(<feature>@1)'
        rule : The production rule (e.g., '<function>' or '<value>').
        key  : The feature key (e.g., the chosen alternative or rule itself).
    '''

    def __init__(self, name: str, rule: str, key: str, /,
                 friendly_name: str = None) -> None:
        self.name = name
        self.rule = rule
        self.key = key
        self._friendly_name = friendly_name or name
        super().__init__()

    def __repr__(self) -> str:
        '''Returns a printable string representation of the feature.'''
        return self.name_rep()

    @abstractmethod
    def name_rep(self) -> str:
        pass

    def friendly_name(self) -> str:
        return self._friendly_name

    @abstractmethod
    def get_feature_value(self, derivation_tree) -> float:
        '''Returns the feature value for a given derivation tree of an input.'''
        pass

    def replace(self, new_key: str) -> 'Feature':
        '''Returns a new feature with the same name but a different key.'''
        return self.__class__(self.name, self.rule, new_key)

class ExistenceFeature(Feature):
    '''
    This class represents existence features of a grammar. Existence features indicate
    whether a particular production rule was used in the derivation sequence of an input.
    For a given production rule P -> A | B, a production existence feature for P and
    alternative existence features for each alternative (i.e., A and B) are defined.

    name : A unique identifier name for this feature. Should not contain Whitespaces.
           e.g., 'exist(<digit>@1)'
    rule : The production rule.
    key  : The feature key, equal to the rule attribute for production features,
           or equal to the corresponding alternative for alternative features.
    '''
    def __init__(self, name: str, rule: str, key: str,
                 friendly_name: str = None) -> None:
        super().__init__(name, rule, key, friendly_name=friendly_name)

    def name_rep(self) -> str:
        if self.rule == self.key:
            return f"exists({self.rule})"
        else:
            return f"exists({self.rule} == {self.key})"

    def get_feature_value(self, derivation_tree) -> float:
        '''Returns the feature value for a given derivation tree of an input.'''
        raise NotImplementedError

    def get_feature_value(self, derivation_tree: DerivationTree) -> float:
        '''Counts the number of times this feature was matched in the derivation tree.'''
        (node, children) = derivation_tree

        # The local match count (1 if the feature is matched for the current node, 0 if not)
        count = 0

        # First check if the current node can be matched with the rule
        if node == self.rule:

            # Production existance feature
            if self.rule == self.key:
                count = 1

            # Production alternative existance feature
            # We compare the children of the expansion with the actual children
            else:
                expansion_children = list(map(lambda x: x[0], expansion_to_children(self.key)))
                node_children = list(map(lambda x: x[0], children))
                if expansion_children == node_children:
                    count= 1

        # Recursively compute the counts for all children and return the sum for the whole tree
        for child in children:
            count = max(count, self.get_feature_value(child)) 

        return count

if __name__ == '__main__':
    from numpy import nanmax, isnan

class NumericInterpretation(Feature):
    '''
    This class represents numeric interpretation features of a grammar. These features
    are defined for productions that only derive words composed of the characters
    [0-9], '.', and '-'. The returned feature value corresponds to the maximum
    floating-point number interpretation of the derived words of a production.

    name : A unique identifier name for this feature. Should not contain Whitespaces.
           e.g., 'num(<integer>)'
    rule : The production rule.
    '''
    def __init__(self, name: str, rule: str, /, 
                 friendly_name: str = None) -> None:
        super().__init__(name, rule, rule, friendly_name=friendly_name)

    def name_rep(self) -> str:
        return f"num({self.key})"

    def get_feature_value(self, derivation_tree) -> float:
        '''Returns the feature value for a given derivation tree of an input.'''
        raise NotImplementedError

    def get_feature_value(self, derivation_tree: DerivationTree) -> float:
        '''Determines the maximum float of this feature in the derivation tree.'''
        (node, children) = derivation_tree

        value = float('nan')
        if node == self.rule:
            try:
                #print(self.name, float(tree_to_string(derivation_tree)))
                value = float(tree_to_string(derivation_tree))
            except ValueError:
                #print(self.name, float(tree_to_string(derivation_tree)), "err")
                pass

        # Return maximum value encountered in tree, ignoring all NaNs
        tree_values = [value] + [self.get_feature_value(c) for c in children]
        if all(isnan(tree_values)):
            return value
        else:
            return nanmax(tree_values)

### Extracting Feature Sets from Grammars

if __name__ == '__main__':
    print('\n### Extracting Feature Sets from Grammars')



def extract_existence_features(grammar: Grammar) -> List[ExistenceFeature]:
    '''
        Extracts all existence features from the grammar and returns them as a list.
        grammar : The input grammar.
    '''

    features = []

    for rule in grammar:
        # add the rule
        features.append(ExistenceFeature(f"exists({rule})", rule, rule))
        # add all alternatives
        for count, expansion in enumerate(grammar[rule]):
            name = f"exists({rule}@{count})"
            friendly_name = f"{rule} == {repr(expansion)}"
            feature = ExistenceFeature(name, rule, expansion,
                                       friendly_name=friendly_name)
            features.append(feature)

    return features

from collections import defaultdict
import re

RE_NONTERMINAL = re.compile(r'(<[^<> ]*>)')

def extract_numeric_features(grammar: Grammar) -> List[NumericInterpretation]:
    '''
        Extracts all numeric interpretation features from the grammar and returns them as a list.

        grammar : The input grammar.
    '''

    features = []

    # Mapping from non-terminals to derivable terminal chars
    derivable_chars = defaultdict(set)

    for rule in grammar:
        for expansion in grammar[rule]:
            # Remove non-terminal symbols and whitespace from expansion
            terminals = re.sub(RE_NONTERMINAL, '', expansion).replace(' ', '')

            # Add each terminal char to the set of derivable chars
            for c in terminals:
                derivable_chars[rule].add(c)

    # Repeatedly update the mapping until convergence
    while True:
        updated = False
        for rule in grammar:
            for r in reachable_nonterminals(grammar, rule):
                before = len(derivable_chars[rule])
                derivable_chars[rule].update(derivable_chars[r])
                after = len(derivable_chars[rule])

                # Set of derivable chars was updated
                if after > before:
                    updated = True

        if not updated:
            break

    numeric_chars = set(['0','1','2','3','4','5','6','7','8','9','.','-'])

    for key in derivable_chars:
        # Check if derivable chars contain only numeric chars
        if len(derivable_chars[key] - numeric_chars) == 0:
            name = f"num({key})"
            friendly_name = f"{key}"

            features.append(NumericInterpretation(f"num({key})", key,
                                                  friendly_name=friendly_name))

    return features

def extract_all_features(grammar: Grammar) -> List[Feature]:
    return (extract_existence_features(grammar)
            + extract_numeric_features(grammar))

if __name__ == '__main__':
    extract_all_features(CALC_GRAMMAR)

if __name__ == '__main__':
    [f.friendly_name() for f in extract_all_features(CALC_GRAMMAR)]

### Extracting Feature Values from Inputs

if __name__ == '__main__':
    print('\n### Extracting Feature Values from Inputs')



def collect_features(sample_list: List[str],
                     grammar: Grammar) -> pandas.DataFrame:

    data = []

    # parse grammar and extract features
    all_features = extract_all_features(grammar)

    # iterate over all samples
    for sample in sample_list:
        parsed_features = {}
        parsed_features["sample"] = sample
        # initate dictionary
        for feature in all_features:
            parsed_features[feature.name] = 0

        # Obtain the parse tree for each input file
        earley = EarleyParser(grammar)
        for tree in earley.parse(sample):

            for feature in all_features:
                parsed_features[feature.name] = feature.get_feature_value(tree)

        data.append(parsed_features)

    return pandas.DataFrame.from_records(data)

if __name__ == '__main__':
    sample_list = ["sqrt(-900)", "sin(24)", "cos(-3.14)"]
    collect_features(sample_list, CALC_GRAMMAR)

def compute_feature_values(sample: str, grammar: Grammar, features: List[Feature]) -> Dict[str, float]:
    '''
        Extracts all feature values from an input.

        sample   : The input.
        grammar  : The input grammar.
        features : The list of input features extracted from the grammar.

    '''
    earley = EarleyParser(CALC_GRAMMAR)

    features = {}
    for tree in earley.parse(sample):
        for feature in extract_all_features(CALC_GRAMMAR):
            features[feature.name_rep()] = feature.get_feature_value(tree)
    return features

if __name__ == '__main__':
    all_features = extract_all_features(CALC_GRAMMAR)
    for sample in sample_list:
        print(f"Features of {sample}:")
        features = compute_feature_values(sample, CALC_GRAMMAR, all_features)
        for feature, value in features.items():
            print(f"    {feature}: {value}")

### Excursion: Transforming Grammars

if __name__ == '__main__':
    print('\n### Excursion: Transforming Grammars')



import random

if __name__ == '__main__':
    random.seed(24)
    f = GrammarFuzzer(EXPR_GRAMMAR, max_nonterminals=3)
    test_input = f.fuzz()
    assert(test_input == tree_to_string(f.derivation_tree))

    display_tree(f.derivation_tree)

def extend_grammar(derivation_tree, grammar):
    (node, children) = derivation_tree

    if is_nonterminal(node):
        assert(node in grammar)
        word = tree_to_string(derivation_tree)

        # Only add to grammar if not already existent
        if word not in grammar[node]:
            grammar[node].append(word)

    for child in children:
        extend_grammar(child, grammar)

import copy

def transform_grammar(sample: str,
                      grammar: Grammar) -> Grammar:
    # copy of the grammar
    transformed_grammar = copy.deepcopy(grammar)

    # parse sample
    earley = EarleyParser(grammar)
    for derivation_tree in earley.parse(sample):
        extend_grammar(derivation_tree, transformed_grammar)

    return transformed_grammar

if __name__ == '__main__':
    transformed_grammar = transform_grammar("1 + 2", EXPR_GRAMMAR)
    for rule in transformed_grammar:
        print(rule.ljust(10), transformed_grammar[rule])

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



## Step 2: Train Classification Model
## ----------------------------------

if __name__ == '__main__':
    print('\n## Step 2: Train Classification Model')



### Decision Trees

if __name__ == '__main__':
    print('\n### Decision Trees')



import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer

if __name__ == '__main__':
    features = [
        {'function-sqrt': 1, 'function-cos': 0, 'function-sin': 0, 'number': -900}, # sqrt(-900)
        {'function-sqrt': 0, 'function-cos': 1, 'function-sin': 0, 'number': 300}, # cos(300)
        {'function-sqrt': 1, 'function-cos': 0, 'function-sin': 0, 'number': -1}, # sqrt(-1)
        {'function-sqrt': 0, 'function-cos': 1, 'function-sin': 0, 'number': -10}, # cos(-10)
        {'function-sqrt': 0, 'function-cos': 0, 'function-sin': 1, 'number': 36}, # sin(36)
        {'function-sqrt': 0, 'function-cos': 0, 'function-sin': 1, 'number': -58}, # sin(-58)
        {'function-sqrt': 1, 'function-cos': 0, 'function-sin': 0, 'number': 27}, # sqrt(27)
    ]

if __name__ == '__main__':
    oracle = [
        OracleResult.BUG,
        OracleResult.NO_BUG,
        OracleResult.BUG,
        OracleResult.NO_BUG,
        OracleResult.NO_BUG,
        OracleResult.NO_BUG,
        OracleResult.NO_BUG
    ]

    # Transform to numpy array
    vec = DictVectorizer()
    X = vec.fit_transform(features).toarray()

if __name__ == '__main__':
    clf = DecisionTreeClassifier(random_state=10)

    # sci-kit learn requires an array of strings
    oracle_clean = [str(c) for c in oracle]
    clf = clf.fit(X, oracle_clean)

import graphviz

def show_decision_tree(clf, feature_names):
    dot_data = sklearn.tree.export_graphviz(clf, out_file=None, 
                                    feature_names=feature_names,
                                    class_names=["BUG", "NO_BUG"],  
                                    filled=True, rounded=True)  
    return graphviz.Source(dot_data)

if __name__ == '__main__':
    show_decision_tree(clf, vec.get_feature_names_out())

import math

def friendly_decision_tree(clf, feature_names,
                           class_names = ['NO_BUG', 'BUG'],
                           indent=0):
    def _tree(index, indent):
        s = ""
        feature = clf.tree_.feature[index]
        feature_name = feature_names[feature]
        threshold = clf.tree_.threshold[index]
        value = clf.tree_.value[index]
        class_ = int(value[0][0])
        class_name = class_names[class_]
        left = clf.tree_.children_left[index]
        right = clf.tree_.children_right[index]
        if left == right:
            # Leaf node
            s += " " * indent + class_name + "\n"
        else:
            if math.isclose(threshold, 0.5):
                s += " " * indent + f"if {feature_name}:\n"
                s += _tree(right, indent + 2)
                s += " " * indent + f"else:\n"
                s += _tree(left, indent + 2)
            else:
                s += " " * indent + f"if {feature_name} <= {threshold:.4f}:\n"
                s += _tree(left, indent + 2)
                s += " " * indent + f"else:\n"
                s += _tree(right, indent + 2)
        return s

    ROOT_INDEX = 0
    return _tree(ROOT_INDEX, indent)

if __name__ == '__main__':
    print(friendly_decision_tree(clf, vec.get_feature_names_out()))

### Learning a Decision Tree

if __name__ == '__main__':
    print('\n### Learning a Decision Tree')



def train_tree(data):
    sample_bug_count = len(data[(data["oracle"].astype(str) == "BUG")])
    assert sample_bug_count > 0, "No bug samples found"
    sample_count = len(data)

    clf = DecisionTreeClassifier(min_samples_leaf=1,
                                     min_samples_split=2,  # minimal value
                                     max_features=None,
                                     max_depth=5, # max depth of the decision tree
                                     class_weight={str("BUG"): (1.0/sample_bug_count),
                                                   str("NO_BUG"):
                                                       (1.0/(sample_count - sample_bug_count))})
    clf = clf.fit(data.drop('oracle', axis=1), data['oracle'].astype(str))
    # MARTIN: This is optional, but is a nice extesion that results in nicer decision trees
    # clf = treetools.remove_infeasible(clf, features)
    return clf

## Step 3: Extract Feature Requirements
## ------------------------------------

if __name__ == '__main__':
    print('\n## Step 3: Extract Feature Requirements')



if __name__ == '__main__':
    features = [
        {'function-sqrt': 1, 'function-cos': 0, 'function-sin': 0, 'number': -900},
        {'function-sqrt': 0, 'function-cos': 1, 'function-sin': 0, 'number': 300},
        {'function-sqrt': 1, 'function-cos': 0, 'function-sin': 0, 'number': -1},
        {'function-sqrt': 0, 'function-cos': 1, 'function-sin': 0, 'number': -10},
        {'function-sqrt': 0, 'function-cos': 0, 'function-sin': 1, 'number': 36},
        {'function-sqrt': 0, 'function-cos': 0, 'function-sin': 1, 'number': -58},
        {'function-sqrt': 1, 'function-cos': 0, 'function-sin': 0, 'number': 27},
    ]

    # Labels for each input
    oracle = [
        "BUG",
        "NO_BUG",
        "BUG",
        "NO_BUG",
        "NO_BUG",
        "NO_BUG",
        "NO_BUG"
    ]

    # We can use the sklearn DictVectorizer to transform the features to numpy array:
    # Notice: Use the correct labeling of the feature_names

    # vec = DictVectorizer()
    # X_vec = vec.fit_transform(features).toarray()
    # feature_names = vec.get_feature_names_out()

    # We can also use a pandas DataFrame and directly parse it to the decision tree learner
    feature_names = ['function-sqrt', 'function-cos', 'function-sin', 'number']
    X_data = pandas.DataFrame.from_records(features)

    # Fix the random state to produce a deterministic result (for illustration purposes only)
    clf = DecisionTreeClassifier(random_state=10)

    # Train with DictVectorizer
    # **Note:** The sklearn `DictVectorizer` uses an internal sort function as default. This will result in different feature_name indices. If you want to use the `Dictvectorizer` please ensure that you only access the feature_names with the function `vec.get_feature_names_out()`.
    # We recommend that you use the `pandas` data frame, since this is also the format used in the feedback loop.
    # clf = clf.fit(X_vec, oracle)

    # Train with Pandas Dataframe
    clf = clf.fit(X_data, oracle)

    dot_data = sklearn.tree.export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=["BUG", "NO BUG"],
                                    filled=True, rounded=True)
    graph = graphviz.Source(dot_data)

if __name__ == '__main__':
    graph

if __name__ == '__main__':
    print(friendly_decision_tree(clf, feature_names, class_names = ['NO_BUG', 'BUG']))

### Excursion: Tree helper functions

if __name__ == '__main__':
    print('\n### Excursion: Tree helper functions')



def all_path(clf, node=0):
    """Iterate over all path in a decision tree. Path will be represented as
    a list of integers, each integer is the index of a node in the clf.tree_ structure."""
    left = clf.tree_.children_left[node]
    right = clf.tree_.children_right[node]

    if left == right:
        yield [node]
    else:
        for path in all_path(clf, left):
            yield [node] + path
        for path in all_path(clf, right):
            yield [node] + path


def path_samples(clf, path):
    """Returns the number of samples for this path. """
    return clf.tree_.n_node_samples[path[-1]]


def generic_feature_names(clf):
    """Gives a list of feature names of the form f1, f2, ..."""
    return ["f{}".format(f) for f in range(0, clf.tree_.n_features)]


def box(clf, path, data=None, feature_names=None):
    """For a decision tree classifier clf and a path path (as returned, e.g. by all_path),
    this method gives a pandas DataFrame with the min and max of each feature value on the given path."""

    if feature_names is None:
        feature_names = generic_feature_names(clf)
    check_for_duplicates(feature_names)
    if data is None:
        bounds = pandas.DataFrame([{'feature': c, 'min': -numpy.inf, 'max': numpy.inf} for c in feature_names],
                                  columns=['feature', 'min', 'max']).set_index(['feature']).transpose()
    else:
        bounds = pandas.DataFrame([{'feature': c, 'min': data[c].min(), 'max': data[c].max()} for c in feature_names],
                                  columns=['feature', 'min', 'max']).set_index(['feature']).transpose()

    for pos in range(0, len(path) - 1):
        node = path[pos]
        child = path[pos + 1]
        feature = feature_names[clf.tree_.feature[node]]
        threshold = clf.tree_.threshold[node]

        if child == clf.tree_.children_left[node]:
            bounds.at['max', feature] = threshold
        else:
            bounds.at['min', feature] = threshold
    return bounds


def rectangles(clf, colormap, data, feature_names=None):
    """yields matplotlib.patches rectangle objects. Each object represents a leaf of the tree."""
    if feature_names is None:
        feature_names = ['in_x', 'in_y']
    if 2 != len(feature_names):
        raise AssertionError("Rectangles can only be generated if there are at most 2 features.")

    x_feature = feature_names[0]
    y_feature = feature_names[1]

    for path in all_path(clf):
        b = box(clf, path, data=data, feature_names=feature_names)
        p = prediction_for_path(clf, path)
        c = colormap[p]
        rect = matplotlib.patches.Rectangle((b[x_feature]['min'], 
                                             b[y_feature]['min']),
                             # coordinates
                             b[x_feature]['max'] - b[x_feature]['min'],  # width
                             b[y_feature]['max'] - b[y_feature]['min'],  # height
                             alpha=.2, facecolor=c, edgecolor='k')
        yield rect


def prediction_for_path(clf, path) -> OracleResult:
    last_value = clf.tree_.value[path[-1]][0]
    p_class = numpy.argmax(last_value)
    return OracleResult(clf.classes_[p_class])


def rule(clf, path, feature_names, class_names=None):
    """Creates a rule from one path in the decision tree."""
    bounds = box(clf, path, feature_names=feature_names)
    prediction = prediction_for_path(clf, path)
    if class_names is not None:
        prediction = class_names[prediction]

    feature_rules = []
    for fname in feature_names:
        min_ = bounds[fname]['min']
        max_ = bounds[fname]['max']

        if numpy.isinf(min_) and numpy.isinf(max_):
            pass  # no rule if both are unbound
        elif numpy.isinf(min_):
            feature_rules.append("{} <= {:.4f}".format(fname, max_))
        elif numpy.isinf(max_):
            feature_rules.append("{} > {:.4f}".format(fname, min_))
        else:
            feature_rules.append("{} in {:.4f} to {:.4f}".format(fname, min_, max_))

    return " AND ".join(feature_rules), prediction, clf.tree_.impurity[path[-1]], clf.tree_.n_node_samples[path[-1]]


def rules(clf, class_names=None, feature_names=None):
    """Formats Decision trees in a rule-like representation."""

    if feature_names is None:
        feature_names = generic_feature_names(clf)

    samples = clf.tree_.n_node_samples[0]
    return "\n".join(["IF {2} THEN PREDICT '{3}' ({0}: {4:.4f}, support: {5} / {1})"
                     .format(clf.criterion, samples,
                             *rule(clf, path, feature_names, class_names=class_names)) for path in all_path(clf)])


def grouped_rules(clf, class_names=None, feature_names=None):
    """Formats decision trees in a rule-like representation, grouped by class."""

    if feature_names is None:
        feature_names = generic_feature_names(clf)

    rules = {}
    for path in all_path(clf):
        rulestr, clz, impurity, support = rule(clf, path, class_names=class_names, feature_names=feature_names)
        if clz not in rules:
            rules[clz] = []
        rules[clz].append((rulestr, impurity, support))

    res = ""
    samples = clf.tree_.n_node_samples[0]
    for clz in rules:
        rulelist = rules[clz]
        res = res + "\n{}:\n\t".format(clz)
        rl = ["{} ({}: {:.4f}, support: {}/{})".format(r, clf.criterion, impurity, support, samples) for r, impurity, support in rulelist]
        res = res + "\n\tor ".join(rl)
    return res.lstrip()


def check_for_duplicates(names):
    seen = set()
    for name in names:
        if name in seen:
            raise AssertionError("Duplicate name: {}".format(name))
        seen.add(name)


def is_leaf(clf, node: int) -> bool:
    """returns true if the given node is a leaf."""
    return clf.tree_.children_left[node] == clf.tree_.children_right[node]


def leaf_label(clf, node: int) -> int:
    """returns the index of the class at this node. The node must be a leaf."""
    assert(is_leaf(clf, node))
    occs = clf.tree_.value[node][0]
    idx = 0
    maxi = occs[idx]
    for i, o in zip(range(0, len(occs)), occs):
        if maxi < o:
            maxi = o
            idx = i
    return idx


def find_existence_index(features: List[Feature], feature: Feature):
    for idx, f in enumerate(features):
        if isinstance(f, ExistenceFeature) and f.key() == feature.key():
            return idx
    raise AssertionError("There is no existence feature with this key!")


def remove_infeasible(clf, features: List[Feature]):
    for node in range(0, clf.tree_.node_count):
        if not is_leaf(clf, node):
            feature = features[clf.tree_.feature[node]]
            threshold = clf.tree_.threshold[node]
            if not feature.is_feasible(threshold):
                clf.tree_.feature[node] = find_existence_index(features, feature)
                clf.tree_.threshold[node] = 0.5
    return clf


def iterate_nodes(clf):
    stack = [0]
    while 0 != len(stack):
        node = stack.pop()
        yield node
        if not is_leaf(clf, node):
            stack.append(clf.tree_.children_left[node])
            stack.append(clf.tree_.children_right[node])


def count_nodes(clf):
    return len(list(iterate_nodes(clf)))


def count_leaves(clf):
    return len([n for n in iterate_nodes(clf) if is_leaf(clf, n)])


def list_features(clf):
    return [clf.tree_.feature[node] for node in iterate_nodes(clf)]


def remove_unequal_decisions(clf):
    """
    This method rewrites a decision tree classifier to remove nodes where the same
    decision is taken on both sides.

    :param clf: a decision tree classifier
    :return: the same classifier, rewritten
    """
    changed = True
    while changed:
        changed = False
        for node in range(0, clf.tree_.node_count):
            if not is_leaf(clf, node) and (is_leaf(clf, clf.tree_.children_left[node]) and is_leaf(clf, clf.tree_.children_right[node])):
                # both children of this node are leaves
                left_label = leaf_label(clf, clf.tree_.children_left[node])
                right_label = leaf_label(clf, clf.tree_.children_right[node])
                if left_label == right_label:
                    clf.tree_.children_left[node] = -1
                    clf.tree_.children_right[node] = -1
                    clf.tree_.feature[node] = -2
                    changed = True
                    assert(left_label == leaf_label(clf, node))
    return clf


### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



### Excursion: Converting Trees to Paths

if __name__ == '__main__':
    print('\n### Excursion: Converting Trees to Paths')



class TreeRequirement:
    def __init__(self, feature: Feature, mini, maxi):
        self.__feature: Feature = feature
        self.__mini = mini
        self.__maxi = maxi

    def feature(self) -> Feature:
        return self.__feature

    def select(self, data):
        """Returns a vector of booleans, suitable for selecting in a pandas data frame."""
        if self.__mini is None:
            return data[self.__feature.name()] <= self.__maxi
        if self.__maxi is None:
            return self.__mini <= data[self.__feature.name()]
        return (self.__mini <= data[self.__feature.name()]) & (data[self.__feature.name()] <= self.__maxi)

    def mini(self):
        return self.__mini

    def maxi(self):
        return self.__maxi

    def get_key(self) -> str:
        return self.__feature.key()

    def is_binary(self) -> bool:
        return self.__feature.is_binary()

    def get_str(self, bounds) -> str:
        if self.is_binary():
            if self.__mini < 0 <= self.__maxi:
                # feature is NOT included
                return f"!{self.__feature.name()}"
            if self.__mini < 1 <= self.__maxi:
                # feature is included
                return self.__feature.name()
            raise AssertionError("How is this possible?")
        else:
            if (not numpy.isinf(self.__mini)) and (not numpy.isinf(self.__maxi)):
                return f"{self.__feature.name()} in [{self.__mini}, {self.__maxi}]"
            elif not numpy.isinf(self.__maxi):
                return f"{self.__feature.name()} <= {self.__maxi}"
            else:
                return f"{self.__feature.name()} > {self.__mini}"

    def get_str_ext(self) -> str:
        if (not numpy.isinf(self.__mini)) and (not numpy.isinf(self.__maxi)):
            return f"{self.__feature} in [{self.__mini}, {self.__maxi}]"
        elif not numpy.isinf(self.__maxi):
            return f"{self.__feature} <= {self.__maxi}"
        else:
            return f"{self.__feature} > {self.__mini}"

    def get_neg(self, bounds) -> List[str]:
        if self.is_binary():
            if self.__mini < 0 <= self.__maxi:
                # feature is NOT included, so, the negated condition is to include it
                return [self.__feature.name()]
            if self.__mini < 1 <= self.__maxi:
                # feature is included, so exclude it
                return [f"!{self.__feature.name()}"]
            raise AssertionError("How is this possible?")
        else:
            if (not numpy.isinf(self.__mini)) and (not numpy.isinf(self.__maxi)):
                return [f"{self.__feature.name()} in [{bounds.at['min', self.__feature.name()]},{self.__mini}]",
                        f"{self.__feature.name()} in [{self.__maxi}, {bounds.at['max', self.__feature.name()]}]"]
            elif not numpy.isinf(self.__maxi):
                return [f"{self.__feature.name()} <= {self.__maxi}"]
            else:
                return [f"{self.__feature.name()} > {self.__mini}"]

    def get_neg_ext(self, bounds) -> List[str]:
        if (not numpy.isinf(self.__mini)) and (not numpy.isinf(self.__maxi)):
            return [f"{self.__feature} in [{bounds.at['min', self.__feature]},{self.__mini}]",
                    f"{self.__feature} in [{self.__maxi}, {bounds.at['max', self.__feature]}]"]
        elif not numpy.isinf(self.__maxi):
            return [f"{self.__feature} > {self.__maxi}"]
        else:
            return [f"{self.__feature} <= {self.__mini}"]

from pathlib import Path

class TreePath:
    def __init__(self, samplefile: Optional[Path], is_bug: bool, requirements: List[TreeRequirement]):
        self.__sample = samplefile
        self.__is_bug = is_bug
        self.__requirements: List[TreeRequirement] = requirements

    def is_bug(self) -> bool:
        return self.__is_bug

    def get(self, idx):
        return self.__requirements[idx]

    def find_sample(self, data):
        for req in self.__requirements:
            data = data[req.select(data)]
        if 0 != len(data):
            return data["abs_file"][0]
        return None

    def __len__(self) -> int:
        return len(self.__requirements)


def lower_middle(start, end):
    if start == end:
        return start - abs(start)
    return start + ((end - start)/2)


def upper_middle(start, end):
    if start == end:
        return end + abs(end)
    return start + ((end - start)/2)


def min_digits(mini):
    return int("1" + "".join([0] * int(mini-1)))


def max_digits(maxi):
    return int("".join([9] * int(maxi)))

def tree_to_paths(tree, features: List[Feature]):
    paths = []
    # go through tree leaf by leaf
    for path in all_path(tree):
        requirements = []
        is_bug = OracleResult.BUG == prediction_for_path(tree, path)
        # find the requirements
        box_ = box(tree, path, feature_names=features).transpose()
        for feature, row in box_.iterrows():
            mini = row['min']
            maxi = row['max']
            if (not numpy.isinf(mini)) or (not numpy.isinf(maxi)):
                requirements.append(TreeRequirement(feature, mini, maxi))
        paths.append(TreePath(None, is_bug, requirements))

    return paths

if __name__ == '__main__':
    all_paths = tree_to_paths(clf, feature_names)

if __name__ == '__main__':
    for count, path in enumerate(all_paths):
        string_path = path.get(0).get_str_ext()
        for box_ in range(1, len(path)):
            string_path += " " + path.get(box_).get_str_ext()
        print(f"Path {count}: {string_path}, is_bug: {path.is_bug()}")

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



## Step 4: Generating New Samples
## ------------------------------

if __name__ == '__main__':
    print('\n## Step 4: Generating New Samples')



### Negating Requirements

if __name__ == '__main__':
    print('\n### Negating Requirements')



if __name__ == '__main__':
    x = pandas.DataFrame.from_records(features)
    bounds = pandas.DataFrame([{'feature': c, 'min': x[c].min(), 'max': x[c].max()}
                               for c in feature_names],
                              columns=['feature', 'min', 'max']).set_index(['feature']).transpose()

if __name__ == '__main__':
    for count, path in enumerate(all_paths):
        negated_string_path = path.get(0).get_neg_ext(bounds)[0]
        for box_ in range(1, len(path)):
            negated_string_path += " " + str(path.get(box_).get_neg_ext(bounds)[0])
        print(f"Path {count}: {negated_string_path}, is_bug: {path.is_bug()}")

### Systematically Negating Paths

if __name__ == '__main__':
    print('\n### Systematically Negating Paths')



def extracting_prediction_paths(clf, feature_names, data):
    # determine the bounds
    bounds = pandas.DataFrame([{'feature': c, 'min': data[c].min(), 'max': data[c].max()}
                           for c in feature_names],
                          columns=['feature', 'min', 'max']).set_index(['feature']).transpose()

    # go through tree leaf by leaf
    all_reqs = set()
    for path in tree_to_paths(clf, feature_names):
        # generate conditions
        for i in range(0, len(path)+1):
            reqs_list = []
            bins = format(i, "#0{}b".format(len(path)+2))[2:]
            for p, b in zip(range(0, len(bins)), bins):
                r = path.get(p)
                if '1' == b:
                    reqs_list.append(r.get_neg_ext(bounds))
                else:
                    reqs_list.append([r.get_str_ext()])
            for reqs in all_combinations(reqs_list):
                all_reqs.add(", ".join(sorted(reqs)))
    return all_reqs

def all_combinations(reqs_lists):
    result = [[]]
    for reqs in reqs_lists:
        t = []
        for r in reqs:
            for i in result:
                t.append(i+[r])
        result = t
    return result

if __name__ == '__main__':
    new_prediction_paths = extracting_prediction_paths(clf, feature_names, data=x)

if __name__ == '__main__':
    for path in new_prediction_paths:
        print(path)

### Input Specification Parser

if __name__ == '__main__':
    print('\n### Input Specification Parser')



import string

SPEC_GRAMMAR: Grammar = {
    "<start>":
        ["<req_list>"],

    "<req_list>": 
        ["<req>", "<req>"", ""<req_list>"],

    "<req>":
        ["<feature>"" ""<quant>"" ""<num>"],

    "<feature>": ["exists(<string>)",
                  "num(<string>)",
                  # currently not used
                  "char(<string>)",
                  "length(<string>)"],

    "<quant>":
        ["<", ">", "<=", ">="],

    "<num>": ["-<value>", "<value>"],

    "<value>":
        ["<integer>.<integer>",
         "<integer>"],

    "<integer>":
        ["<digit><integer>", "<digit>"],

    "<digit>":
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],

    '<string>': ['<letters>'],
    '<letters>': ['<letter><letters>', '<letter>'],
    '<letter>': list(string.ascii_letters + string.digits + string.punctuation)
}

assert is_valid_grammar(SPEC_GRAMMAR)

### Excursion: Validating the Parser

if __name__ == '__main__':
    print('\n### Excursion: Validating the Parser')



if __name__ == '__main__':
    g = GrammarFuzzer(SPEC_GRAMMAR, max_nonterminals=100)
    earley = EarleyParser(SPEC_GRAMMAR)
    for i in range(10):
        sample = g.fuzz()
        print(sample)

if __name__ == '__main__':
    g = GrammarFuzzer(SPEC_GRAMMAR, max_nonterminals= 100)
    earley = EarleyParser(SPEC_GRAMMAR)
    for i in range(100):
        sample = g.fuzz()
        for tree in earley.parse(sample):
            assert tree_to_string(tree) == sample, f"{tree_to_string(tree)} and {sample} are not equal"

if __name__ == '__main__':
    earley = EarleyParser(SPEC_GRAMMAR)
    teststrings = ['exists(<function>@0) > 0.5, exists(<term>@0) <= 0.5, exists(<value>@1) <= 0.5',
                   'exists(<digit>@9) <= 0.5, exists(<function>@0) > 0.5, num(<term>) > 0.05000000074505806',
                   'exists(<digit>@2) <= 0.5, exists(<function>@0) < 0.5, num(<term>) <= 0.05000000074505806',
                   'exists(<function>@0) > 0.5, num(<term>) > -3965678.1875']
    for count, sample in enumerate(teststrings):
        for tree in earley.parse(sample):
            assert tree_to_string(tree) == teststrings[count], \
            f"{tree_to_string(tree)} and {teststrings[count]} are not equal"

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



### Retrieving New input Specifications

if __name__ == '__main__':
    print('\n### Retrieving New input Specifications')



class SpecRequirement:
    '''
    This class represents a requirement for a new input sample that should be generated.
    This class contains the feature that should be fullfiled (Feature), a quantifier
    ("<", ">", "<=", ">=") and a value. For instance exist(feature) >= 0.5 states that
    the syntactical existence feature should be used to produce a new input.

    feature  : Is the associated feature class
    quant    : The quantifier
    value    : The value of the requirement. Note that for existence features this value
                is allways between 0 and 1.
    '''

    def __init__(self, feature: Feature, quantificator, value):
        self.feature: Feature = feature
        self.quant = quantificator
        self.value = value

    def __str__(self):
        return f"Requirement({self.feature.name} {self.quant} {self.value})"

    def __repr__(self):
        return f"Requirement({self.feature.name}, {self.quant}, {self.value})"

    def friendly(self):
        def value(x):
            try:
                return float(x)
            except Exception:
                return None

        if isinstance(self.feature, ExistenceFeature):
            if value(self.value) > 0:
                return f"{self.feature.friendly_name()}"
            elif value(self.value) < 0:
                return f"not {self.feature.friendly_name()}"

        return f"{self.feature.friendly_name()} {self.quant} {self.value}"

class InputSpecification:
    '''
    This class represents a complete input specification of a new input. A input specification
    consists of one or more requirements.
    requirements  : Is a list of all requirements that must be used.
    '''

    def __init__(self, requirements: List[SpecRequirement]):
        self.requirements: List[SpecRequirement] = requirements

    def __str__(self):
        s = ", ".join(str(r) for r in self.requirements)
        return f"InputSpecification({s})"

    def friendly(self):
        return " and ".join(r.friendly() for r in self.requirements)

    def __repr__(self):
        return self.__str__()

def get_all_subtrees(derivation_tree, non_terminal):
    '''
    Iteratively returns a list of subtrees that start with a given non_terminal.
    '''

    subtrees = []
    (node, children) = derivation_tree

    if node == non_terminal:
        subtrees.append(derivation_tree)

    for child in children:
        subtrees = subtrees + get_all_subtrees(child, non_terminal)

    return subtrees

def create_new_input_specification(derivation_tree, all_features) -> InputSpecification:
    '''
    This function creates a new input specification for a parsed decision tree path.
    The input derivation_tree corresponds to a already negated path in the decision tree.
    '''

    requirement_list = []

    for req in get_all_subtrees(derivation_tree, '<req>'):
        feature_name = tree_to_string(get_all_subtrees(req, '<feature>')[0])
        quant = tree_to_string(get_all_subtrees(req, '<quant>')[0])
        value = tree_to_string(get_all_subtrees(req, '<num>')[0])

        feature_class = None
        for f in all_features:
            if f.name == feature_name:
                feature_class = f

        requirement_list.append(SpecRequirement(feature_class, quant, value))

    return InputSpecification(requirement_list)

def get_all_input_specifications(dec_tree,
                                 all_features: List[Feature],
                                 feature_names: List[str],
                                 data) -> List[InputSpecification]:
    '''
    Returns a complete list new input specification that were extracted from a learned decision tree.

    INPUT:
        - dec_tree       : The learned decision tree.
        - all_features   : A list of all features
        - feature_names  : The list of the feature names (feature.name)
        - data.          : The data that was used to learn the decision tree

    OUTPUT:
        - Returns a list of InputSpecifications
    '''
    prediction_paths = extracting_prediction_paths(dec_tree, feature_names, data)
    input_specifications = []

    # parse all extracted paths
    for r in prediction_paths:
        earley = EarleyParser(SPEC_GRAMMAR)
        try:
            for tree in earley.parse(r):
                input_specifications.append(create_new_input_specification(tree, all_features))
        except SyntaxError:
            # Catch Parsing Syntax Errors: num(<term>) in [-900, 0] will fail; Might fix later
            # For now, inputs following that form will be ignored
            pass

    return input_specifications

### Excursion: Testing Specifications

if __name__ == '__main__':
    print('\n### Excursion: Testing Specifications')



if __name__ == '__main__':
    sample_prediction_paths = ['exists(<function>@0) > 0.5, num(<term>) <= -38244758.0',
                            'exists(<digit>@7) <= 0.5, exists(<function>@0) > 0.5, num(<term>) <= 0.05000000074505806',
                            'exists(<digit>) > 1.5, exists(<function>@0) > 0.5, num(<term>) <= 0.21850000321865082', 
                            'exists(<function>@0) > 0.5']

    expected_input_specifications = ['InputSpecification(Requirement(exists(<function>@0) > 0.5), Requirement(num(<term>) <= -38244758.0))',
                                     'InputSpecification(Requirement(exists(<digit>@7) <= 0.5), Requirement(exists(<function>@0) > 0.5), Requirement(num(<term>) <= 0.05000000074505806))',
                                     'InputSpecification(Requirement(exists(<digit>) > 1.5), Requirement(exists(<function>@0) > 0.5), Requirement(num(<term>) <= 0.21850000321865082))',
                                     'InputSpecification(Requirement(exists(<function>@0) > 0.5))']

    all_features = extract_all_features(CALC_GRAMMAR)

    earley = EarleyParser(SPEC_GRAMMAR)
    for count, sample in enumerate(sample_prediction_paths):
        for tree in earley.parse(sample):
            input_specification = create_new_input_specification(tree, all_features)
            assert str(input_specification) == expected_input_specifications[count], \
                f"{str(input_specification)} is not equal to {expected_input_specifications[count]}"

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



import time
import copy
from copy import deepcopy
import random
from itertools import chain

def best_trees(forest, spec):
    samples = [tree_to_string(tree) for tree in forest]
    fulfilled_fractions= []
    for sample in samples:
        gen_features = collect_features([sample], CALC_GRAMMAR)

        # calculate percentage of fulfilled requirements (used to rank the sample)
        fulfilled_count = 0
        total_count = len(spec.requirements)
        for req in spec.requirements:
            # for now, interpret requirement(exists(<...>) <= number) as false and requirement(exists(<...>) > number) as true
            if isinstance(req.feature, ExistenceFeature):
                expected = 1.0 if req.quant == '>' or req.quant == '>=' else 0.0
                actual = gen_features[req.feature.name][0]
                if actual == expected:
                    fulfilled_count += 1
                else:
                    pass
                    # print(f'{req.feature} expected: {expected}, actual:{actual}')
            elif isinstance(req.feature, NumericInterpretation):
                expected_value = float(req.value)
                actual_value = gen_features[req.feature.name][0]
                fulfilled = False
                if req.quant == '<':
                    fulfilled = actual_value < expected_value
                elif req.quant == '<=':
                    fulfilled = actual_value <= expected_value
                elif req.quant == '>':
                    fulfilled = actual_value > expected_value
                elif req.quant == '>=':
                    fulfilled = actual_value >= expected_value

                if fulfilled:
                    fulfilled_count += 1
                else:
                    pass
                    # print(f'{req.feature} expected: {expected_value}, actual:{actual_value}')
        fulfilled_fractions.append(fulfilled_count / total_count)
        # print(f'Fraction of fulfilled requirements: {fulfilled_count / total_count}')
    max_frac = max(fulfilled_fractions)
    best_chosen = []
    if max_frac == 1.0:
        return True, forest[fulfilled_fractions.index(1.0)]

    for i, t in enumerate(forest):
        if fulfilled_fractions[i] == max_frac:
            best_chosen.append(t)
    return False, best_chosen


def generate_samples_advanced(grammar: Grammar,
                     new_input_specifications: List[InputSpecification],
                     timeout: int) -> List[str]:

    # if there are no input specifications: generate some random samples
    if len(new_input_specifications) == 0:
        fuzzer = GrammarFuzzer(grammar)
        samples = [fuzzer.fuzz() for _ in range(100)]
        return samples

    final_samples = []
    each_spec_timeout = timeout / len(new_input_specifications)

    rhs_nonterminals = grammar.keys()# list(chain(*[nonterminals(expansion) for expansion in grammar[rule]]))

    fuzzer = GrammarFuzzer(grammar)

    for spec in new_input_specifications:
        done = False
        starttime = time.time()
        best_chosen = [fuzzer.fuzz_tree() for _ in range(100)]
        done, best_chosen = best_trees(best_chosen, spec)
        if done:
            final_samples.append(tree_to_string(best_chosen))

        while not done and time.time() - starttime < each_spec_timeout:
            # split in prefix, postfix and try to reach targets
            for tree in best_chosen:
                prefix_len = random.randint(1, 3)
                curr = tree
                valid = True
                for i in range(prefix_len):
                    nt, children = curr
                    poss_desc_idxs = []
                    for c_idx, c in enumerate(children):
                        s, _ = c
                        possible_descend = s in rhs_nonterminals
                        if possible_descend:
                            poss_desc_idxs.append(c_idx)
                    if len(poss_desc_idxs) < 1:
                        valid = False
                        break
                    desc = random.randint(0, len(poss_desc_idxs) - 1)
                    curr = children[poss_desc_idxs[desc]]
                if valid:
                    nt, _ = curr
                    for req in spec.requirements:
                        if isinstance(req.feature, NumericInterpretation) and nt == req.feature.key:
                            # hacky: generate a derivation tree for this numeric interpretation
                            hacky_grammar = copy.deepcopy(grammar)
                            hacky_grammar["<start>"] = [nt]
                            parser = EarleyParser(hacky_grammar)
                            try:
                                test = parser.parse(req.value)
                                x = list(test)[0]
                                _, s = x
                                # print(str(s[0]))
                                # replace curr in tree with this new tree
                                curr = s[0]
                            except SyntaxError:
                                pass
            done, best_chosen = best_trees(best_chosen, spec)
            if done:
                final_samples.append(tree_to_string(best_chosen))
        if not done:
            final_samples.extend([tree_to_string(t) for t in best_chosen])

    return final_samples

def generate_samples_random(grammar, new_input_specifications, num):
    f = GrammarFuzzer(grammar ,max_nonterminals=50, log=False)
    data = []
    for _ in range(num):
        new_input = f.fuzz()
        data.append(new_input)

    return data

if __name__ == '__main__':
    generate_samples = generate_samples_advanced

### Excursion: Some Tests

if __name__ == '__main__':
    print('\n### Excursion: Some Tests')



if __name__ == '__main__':
    exsqrt = ExistenceFeature('exists(<function>@0)', '<function>', 'sqrt')
    exdigit = ExistenceFeature('exists(<digit>)', '<digit>', '<digit>')

    reqDigit = SpecRequirement(exdigit, '>', '0.5')
    fbdDigit = SpecRequirement(exdigit, '<=', '0.5')

    req0 = SpecRequirement(exsqrt, '>', '-6.0')
    testspec0 = InputSpecification([req0, reqDigit])
    req1 = SpecRequirement(exsqrt, '<=', '-6.0')
    testspec1 = InputSpecification([req1, fbdDigit])

    numterm = NumericInterpretation('num(<term>)', '<term>')
    req2 = SpecRequirement(numterm, '<', '-31.0')
    testspec2 = InputSpecification([req2, req0, reqDigit])

    print('--generating samples--')
    # samples = generate_samples(CALC_GRAMMAR, [testspec0, testspec1], 10)
    samples = generate_samples(CALC_GRAMMAR, [testspec2], 10)
    samples

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



## Step 5: Executing New Inputs
## ----------------------------

if __name__ == '__main__':
    print('\n## Step 5: Executing New Inputs')



### The Alhazen Class

if __name__ == '__main__':
    print('\n### The Alhazen Class')



class Alhazen:
    def __init__(self,
                 runner: Any,
                 grammar: Grammar,
                 initial_inputs: List[str], /,
                 verbose: bool = False,
                 max_iterations: int = 10,
                 generator_timeout: int = 10):
        self._initial_inputs = initial_inputs
        self._runner = runner
        self._grammar = grammar
        self._verbose = verbose
        self._max_iter = max_iterations
        self._previous_samples = None
        self._data = None
        self._trees = []
        self._generator_timeout = generator_timeout
        self._setup()

class Alhazen(Alhazen):
    def _setup(self):
        self._previous_samples = self._initial_inputs

        self._all_features = extract_all_features(self._grammar)
        self._feature_names = [f.name for f in self._all_features]
        if self._verbose:
            print("Features:", ", ".join(f.friendly_name()
                                         for f in self._all_features))

class Alhazen(Alhazen):
    def _add_new_data(self, exec_data, feature_data):
        joined_data = exec_data.join(feature_data.drop(['sample'], axis=1))

        # Only add valid data
        new_data = joined_data[(joined_data['oracle'] != OracleResult.UNDEF)]
        new_data = joined_data.drop(joined_data[joined_data.oracle.astype(str) == "UNDEF"].index)
        if 0 != len(new_data):
            if self._data is None:
                self._data = new_data
            else:
                self._data = pandas.concat([self._data, new_data], sort=False)

class Alhazen(Alhazen):
    def execute_samples(self, sample_list = None):
        if sample_list is None:
            sample_list = self._initial_inputs

        data = []
        for sample in sample_list:
            result = self._runner(sample)
            data.append({"oracle": result })

        return pandas.DataFrame.from_records(data)

class Alhazen(Alhazen):
    def run(self):
        for iteration in range(1, self._max_iter + 1):
            if self._verbose:
                print(f"\nIteration #{iteration}")
            self._iterate(self._previous_samples)

class Alhazen(Alhazen):
    def all_trees(self, /, prune: bool = True):
        trees = self._trees
        if prune:
            trees = [remove_unequal_decisions(tree) for tree in self._trees]
        return trees

    def last_tree(self, /, prune: bool = True):
        return self.all_trees(prune=prune)[-1]

class Alhazen(Alhazen):
    def _iterate(self, sample_list):
        # Run samples, obtain test outcomes
        exec_data = self.execute_samples(sample_list)

        # Step 1: Extract features from the new samples
        feature_data = collect_features(sample_list, self._grammar)

        # Combine the new data with the already existing data
        self._add_new_data(exec_data, feature_data)
        # display(self._data)

        # Step 2: Train the Decision Tree Classifier
        dec_tree = train_tree(self._data)
        self._trees.append(dec_tree)

        if self._verbose:
            print("  Decision Tree:")
            all_features = extract_all_features(self._grammar)
            all_feature_names = [f.friendly_name() for f in all_features]
            print(friendly_decision_tree(dec_tree, all_feature_names, indent=4))

        # Step 3: Extract new requirements from the tree
        new_input_specifications = get_all_input_specifications(dec_tree,
                                                self._all_features,
                                                self._feature_names,
                                                self._data.drop(['oracle'], axis=1))
        if self._verbose:
            print(f"  New input specifications:")
            for spec in new_input_specifications:
                print(f"    {spec.friendly()}")

        # Step 4: Generate new inputs according to the new input specifications
        new_samples = generate_samples(self._grammar,
                                       new_input_specifications,
                                       self._generator_timeout)
        if self._verbose:
            print(f"  New samples:")
            print(f"    {', '.join(new_samples)}")

        self._previous_samples = new_samples

class Alhazen(Alhazen):
    def all_feature_names(self, friendly: bool = True) -> List[str]:
        if friendly:
            all_feature_names = [f.friendly_name() for f in self._all_features]
        else:
            all_feature_names = [f.name for f in self._all_features]
        return all_feature_names

class Alhazen(Alhazen):
    def show_decision_tree(self, tree = None, friendly: bool = True):
        return show_decision_tree(tree or self.last_tree(),
                                  self.all_feature_names())

class Alhazen(Alhazen):
    def friendly_decision_tree(self, tree = None):
        return friendly_decision_tree(tree or self.last_tree(),
                                      self.all_feature_names())

## A Sample Run
## ------------

if __name__ == '__main__':
    print('\n## A Sample Run')



MAX_ITERATIONS = 20
GENERATOR_TIMEOUT = 10 # timeout in seconds

if __name__ == '__main__':
    initial_sample_list

if __name__ == '__main__':
    alhazen = Alhazen(sample_runner, CALC_GRAMMAR, initial_sample_list,
                      verbose=True,
                      max_iterations=MAX_ITERATIONS,
                      generator_timeout=GENERATOR_TIMEOUT)
    alhazen.run()

if __name__ == '__main__':
    alhazen.last_tree()

if __name__ == '__main__':
    alhazen.show_decision_tree()

if __name__ == '__main__':
    print(alhazen.friendly_decision_tree())

import inspect

if __name__ == '__main__':
    print(inspect.getsource(task_sqrt))

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



if __name__ == '__main__':
    alhazen = Alhazen(sample_runner, CALC_GRAMMAR, initial_sample_list,
                      max_iterations=20)
    alhazen.run()

if __name__ == '__main__':
    alhazen.show_decision_tree()

if __name__ == '__main__':
    print(alhazen.friendly_decision_tree())

import inspect

if __name__ == '__main__':
    print(inspect.getsource(task_sqrt))

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


