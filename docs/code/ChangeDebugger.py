#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# "Isolating Failure-Inducing Changes" - a chapter of "The Debugging Book"
# Web site: https://www.debuggingbook.org/html/ChangeDebugger.html
# Last change: 2025-01-13 15:54:45+01:00
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
The Debugging Book - Isolating Failure-Inducing Changes

This file can be _executed_ as a script, running all experiments:

    $ python ChangeDebugger.py

or _imported_ as a package, providing classes, functions, and constants:

    >>> from debuggingbook.ChangeDebugger import <identifier>

but before you do so, _read_ it and _interact_ with it at:

    https://www.debuggingbook.org/html/ChangeDebugger.html

This chapter introduces a class `ChangeDebugger` that automatically determines failure-inducing code changes.

### High-Level Interface

You are given two Python source codes `source_pass` and `source_fail`, and a function `test()` that works using the definitions in `source_pass`, but raises an exception using the definitions in `source_fail`. Then, you can use `ChangeDebugger` as follows:

with ChangeDebugger(source_pass, source_fail) as cd:
    test()
cd


This will produce the failure-inducing change between `source_pass` and `source_fail`, using [Delta Debugging](DeltaDebugger.ipynb) to determine minimal differences in patches applied.

Here is an example. The function `test()` passes (raises no exception) if `remove_html_markup()` is defined as follows:

>>> print_content(source_pass, '.py')
def remove_html_markup(s):  # type: ignore
    tag = False
    out = ""

    for c in s:
        if c == '<':    # start of markup
            tag = True
        elif c == '>':  # end of markup
            tag = False
        elif not tag:
            out = out + c

    return out
>>> def test() -> None:
>>>     assert remove_html_markup('"foo"') == '"foo"'
>>> exec(source_pass)
>>> test()

If `remove_html_markup()` is changed as follows, though, then
`test()` raises an exception and fails:

>>> print_content(source_fail, '.py')
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

    return out
>>> exec(source_fail)
>>> with ExpectError(AssertionError):
>>>     test()
Traceback (most recent call last):
  File "/var/folders/n2/xd9445p97rb3xh7m1dfx8_4h0006ts/T/ipykernel_3699/4262003862.py", line 3, in 
    test()
  File "/var/folders/n2/xd9445p97rb3xh7m1dfx8_4h0006ts/T/ipykernel_3699/3045937450.py", line 2, in test
    assert remove_html_markup('"foo"') == '"foo"'
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError (expected)


We can use `ChangeDebugger` to automatically identify the failure-inducing difference:

>>> with ChangeDebugger(source_pass, source_fail) as cd:
>>>     test()
>>> cd
@@ -215,24 +215,97 @@
 tag = False

+        elif c == '"' or c == "'" and tag:
            quote = not quote

         elif

The lines prefixed with `+` from are the ones in `source_fail` that cause the failure when added. (They also are the ones that should be fixed.)

### Programmatic Interface

For more details or more automation, use the programmatic interface. The method `min_patches()` returns a triple (`pass_patches`, `fail_patches`, `diffs`) where

* applying `pass_patches` still make the call pass
* applying `fail_patches` causes the call to fail
* `diffs` is the (minimal) difference between the two.

The patches come as list of `patch_obj` objects, as defined by Google's [diff-match-patch library](https://github.com/google/diff-match-patch).

>>> pass_patches, fail_patches, diffs = cd.min_patches()

One can apply all patches in `pass_patches` and still not cause the test to fail:

>>> for p in pass_patches:
>>>     print_patch(p)
@@ -48,24 +48,42 @@
 tag = False

+    quote = False

     out = ""

@@ -104,50 +104,43 @@
  s:

-        if c == '<':    # start of markup

+        if c == '<' and not quote:


However, as soon as we also apply the patches in `diffs`, we get the failure. (This is also what is shown when we output a `ChangeDebugger`.)

>>> for p in diffs:
>>>     print_patch(p)
@@ -215,24 +215,97 @@
 tag = False

+        elif c == '"' or c == "'" and tag:
            quote = not quote

         elif


The full set of methods in `ChangeDebugger` is shown below.
### Supporting Functions

`ChangeDebugger` relies on lower level `patch()` and `diff()` functions.

To apply patch objects on source code, use the `patch()` function. It takes a source code and a list of patches to be applied.

>>> print_content(patch(source_pass, diffs), '.py')
def remove_html_markup(s):  # type: ignore
    tag = False
    out = ""

    for c in s:
        if c == '<':    # start of markup
            tag = True
        elif c == '>':  # end of markup
            tag = False
        elif c == '"' or c == "'" and tag:
            quote = not quote
        elif not tag:
            out = out + c

    return out

Conversely, the `diff()` function computes patches between two texts. It returns a list of patch objects that can be applied on text.

>>> for p in diff(source_pass, source_fail):
>>>     print_patch(p)
@@ -48,24 +48,42 @@
 tag = False

+    quote = False

     out = ""
@@ -104,50 +104,43 @@
  s:

-        if c == '<':    # start of markup

+        if c == '<' and not quote:
@@ -162,48 +162,45 @@
 rue

-        elif c == '>':  # end of markup

+        elif c == '>' and not quote:
@@ -215,24 +215,97 @@
 tag = False

+        elif c == '"' or c == "'" and tag:
            quote = not quote

         elif



For more details, source, and documentation, see
"The Debugging Book - Isolating Failure-Inducing Changes"
at https://www.debuggingbook.org/html/ChangeDebugger.html
'''


# Allow to use 'from . import <module>' when run as script (cf. PEP 366)
if __name__ == '__main__' and __package__ is None:
    __package__ = 'debuggingbook'


# Isolating Failure-Inducing Changes
# ==================================

if __name__ == '__main__':
    print('# Isolating Failure-Inducing Changes')



if __name__ == '__main__':
    from .bookutils import YouTubeVideo
    YouTubeVideo("hX9ViNEXGL8")

if __name__ == '__main__':
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)

from .bookutils import quiz, print_file, print_content

from typing import Dict, Callable, TextIO, List, Tuple, Set, Any, Type, Optional

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



## Changes and Bugs
## ----------------

if __name__ == '__main__':
    print('\n## Changes and Bugs')



## Leveraging Version Histories
## ----------------------------

if __name__ == '__main__':
    print('\n## Leveraging Version Histories')



from graphviz import Digraph, nohtml

if __name__ == '__main__':
    from IPython.display import display

PASS = "✔"
FAIL = "✘"
UNRESOLVED = "?"

PASS_COLOR = 'darkgreen'  # '#006400' # darkgreen
FAIL_COLOR = 'red4'  # '#8B0000' # darkred

STEP_COLOR = 'peachpuff'
FONT_NAME = 'Raleway'

def graph(comment: str = "default") -> Digraph:
    return Digraph(name='', comment=comment,
                   graph_attr={
                       'rankdir': 'LR',
                   },
                   node_attr={
                       'style': 'filled',
                       'shape': 'box',
                       'fillcolor': STEP_COLOR,
                       'fontname': FONT_NAME,
                   },
                   edge_attr={
                       'fontname': FONT_NAME,
                   })

VERSIONS = 8


def display_versions(outcomes: Dict[int, str]) -> Digraph:
    state_machine = graph()
    for version_number in range(1, VERSIONS + 1):
        id = f'v{version_number}'
        label = f' {outcomes [version_number]}' \
            if version_number in outcomes else ''
        state_machine.node(id, label=f'{id}{label}')
        if version_number > 1:
            last_id = f'v{version_number - 1}'
            state_machine.edge(last_id, id)

    display(state_machine)

if __name__ == '__main__':
    display_versions({1: PASS, 8: FAIL})

## An Example Version History
## --------------------------

if __name__ == '__main__':
    print('\n## An Example Version History')



### Create a Working Directory

if __name__ == '__main__':
    print('\n### Create a Working Directory')



PROJECT = 'my_project'

import os
import shutil

if __name__ == '__main__':
    try:
        shutil.rmtree(PROJECT)
    except FileNotFoundError:
        pass
    os.mkdir(PROJECT)

import sys

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    os.chdir(PROJECT)

### Initialize Git

if __name__ == '__main__':
    print('\n### Initialize Git')



if __name__ == '__main__':
    import os
    os.system(f'git init')

if __name__ == '__main__':
    import os
    os.system(f'git config user.name "Demo User"')

if __name__ == '__main__':
    import os
    os.system(f'git config user.email "demo-user@example.com"')

if __name__ == '__main__':
    import os
    os.system(f'git config advice.detachedHead False')

def remove_html_markup(s):  # type: ignore
    tag = False
    out = ""

    for c in s:
        if c == '<':    # start of markup
            tag = True
        elif c == '>':  # end of markup
            tag = False
        elif not tag:
            out = out + c

    return out

import inspect

def write_source(fun: Callable, filename: Optional[str] = None) -> None:
    if filename is None:
        filename = fun.__name__ + '.py'
    with open(filename, 'w') as fh:
        fh.write(inspect.getsource(fun))

if __name__ == '__main__':
    write_source(remove_html_markup)

if __name__ == '__main__':
    print_file('remove_html_markup.py')

if __name__ == '__main__':
    import os
    os.system(f'git add remove_html_markup.py')

if __name__ == '__main__':
    import os
    os.system(f'git commit -m "First version"')

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

    return out

if __name__ == '__main__':
    write_source(remove_html_markup)

if __name__ == '__main__':
    import os
    os.system(f'git diff remove_html_markup.py')

if __name__ == '__main__':
    import os
    os.system(f'git commit -m "Second version" remove_html_markup.py')

### Excursion: Adding More Revisions

if __name__ == '__main__':
    print('\n### Excursion: Adding More Revisions')



def remove_html_markup(s):  # type: ignore
    tag = False
    quote = False
    out = ""

    for c in s:
        print("c =", repr(c), "tag =", tag, "quote =", quote)

        if c == '<' and not quote:
            tag = True
        elif c == '>' and not quote:
            tag = False
        elif c == '"' or c == "'" and tag:
            quote = not quote
        elif not tag:
            out = out + c

    return out

if __name__ == '__main__':
    write_source(remove_html_markup)

if __name__ == '__main__':
    import os
    os.system(f'git commit -m "Third version (with debugging output)" remove_html_markup.py')

def remove_html_markup(s):  # type: ignore
    tag = False
    quote = False
    out = ""

    for c in s:
        if c == '<':  # and not quote:
            tag = True
        elif c == '>':  # and not quote:
            tag = False
        elif c == '"' or c == "'" and tag:
            quote = not quote
        elif not tag:
            out = out + c

    return out

if __name__ == '__main__':
    write_source(remove_html_markup)

if __name__ == '__main__':
    import os
    os.system(f'git commit -m "Fourth version (clueless)" remove_html_markup.py')

def remove_html_markup(s):  # type: ignore
    tag = False
    quote = False
    out = ""

    for c in s:
        assert not tag  # <=== Just added

        if c == '<' and not quote:
            tag = True
        elif c == '>' and not quote:
            tag = False
        elif c == '"' or c == "'" and tag:
            quote = not quote
        elif not tag:
            out = out + c

    return out

if __name__ == '__main__':
    write_source(remove_html_markup)

if __name__ == '__main__':
    import os
    os.system(f'git commit -m "Fifth version (with assert)" remove_html_markup.py')

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
            assert False  # <=== Just added
            quote = not quote
        elif not tag:
            out = out + c

    return out

if __name__ == '__main__':
    write_source(remove_html_markup)

if __name__ == '__main__':
    import os
    os.system(f'git commit -m "Sixth version (with another assert)" remove_html_markup.py')

def remove_html_markup(s):  # type: ignore
    tag = False
    quote = False
    out = ""

    for c in s:
        if c == '<' and not quote:
            tag = True
        elif c == '>' and not quote:
            tag = False
        elif (c == '"' or c == "'") and tag:  # <-- FIX
            quote = not quote
        elif not tag:
            out = out + c

    return out

if __name__ == '__main__':
    write_source(remove_html_markup)

if __name__ == '__main__':
    import os
    os.system(f'git commit -m "Seventh version (fixed)" remove_html_markup.py')

### End of Excursion

if __name__ == '__main__':
    print('\n### End of Excursion')



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
    write_source(remove_html_markup)

if __name__ == '__main__':
    import os
    os.system(f'git commit -m "Eighth version (with proper assertion)" remove_html_markup.py')

if __name__ == '__main__':
    remove_html_markup('"foo"')

from .ExpectError import ExpectError

if __name__ == '__main__':
    with ExpectError():
        assert remove_html_markup('"foo"') == '"foo"'

## Accessing Versions
## ------------------

if __name__ == '__main__':
    print('\n## Accessing Versions')



if __name__ == '__main__':
    import os
    os.system(f'git log')

import subprocess

def get_output(command: List[str]) -> str:
    result = subprocess.run(command, 
                            stdout=subprocess.PIPE,
                            universal_newlines=True)
    return result.stdout

if __name__ == '__main__':
    log = get_output(['git', 'log', '--pretty=oneline'])
    print(log)

if __name__ == '__main__':
    versions = [line.split()[0] for line in log.split('\n') if line]
    versions.reverse()
    versions[0]

if __name__ == '__main__':
    import os
    os.system(f'git checkout {versions[0]}')

if __name__ == '__main__':
    print_file('remove_html_markup.py')

if __name__ == '__main__':
    exec(open('remove_html_markup.py').read())

if __name__ == '__main__':
    remove_html_markup('"foo"')

if __name__ == '__main__':
    import os
    os.system(f'git checkout {versions[7]}')

if __name__ == '__main__':
    print_file('remove_html_markup.py')

if __name__ == '__main__':
    exec(open('remove_html_markup.py').read())

if __name__ == '__main__':
    remove_html_markup('"foo"')

## Manual Bisecting
## ----------------

if __name__ == '__main__':
    print('\n## Manual Bisecting')



if __name__ == '__main__':
    import os
    os.system(f'git bisect start')

if __name__ == '__main__':
    import os
    os.system(f'git bisect good {versions[0]}')

if __name__ == '__main__':
    import os
    os.system(f'git bisect bad {versions[7]}')

if __name__ == '__main__':
    display_versions({1: PASS, 4: UNRESOLVED, 8: FAIL})

if __name__ == '__main__':
    print_file('remove_html_markup.py')

if __name__ == '__main__':
    exec(open('remove_html_markup.py').read())

if __name__ == '__main__':
    remove_html_markup('"foo"')

if __name__ == '__main__':
    import os
    os.system(f'git bisect bad')

if __name__ == '__main__':
    display_versions({1: PASS, 3: UNRESOLVED, 4: FAIL, 8: FAIL})

if __name__ == '__main__':
    print_file('remove_html_markup.py')

if __name__ == '__main__':
    exec(open('remove_html_markup.py').read())

if __name__ == '__main__':
    remove_html_markup('"foo"')

if __name__ == '__main__':
    import os
    os.system(f'git bisect bad')

if __name__ == '__main__':
    display_versions({1: PASS, 2: UNRESOLVED, 3: FAIL, 4: FAIL, 8: FAIL})

if __name__ == '__main__':
    print_file('remove_html_markup.py')

if __name__ == '__main__':
    exec(open('remove_html_markup.py').read())

if __name__ == '__main__':
    remove_html_markup('"foo"')

if __name__ == '__main__':
    display_versions({1: PASS, 2: FAIL, 3: FAIL, 4: FAIL, 8: FAIL})

if __name__ == '__main__':
    import os
    os.system(f'git bisect bad')

if __name__ == '__main__':
    import os
    os.system(f'git diff HEAD^')

if __name__ == '__main__':
    import os
    os.system(f'git bisect reset')

## Automatic Bisecting
## -------------------

if __name__ == '__main__':
    print('\n## Automatic Bisecting')



if __name__ == '__main__':
    open('test.py', 'w').write('''
#!/usr/bin/env python

from remove_html_markup import remove_html_markup
import sys

result = remove_html_markup('"foo"')
if result == '"foo"':
    sys.exit(0)  # good/pass
elif result == 'foo':
    sys.exit(1)  # bad/fail
else:
    sys.exit(125)  # unresolved
''');

if __name__ == '__main__':
    print_file('test.py')

if __name__ == '__main__':
    import os
    os.system(f'python ./test.py; echo $?')

if __name__ == '__main__':
    import os
    os.system(f'git bisect start')

if __name__ == '__main__':
    import os
    os.system(f'git bisect good {versions[0]}')

if __name__ == '__main__':
    import os
    os.system(f'git bisect bad {versions[7]}')

if __name__ == '__main__':
    import os
    os.system(f'git bisect run python test.py')

if __name__ == '__main__':
    import os
    os.system(f'git diff HEAD^')

if __name__ == '__main__':
    import os
    os.system(f'git bisect reset')

## Computing and Applying Patches
## ------------------------------

if __name__ == '__main__':
    print('\n## Computing and Applying Patches')



if __name__ == '__main__':
    version_1 = get_output(['git', 'show',
                            f'{versions[0]}:remove_html_markup.py'])

if __name__ == '__main__':
    print_content(version_1, '.py')

if __name__ == '__main__':
    version_2 = get_output(['git', 'show',
                            f'{versions[1]}:remove_html_markup.py'])

if __name__ == '__main__':
    print_content(version_2, '.py')

if __name__ == '__main__':
    import os
    os.system(f'git diff {versions[0]} {versions[1]}')

from diff_match_patch import diff_match_patch, patch_obj

def diff(s1: str, s2: str, mode: str = 'lines') -> List[patch_obj]:
    """Compare s1 and s2 like `diff`; return a list of patches"""

    # Sometimes, we may get bytes instead of strings
    # Let's convert these in a conservative way
    if not isinstance(s1, str):
        s1 = str(s1, 'latin1')
    if not isinstance(s2, str):
        s2 = str(s2, 'latin1')

    dmp = diff_match_patch()
    if mode == 'lines':
        (text1, text2, linearray) = dmp.diff_linesToChars(s1, s2)

        diffs = dmp.diff_main(text1, text2)
        dmp.diff_charsToLines(diffs, linearray)
        return dmp.patch_make(diffs)

    if mode == 'chars':
        diffs = dmp.diff_main(s1, s2)
        return dmp.patch_make(s1, diffs)

    raise ValueError("mode must be 'lines' or 'chars'")

if __name__ == '__main__':
    patches = diff(version_1, version_2)
    patches

import urllib

def patch_string(p: patch_obj) -> str:
    return urllib.parse.unquote(str(p).strip())  # type: ignore

def print_patch(p: patch_obj) -> None:
    print_content(patch_string(p), '.py')
    print()

if __name__ == '__main__':
    for p in patches:
        print_patch(p)

def patch(text: str, patches: List[patch_obj]) -> str:
    """Apply given patches on given text; return patched text."""
    dmp = diff_match_patch()
    patched_text, success = dmp.patch_apply(patches, text)
    assert all(success), "Could not apply some patch(es)"
    return patched_text

if __name__ == '__main__':
    print_content(patch(version_1, patches), '.py')

if __name__ == '__main__':
    assert patch(version_1, patches) == version_2

if __name__ == '__main__':
    assert patch(version_1, []) == version_1

if __name__ == '__main__':
    print(patch_string(patches[0]))

if __name__ == '__main__':
    print_content(patch(version_1, [patches[0]]), '.py')

if __name__ == '__main__':
    print_content(patch(version_1, [patches[1]]), '.py')

if __name__ == '__main__':
    quiz("What has changed in version 1 after applying the second patch?",
         [
             "The initialization of quote is deleted",
             "The condition after `if c == '<'` is expanded",
             "The tag variable gets a different value",
             "None of the above"
         ], '1 / 1 + 1 ** 1 - 1 % 1 * 1')

## Delta Debugging on Patches
## --------------------------

if __name__ == '__main__':
    print('\n## Delta Debugging on Patches')



def test_remove_html_markup_patches(patches: patch_obj) -> None:
    new_version = patch(version_1, patches)
    exec(new_version, globals())
    assert remove_html_markup('"foo"') == '"foo"'

if __name__ == '__main__':
    test_remove_html_markup_patches([])

if __name__ == '__main__':
    with ExpectError(AssertionError):
        test_remove_html_markup_patches(patches)

### A Minimal Set of Patches

if __name__ == '__main__':
    print('\n### A Minimal Set of Patches')



from .DeltaDebugger import DeltaDebugger

if __name__ == '__main__':
    with DeltaDebugger() as dd:
        test_remove_html_markup_patches(patches)

if __name__ == '__main__':
    reduced_patches = dd.min_args()['patches']

if __name__ == '__main__':
    for p in reduced_patches:
        print_patch(p)

if __name__ == '__main__':
    print_content(patch(version_1, reduced_patches), '.py')

### A Minimal Difference

if __name__ == '__main__':
    print('\n### A Minimal Difference')



if __name__ == '__main__':
    pass_patches, fail_patches, diffs = \
        tuple(arg['patches'] for arg in dd.min_arg_diff())

if __name__ == '__main__':
    print_content(patch(version_1, pass_patches), '.py')

if __name__ == '__main__':
    print_content(patch(version_1, fail_patches), '.py')

if __name__ == '__main__':
    for p in diffs:
        print_patch(p)

## A ChangeDebugger class
## ----------------------

if __name__ == '__main__':
    print('\n## A ChangeDebugger class')



from .DeltaDebugger import CallCollector

class ChangeDebugger(CallCollector):
    def __init__(self, pass_source: str, fail_source: str, **ddargs: Any) -> None:
        """Constructor. Takes a passing source file (`pass_source`)
        and a failing source file (`fail_source`).
        Additional arguments are passed to `DeltaDebugger` constructor.
        """
        super().__init__()
        self._pass_source = pass_source
        self._fail_source = fail_source
        self._patches = diff(pass_source, fail_source)
        self._ddargs = ddargs
        self.log = ddargs['log'] if 'log' in ddargs else False

    def pass_source(self) -> str:
        """Return the passing source file."""
        return self._pass_source

    def fail_source(self) -> str:
        """Return the failing source file."""
        return self._fail_source

    def patches(self) -> List[patch_obj]:
        """Return the diff between passing and failing source files."""
        return self._patches

def test_remove_html_markup() -> None:
    assert remove_html_markup('"foo"') == '"foo"'

if __name__ == '__main__':
    with ChangeDebugger(version_1, version_2) as cd:
        test_remove_html_markup()

if __name__ == '__main__':
    with ExpectError(AssertionError):
        cd.call()

if __name__ == '__main__':
    print_content(cd.pass_source(), '.py')

if __name__ == '__main__':
    print_content(cd.fail_source(), '.py')

if __name__ == '__main__':
    cd.patches()

class ChangeDebugger(ChangeDebugger):
    def test_patches(self, patches: List[patch_obj]) -> None:
        new_version = patch(self.pass_source(), patches)
        exec(new_version, globals())
        self.call()

class ChangeDebugger(ChangeDebugger):
    def __enter__(self) -> Any:
        """Called at begin of a `with` block. Checks if current source fails."""
        exec(self.fail_source(), globals())
        return super().__enter__()

if __name__ == '__main__':
    with ChangeDebugger(version_1, version_2) as cd:
        test_remove_html_markup()

if __name__ == '__main__':
    cd.test_patches([])

if __name__ == '__main__':
    with ExpectError(AssertionError):
        cd.test_patches(cd.patches())

class ChangeDebugger(ChangeDebugger):
    def min_patches(self) -> Tuple[List[patch_obj], List[patch_obj], List[patch_obj]]:
        """
        Compute a minimal set of patches.
        Returns a triple (`pass_patches`, `fail_patches`, `diff_patches`) 
        where `diff_patches` is the minimal difference between 
        the set `pass_patches` (which, when applied, make the test pass) and 
        the set `fail_patches` (which, when applied, make the test fail).
        """
        patches = self.patches()
        with DeltaDebugger(**self._ddargs) as dd:
            self.test_patches(patches)

        args = dd.min_arg_diff()
        pass_patches = args[0]['patches']
        fail_patches = args[1]['patches']
        diff_patches = args[2]['patches']

        return (pass_patches, fail_patches, diff_patches)

class ChangeDebugger(ChangeDebugger):
    def __repr__(self) -> str:
        """Return readable list of minimal patches"""
        pass_patches, fail_patches, diff_patches = self.min_patches()
        return "".join(patch_string(p) for p in diff_patches)

if __name__ == '__main__':
    with ChangeDebugger(version_1, version_2) as cd:
        test_remove_html_markup()

if __name__ == '__main__':
    cd.patches()

if __name__ == '__main__':
    pass_patches, fail_patches, diffs = cd.min_patches()
    diffs

if __name__ == '__main__':
    print(patch_string(diffs[0]))

if __name__ == '__main__':
    cd

if __name__ == '__main__':
    version_8 = get_output(['git', 'show', 
                                f'{versions[7]}:remove_html_markup.py'])

if __name__ == '__main__':
    with ChangeDebugger(version_1, version_8) as cd:
        test_remove_html_markup()

if __name__ == '__main__':
    len(cd.patches())

if __name__ == '__main__':
    cd

from .DeltaDebugger import NoCallError, NotFailingError

class NotPassingError(ValueError):
    pass

class ChangeDebugger(ChangeDebugger):
    def after_collection(self) -> None:
        """Diagnostics."""
        if self.function() is None:
            raise NoCallError("No function call observed")
        if self.exception() is None:
            raise NotFailingError(f"{self.format_call()} did not raise an exception")

        try:
            self.test_patches([])
        except Exception:
            raise NotPassingError(f"{self.format_call()} raised an exception in its passing version")

        try:
            self.test_patches(self.patches())
            raise NotFailingError(f"{self.format_call()} did not raise an exception in failing version")
        except Exception:
            pass

        if self.log:
            print(f"Observed {self.format_call()}" +
                  f" raising {self.format_exception(self.exception())}")  

if __name__ == '__main__':
    with ExpectError(NotPassingError):
        with ChangeDebugger(version_1, version_2) as cd:
            test_remove_html_markup()

## Synopsis
## --------

if __name__ == '__main__':
    print('\n## Synopsis')



### High-Level Interface

if __name__ == '__main__':
    print('\n### High-Level Interface')



if __name__ == '__main__':
    source_pass = version_1
    source_fail = version_2

if __name__ == '__main__':
    print_content(source_pass, '.py')

def test() -> None:
    assert remove_html_markup('"foo"') == '"foo"'

if __name__ == '__main__':
    exec(source_pass)
    test()

if __name__ == '__main__':
    print_content(source_fail, '.py')

if __name__ == '__main__':
    exec(source_fail)
    with ExpectError(AssertionError):
        test()

if __name__ == '__main__':
    with ChangeDebugger(source_pass, source_fail) as cd:
        test()
    cd

### Programmatic Interface

if __name__ == '__main__':
    print('\n### Programmatic Interface')



if __name__ == '__main__':
    pass_patches, fail_patches, diffs = cd.min_patches()

if __name__ == '__main__':
    for p in pass_patches:
        print_patch(p)

if __name__ == '__main__':
    for p in diffs:
        print_patch(p)

from .ClassDiagram import display_class_hierarchy

if __name__ == '__main__':
    display_class_hierarchy([ChangeDebugger],
                            public_methods=[
                                CallCollector.__init__,
                                CallCollector.__enter__,
                                CallCollector.__exit__,
                                CallCollector.call,  # type: ignore
                                CallCollector.args,
                                CallCollector.function,
                                CallCollector.exception,
                                ChangeDebugger.__init__,
                                ChangeDebugger.min_patches,
                                ChangeDebugger.patches,
                                ChangeDebugger.pass_source,
                                ChangeDebugger.fail_source,
                                ChangeDebugger.__repr__,
                                ChangeDebugger.__enter__
                            ],
                            project='debuggingbook')

### Supporting Functions

if __name__ == '__main__':
    print('\n### Supporting Functions')



if __name__ == '__main__':
    print_content(patch(source_pass, diffs), '.py')

if __name__ == '__main__':
    for p in diff(source_pass, source_fail):
        print_patch(p)

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



if __name__ == '__main__':
    try:
        shutil.rmtree(PROJECT)
    except FileNotFoundError:
        pass

## Exercises
## ---------

if __name__ == '__main__':
    print('\n## Exercises')



### Exercise 1: Fine-Grained Changes

if __name__ == '__main__':
    print('\n### Exercise 1: Fine-Grained Changes')



if __name__ == '__main__':
    patches = diff(version_1, version_2, mode='chars')

if __name__ == '__main__':
    for p in patches:
        print(patch_string(p))

### Exercise 2: Failure-Inducing Changes in the Large

if __name__ == '__main__':
    print('\n### Exercise 2: Failure-Inducing Changes in the Large')



### Exercise 3: Hierarchical Change Debugging

if __name__ == '__main__':
    print('\n### Exercise 3: Hierarchical Change Debugging')


