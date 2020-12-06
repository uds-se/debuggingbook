#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Fuzzing Book".
# Web site: https://www.fuzzingbook.org/html/ChangeDebugger.html
# Last change: 2020-12-06 17:03:52+01:00
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


# # Failure-Inducing Changes

if __name__ == "__main__":
    print('# Failure-Inducing Changes')




if __name__ == "__main__":
    from bookutils import YouTubeVideo
    YouTubeVideo("w4u5gCgPlmg")


if __name__ == "__main__":
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)


if __package__ is None or __package__ == "":
    from bookutils import quiz, print_file, print_content
else:
    from .bookutils import quiz, print_file, print_content


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




# ## A Version History

if __name__ == "__main__":
    print('\n## A Version History')




# ### Create a Working Directory

if __name__ == "__main__":
    print('\n### Create a Working Directory')




PROJECT = 'my_project'

import os
import shutil

if __name__ == "__main__":
    try:
        shutil.rmtree(PROJECT)
    except FileNotFoundError:
        pass
    os.mkdir(PROJECT)


import sys

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    os.chdir(PROJECT)


# ### Initialize git

if __name__ == "__main__":
    print('\n### Initialize git')




if __name__ == "__main__":
    import os
    os.system(f'git init')


if __name__ == "__main__":
    import os
    os.system(f'git config advice.detachedHead False')


def remove_html_markup(s):
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

def write_source(fun, filename=None):
    if filename is None:
        filename = fun.__name__ + '.py'
    with open(filename, 'w') as fh:
        fh.write(inspect.getsource(fun))

if __name__ == "__main__":
    write_source(remove_html_markup)


if __name__ == "__main__":
    print_file('remove_html_markup.py')


if __name__ == "__main__":
    import os
    os.system(f'git add remove_html_markup.py')


if __name__ == "__main__":
    import os
    os.system(f'git commit -m "First version"')


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

if __name__ == "__main__":
    write_source(remove_html_markup)


if __name__ == "__main__":
    import os
    os.system(f'git diff remove_html_markup.py')


if __name__ == "__main__":
    import os
    os.system(f'git commit -m "Second version" remove_html_markup.py')


def remove_html_markup(s):
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

if __name__ == "__main__":
    write_source(remove_html_markup)


if __name__ == "__main__":
    import os
    os.system(f'git commit -m "Third version (with debugging output)" remove_html_markup.py')


def remove_html_markup(s):
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

if __name__ == "__main__":
    write_source(remove_html_markup)


if __name__ == "__main__":
    import os
    os.system(f'git commit -m "Fourth version (clueless)" remove_html_markup.py')


def remove_html_markup(s):
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

if __name__ == "__main__":
    write_source(remove_html_markup)


if __name__ == "__main__":
    import os
    os.system(f'git commit -m "Fifth version (with assert)" remove_html_markup.py')


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
            assert False  # <=== Just added
            quote = not quote
        elif not tag:
            out = out + c

    return out

if __name__ == "__main__":
    write_source(remove_html_markup)


if __name__ == "__main__":
    import os
    os.system(f'git commit -m "Sixth version (with another assert)" remove_html_markup.py')


def remove_html_markup(s):
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

if __name__ == "__main__":
    write_source(remove_html_markup)


if __name__ == "__main__":
    import os
    os.system(f'git commit -m "Seventh version (fixed)" remove_html_markup.py')


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

if __name__ == "__main__":
    write_source(remove_html_markup)


if __name__ == "__main__":
    import os
    os.system(f'git commit -m "Eighth version (with proper assertion)" remove_html_markup.py')


if __package__ is None or __package__ == "":
    from ExpectError import ExpectError
else:
    from .ExpectError import ExpectError


if __name__ == "__main__":
    with ExpectError():
        assert remove_html_markup('"foo"') == '"foo"'


# ## Accessing Versions

if __name__ == "__main__":
    print('\n## Accessing Versions')




if __name__ == "__main__":
    import os
    os.system(f'git log --pretty=oneline')


import subprocess

def get_output(command):
    result = subprocess.run(command, 
                            stdout=subprocess.PIPE,
                            universal_newlines=True)
    return result.stdout

if __name__ == "__main__":
    log = get_output(['git', 'log', '--pretty=oneline'])
    print(log)


if __name__ == "__main__":
    versions = [line.split()[0] for line in log.split('\n') if line]
    versions.reverse()


if __name__ == "__main__":
    import os
    os.system(f'git checkout {versions[0]}')


if __name__ == "__main__":
    print_file('remove_html_markup.py')


if __name__ == "__main__":
    exec(open('remove_html_markup.py').read())


if __name__ == "__main__":
    remove_html_markup('"foo"')


if __name__ == "__main__":
    import os
    os.system(f'git checkout {versions[7]}')


if __name__ == "__main__":
    print_file('remove_html_markup.py')


if __name__ == "__main__":
    exec(open('remove_html_markup.py').read())


if __name__ == "__main__":
    remove_html_markup('"foo"')


# ## Manual Bisecting

if __name__ == "__main__":
    print('\n## Manual Bisecting')




if __name__ == "__main__":
    import os
    os.system(f'git bisect start')


if __name__ == "__main__":
    import os
    os.system(f'git bisect good {versions[0]}')


if __name__ == "__main__":
    import os
    os.system(f'git bisect bad {versions[7]}')


if __name__ == "__main__":
    print_file('remove_html_markup.py')


if __name__ == "__main__":
    exec(open('remove_html_markup.py').read())


if __name__ == "__main__":
    remove_html_markup('"foo"')


if __name__ == "__main__":
    import os
    os.system(f'git bisect bad')


if __name__ == "__main__":
    print_file('remove_html_markup.py')


if __name__ == "__main__":
    exec(open('remove_html_markup.py').read())


if __name__ == "__main__":
    remove_html_markup('"foo"')


if __name__ == "__main__":
    import os
    os.system(f'git bisect bad')


if __name__ == "__main__":
    print_file('remove_html_markup.py')


if __name__ == "__main__":
    exec(open('remove_html_markup.py').read())


if __name__ == "__main__":
    remove_html_markup('"foo"')


if __name__ == "__main__":
    import os
    os.system(f'git bisect bad')


if __name__ == "__main__":
    import os
    os.system(f'git diff HEAD^')


if __name__ == "__main__":
    import os
    os.system(f'git bisect view')


if __name__ == "__main__":
    import os
    os.system(f'git bisect reset')


# ## Automatic Bisecting

if __name__ == "__main__":
    print('\n## Automatic Bisecting')




if __name__ == "__main__":
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
    ''')
    print_file('test.py')


if __name__ == "__main__":
    import os
    os.system(f'python ./test.py; echo $?')


if __name__ == "__main__":
    import os
    os.system(f'git bisect start')


if __name__ == "__main__":
    import os
    os.system(f'git bisect good {versions[0]}')


if __name__ == "__main__":
    import os
    os.system(f'git bisect bad {versions[7]}')


if __name__ == "__main__":
    import os
    os.system(f'git bisect run python test.py')


if __name__ == "__main__":
    print_file('remove_html_markup.py')


if __name__ == "__main__":
    import os
    os.system(f'git diff HEAD^')


if __name__ == "__main__":
    import os
    os.system(f'git bisect reset')


# ## Delta Debugging on Changes

if __name__ == "__main__":
    print('\n## Delta Debugging on Changes')




if __package__ is None or __package__ == "":
    from DeltaDebugger import DeltaDebugger
else:
    from .DeltaDebugger import DeltaDebugger


if __name__ == "__main__":
    version_1 = get_output(['git', 'show', 
                                f'{versions[0]}:remove_html_markup.py'])


if __name__ == "__main__":
    print_content(version_1, '.py')


if __name__ == "__main__":
    version_2 = get_output(['git', 'show', 
                                f'{versions[1]}:remove_html_markup.py'])


if __name__ == "__main__":
    print_content(version_2, '.py')


if __name__ == "__main__":
    import os
    os.system(f'git diff {versions[0]} {versions[1]}')


from diff_match_patch import diff_match_patch

def diff(s1, s2, mode='lines'):
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

if __name__ == "__main__":
    patches = diff(version_1, version_2)
    patches


import urllib

def patch_string(p):
    return urllib.parse.unquote(str(p).strip())

if __name__ == "__main__":
    for p in patches:
        print(patch_string(p))


def patch(s, patches):
    dmp = diff_match_patch()
    text, success = dmp.patch_apply(patches, s)
    assert all(success)
    return text

if __name__ == "__main__":
    print_content(patch(version_1, patches), '.py')


if __name__ == "__main__":
    assert patch(version_1, patches) == version_2


if __name__ == "__main__":
    assert patch(version_1, []) == version_1


if __name__ == "__main__":
    print(patch_string(patches[0]))


if __name__ == "__main__":
    print_content(patch(version_1, [patches[0]]))


if __name__ == "__main__":
    print_content(patch(version_1, [patches[1]]))


def test_remove_html_markup(patches):
    new_version = patch(version_1, patches)
    exec(new_version, globals())
    assert remove_html_markup('"foo"') == '"foo"'

if __name__ == "__main__":
    test_remove_html_markup([])


if __name__ == "__main__":
    with ExpectError():
        test_remove_html_markup(patches)


if __name__ == "__main__":
    with DeltaDebugger() as dd:
        test_remove_html_markup(patches)


if __name__ == "__main__":
    reduced_patches = dd.min_args()['patches']


if __name__ == "__main__":
    for p in reduced_patches:
        print(urllib.parse.unquote(str(p)))


if __name__ == "__main__":
    print_content(patch(version_1, reduced_patches), '.py')


if __name__ == "__main__":
    pass_patches, fail_patches = (arg['patches'] for arg in dd.min_arg_diff())


if __name__ == "__main__":
    for p in pass_patches:
        print(urllib.parse.unquote(str(p)))


if __name__ == "__main__":
    for p in fail_patches:
        print(urllib.parse.unquote(str(p)))


# ## A ChangeDebugger class

if __name__ == "__main__":
    print('\n## A ChangeDebugger class')




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
    # More code
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
    # Some code that is part of the exercise
    pass


if __name__ == "__main__":
    # Some code for the solution
    2 + 2


# ### Exercise 2: _Title_

if __name__ == "__main__":
    print('\n### Exercise 2: _Title_')



