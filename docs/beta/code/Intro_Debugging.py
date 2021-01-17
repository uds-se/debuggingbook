#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Debugging Book".
# Web site: https://www.debuggingbook.org/html/Intro_Debugging.html
# Last change: 2021-01-17 15:28:20+01:00
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


# # Introduction to Debugging

if __name__ == "__main__":
    print('# Introduction to Debugging')




if __name__ == "__main__":
    from bookutils import YouTubeVideo, quiz
    YouTubeVideo("bCHRCehDOq0")


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




# ## A Simple Function

if __name__ == "__main__":
    print('\n## A Simple Function')




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

# ### Understanding Python Programs

if __name__ == "__main__":
    print('\n### Understanding Python Programs')




# ### Running a Function

if __name__ == "__main__":
    print('\n### Running a Function')




if __name__ == "__main__":
    remove_html_markup("Here's some <strong>strong argument</strong>.")


# ### Interacting with Notebooks

if __name__ == "__main__":
    print('\n### Interacting with Notebooks')




# ### Testing a Function

if __name__ == "__main__":
    print('\n### Testing a Function')




if __name__ == "__main__":
    assert remove_html_markup("Here's some <strong>strong argument</strong>.") == \
        "Here's some strong argument."


# ## Oops! A Bug!

if __name__ == "__main__":
    print('\n## Oops! A Bug!')




if __name__ == "__main__":
    remove_html_markup('<input type="text" value="<your name>">')


if __package__ is None or __package__ == "":
    from ExpectError import ExpectError
else:
    from .ExpectError import ExpectError


if __name__ == "__main__":
    with ExpectError():
        assert remove_html_markup('<input type="text" value="<your name>">') == ""


# ## Visualizing Code

if __name__ == "__main__":
    print('\n## Visualizing Code')




from graphviz import Digraph, nohtml

if __name__ == "__main__":
    from IPython.display import display


PASS = "✔"
FAIL = "✘"

PASS_COLOR = 'darkgreen'  # '#006400' # darkgreen
FAIL_COLOR = 'red4'  # '#8B0000' # darkred

STEP_COLOR = 'peachpuff'
FONT_NAME = 'Raleway'

def graph(comment="default"):
    return Digraph(name='', comment=comment, graph_attr={'rankdir': 'LR'},
        node_attr={'style': 'filled',
                   'fillcolor': STEP_COLOR,
                   'fontname': FONT_NAME},
        edge_attr={'fontname': FONT_NAME})

if __name__ == "__main__":
    state_machine = graph()
    state_machine.node('Start', )
    state_machine.edge('Start', '¬ tag')
    state_machine.edge('¬ tag', '¬ tag', label=" ¬ '<'\nadd character")
    state_machine.edge('¬ tag', 'tag', label="'<'")
    state_machine.edge('tag', '¬ tag', label="'>'")
    state_machine.edge('tag', 'tag', label="¬ '>'")


if __name__ == "__main__":
    display(state_machine)


# ## A First Fix

if __name__ == "__main__":
    print('\n## A First Fix')




if __name__ == "__main__":
    state_machine = graph()
    state_machine.node('Start')
    state_machine.edge('Start', '¬ quote\n¬ tag')
    state_machine.edge('¬ quote\n¬ tag', '¬ quote\n¬ tag',
                       label="¬ '<'\nadd character")
    state_machine.edge('¬ quote\n¬ tag', '¬ quote\ntag', label="'<'")
    state_machine.edge('¬ quote\ntag', 'quote\ntag', label="'\"'")
    state_machine.edge('¬ quote\ntag', '¬ quote\ntag', label="¬ '\"' ∧ ¬ '>'")
    state_machine.edge('quote\ntag', 'quote\ntag', label="¬ '\"'")
    state_machine.edge('quote\ntag', '¬ quote\ntag', label="'\"'")
    state_machine.edge('¬ quote\ntag', '¬ quote\n¬ tag', label="'>'")


if __name__ == "__main__":
    display(state_machine)


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
    remove_html_markup('<input type="text" value="<your name>">')


if __name__ == "__main__":
    assert remove_html_markup("Here's some <strong>strong argument</strong>.") == \
        "Here's some strong argument."


if __name__ == "__main__":
    assert remove_html_markup('<input type="text" value="<your name>">') == ""


if __name__ == "__main__":
    with ExpectError():
        assert remove_html_markup('<b>foo</b>') == 'foo'


if __name__ == "__main__":
    with ExpectError():
        assert remove_html_markup('<b>"foo"</b>') == '"foo"'


if __name__ == "__main__":
    with ExpectError():
        assert remove_html_markup('"<b>foo</b>"') == '"foo"'


if __name__ == "__main__":
    with ExpectError():
        assert remove_html_markup('<"b">foo</"b">') == 'foo'


# ## The Devil's Guide to Debugging

if __name__ == "__main__":
    print("\n## The Devil's Guide to Debugging")




# ### Printf Debugging

if __name__ == "__main__":
    print('\n### Printf Debugging')




def remove_html_markup_with_print(s):
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
    remove_html_markup_with_print('<b>"foo"</b>')


# ### Debugging into Existence

if __name__ == "__main__":
    print('\n### Debugging into Existence')




def remove_html_markup_without_quotes(s):
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
    assert remove_html_markup_without_quotes('<"b">foo</"b">') == 'foo'


if __name__ == "__main__":
    with ExpectError():
        assert remove_html_markup_without_quotes('<b>"foo"</b>') == '"foo"'


# ### Use the Most Obvious Fix

if __name__ == "__main__":
    print('\n### Use the Most Obvious Fix')




def remove_html_markup_fixed(s):
    if s == '<b>"foo"</b>':
        return '"foo"'
    ...

# ### Things to do Instead

if __name__ == "__main__":
    print('\n### Things to do Instead')




# ## From Defect to Failure

if __name__ == "__main__":
    print('\n## From Defect to Failure')




def execution_diagram(show_steps=True, variables=[],
                      steps=3, error_step=666,
                      until=666, fault_path=[]):
    dot = graph()

    dot.node('input', shape='none', fillcolor='white', label=f"Input {PASS}",
             fontcolor=PASS_COLOR)
    last_outgoing_states = ['input']

    for step in range(1, min(steps + 1, until)):

        if step == error_step:
            step_label = f'Step {step} {FAIL}'
            step_color = FAIL_COLOR
        else:
            step_label = f'Step {step}'
            step_color = None

        if step >= error_step:
            state_label = f'State {step} {FAIL}'
            state_color = FAIL_COLOR
        else:
            state_label = f'State {step} {PASS}'
            state_color = PASS_COLOR

        state_name = f's{step}'
        outgoing_states = []
        incoming_states = []

        if not variables:
            dot.node(name=state_name, shape='box',
                     label=state_label, color=state_color,
                     fontcolor=state_color)
        else:
            var_labels = []
            for v in variables:
                vpath = f's{step}:{v}'
                if vpath in fault_path:
                    var_label = f'<{v}>{v} ✘'
                    outgoing_states.append(vpath)
                    incoming_states.append(vpath)
                else:
                    var_label = f'<{v}>{v}'
                var_labels.append(var_label)
            record_string = " | ".join(var_labels)
            dot.node(name=state_name, shape='record',
                     label=nohtml(record_string), color=state_color,
                     fontcolor=state_color)

        if not outgoing_states:
            outgoing_states = [state_name]
        if not incoming_states:
            incoming_states = [state_name]

        for outgoing_state in last_outgoing_states:
            for incoming_state in incoming_states:
                if show_steps:
                    dot.edge(outgoing_state, incoming_state,
                             label=step_label, fontcolor=step_color)
                else:
                    dot.edge(outgoing_state, incoming_state)

        last_outgoing_states = outgoing_states

    if until > steps + 1:
        # Show output
        if error_step > steps:
            dot.node('output', shape='none', fillcolor='white',
                     label=f"Output {PASS}", fontcolor=PASS_COLOR)
        else:
            dot.node('output', shape='none', fillcolor='white',
                     label=f"Output {FAIL}", fontcolor=FAIL_COLOR)

        for outgoing_state in last_outgoing_states:
            label = "Execution" if steps == 0 else None
            dot.edge(outgoing_state, 'output', label=label)

    display(dot)

if __name__ == "__main__":
    execution_diagram(show_steps=False, steps=0, error_step=0)


if __name__ == "__main__":
    for until in range(1, 6):
        execution_diagram(show_steps=False, until=until, error_step=2)


if __name__ == "__main__":
    for until in range(1, 6):
        execution_diagram(show_steps=True, until=until, error_step=2)


if __name__ == "__main__":
    for until in range(1, 6):
        execution_diagram(show_steps=True, variables=['v1', 'v2', 'v3'],
                          error_step=2,
                          until=until, fault_path=['s2:v2', 's3:v2'])


# ## From Failure to Defect

if __name__ == "__main__":
    print('\n## From Failure to Defect')




# ## The Scientific Method

if __name__ == "__main__":
    print('\n## The Scientific Method')




if __name__ == "__main__":
    dot = graph()

    dot.node('Hypothesis')
    dot.node('Observation')
    dot.node('Prediction')
    dot.node('Experiment')

    dot.edge('Hypothesis', 'Observation',
             label="<Hypothesis<BR/>is <I>supported:</I><BR/>Refine it>",
             dir='back')
    dot.edge('Hypothesis', 'Prediction')

    dot.node('Problem Report', shape='none', fillcolor='white')
    dot.edge('Problem Report', 'Hypothesis')

    dot.node('Code', shape='none', fillcolor='white')
    dot.edge('Code', 'Hypothesis')

    dot.node('Runs', shape='none', fillcolor='white')
    dot.edge('Runs', 'Hypothesis')

    dot.node('More Runs', shape='none', fillcolor='white')
    dot.edge('More Runs', 'Hypothesis')

    dot.edge('Prediction', 'Experiment')
    dot.edge('Experiment', 'Observation')
    dot.edge('Observation', 'Hypothesis',
             label="<Hypothesis<BR/>is <I>rejected:</I><BR/>Seek alternative>")


if __name__ == "__main__":
    display(dot)


# ### Finding a Hypothesis

if __name__ == "__main__":
    print('\n### Finding a Hypothesis')




if __name__ == "__main__":
    for i, html in enumerate(['<b>foo</b>',
                              '<b>"foo"</b>',
                              '"<b>foo</b>"',
                              '<"b">foo</"b">']):
        result = remove_html_markup(html)
        print("%-2d %-15s %s" % (i + 1, html, result))


if __name__ == "__main__":
    quiz("From the difference between success and failure,"
         " we can already devise some observations about "
         " what's wrong with the output."
         " Which of these can we turn into general hypotheses?",
        ["Double quotes are stripped from the tagged input.",
         "Tags in double quotes are not stripped.",
         "The tag '&lt;b&gt;' is always stripped from the input.",
         "Four-letter words are stripped."], [298 % 33, 1234 % 616])


# ### Testing a Hypothesis

if __name__ == "__main__":
    print('\n### Testing a Hypothesis')




if __name__ == "__main__":
    remove_html_markup('"foo"')


# ### Refining a Hypothesis

if __name__ == "__main__":
    print('\n### Refining a Hypothesis')




def remove_html_markup_with_tag_assert(s):
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
    quiz("What happens after inserting the above assertion?",
        ["The program raises an exception. (i.e., tag is set)",
         "The output is as before, i.e., foo without quotes."
         " (which means that tag is not set)"],
         2)


if __name__ == "__main__":
    with ExpectError():
        result = remove_html_markup_with_tag_assert('"foo"')
    result


# ### Refuting a Hypothesis

if __name__ == "__main__":
    print('\n### Refuting a Hypothesis')




def remove_html_markup_with_quote_assert(s):
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
    quiz("What happens after inserting the 'assert' tag?",
    ["The program raises an exception (i.e., the quote condition holds)",
     "The output is still foo (i.e., the quote condition does not hold)"], 29 % 7)


if __name__ == "__main__":
    with ExpectError():
        result = remove_html_markup_with_quote_assert('"foo"')


if __name__ == "__main__":
    remove_html_markup("'foo'")


if __name__ == "__main__":
    quiz("How should the condition read?",
         ["Choice 1", "Choice 2", "Choice 3", "Something else"],
         399 % 4)


# ## Fixing the Bug

if __name__ == "__main__":
    print('\n## Fixing the Bug')




# ### Checking Diagnoses

if __name__ == "__main__":
    print('\n### Checking Diagnoses')




# ### Fixing the Code

if __name__ == "__main__":
    print('\n### Fixing the Code')




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
    assert remove_html_markup("Here's some <strong>strong argument</strong>.") == \
        "Here's some strong argument."
    assert remove_html_markup(
        '<input type="text" value="<your name>">') == ""
    assert remove_html_markup('<b>foo</b>') == 'foo'
    assert remove_html_markup('<b>"foo"</b>') == '"foo"'
    assert remove_html_markup('"<b>foo</b>"') == '"foo"'
    assert remove_html_markup('<"b">foo</"b">') == 'foo'


# ### Alternate Paths

if __name__ == "__main__":
    print('\n### Alternate Paths')




# ## Homework after the Fix

if __name__ == "__main__":
    print('\n## Homework after the Fix')




# ### Check for further Defect Occurrences

if __name__ == "__main__":
    print('\n### Check for further Defect Occurrences')




# ### Check your Tests

if __name__ == "__main__":
    print('\n### Check your Tests')




# ### Add Assertions

if __name__ == "__main__":
    print('\n### Add Assertions')




if __name__ == "__main__":
    quiz("Which assertion would have caught the problem?",
        ["assert quote and not tag",
         "assert quote or not tag",
         "assert tag or not quote",
         "assert tag and not quote"],
        3270 - 3267)


if __name__ == "__main__":
    display(state_machine)


def remove_html_markup(s):
    tag = False
    quote = False
    out = ""

    for c in s:
        assert tag or not quote

        if c == '<' and not quote:
            tag = True
        elif c == '>' and not quote:
            tag = False
        elif (c == '"' or c == "'") and tag:
            quote = not quote
        elif not tag:
            out = out + c

    return out

# ### Commit the Fix

if __name__ == "__main__":
    print('\n### Commit the Fix')




# ### Close the Bug Report

if __name__ == "__main__":
    print('\n### Close the Bug Report')




# ## Become a Better Debugger

if __name__ == "__main__":
    print('\n## Become a Better Debugger')




# ### Keep a Log

if __name__ == "__main__":
    print('\n### Keep a Log')




# ### Rubberducking

if __name__ == "__main__":
    print('\n### Rubberducking')




# ## The Cost of Debugging

if __name__ == "__main__":
    print('\n## The Cost of Debugging')




# ## History of Debugging

if __name__ == "__main__":
    print('\n## History of Debugging')




import hashlib

bughash = hashlib.md5(b"debug").hexdigest()

if __name__ == "__main__":
    quiz('Where has the name "bug" been used to denote disruptive events?',
         [
            'In the early days of Morse telegraphy, referring to a special key '
              'that would send a string of dots',
            'Among radio technicians to describe a device that '
              'converts electromagnetic field variations into acoustic signals',
            "In Shakespeare's " '"Henry VI", referring to a walking spectre',
            'In Middle English, where the word "bugge" is the basis for terms '
              'like "bugbear" and "bugaboo"'
         ],
        [bughash.index(i) for i in "d42f"]
        )


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




# ### Exercise 1: Get Acquainted with Notebooks and Python

if __name__ == "__main__":
    print('\n### Exercise 1: Get Acquainted with Notebooks and Python')




# #### Beginner Level: Run Notebooks in Your Browser

if __name__ == "__main__":
    print('\n#### Beginner Level: Run Notebooks in Your Browser')




# #### Advanced Level: Run Python Code on Your Machine

if __name__ == "__main__":
    print('\n#### Advanced Level: Run Python Code on Your Machine')




# #### Pro Level: Run Notebooks on Your Machine

if __name__ == "__main__":
    print('\n#### Pro Level: Run Notebooks on Your Machine')




# #### Boss Level: Contribute!

if __name__ == "__main__":
    print('\n#### Boss Level: Contribute!')




# ### Exercise 2: More Bugs!

if __name__ == "__main__":
    print('\n### Exercise 2: More Bugs!')




# #### Part 1: Find the Problem

if __name__ == "__main__":
    print('\n#### Part 1: Find the Problem')




if __name__ == "__main__":
    assert(...)


if __name__ == "__main__":
    s = '<b title="<Shakespeare' + "'s play>" + '">foo</b>'
    s


if __name__ == "__main__":
    remove_html_markup(s)


if __name__ == "__main__":
    with ExpectError():
        assert(remove_html_markup(s) == "foo")


# #### Part 2: Identify Extent and Cause

if __name__ == "__main__":
    print('\n#### Part 2: Identify Extent and Cause')




# #### Part 3: Fix the Problem

if __name__ == "__main__":
    print('\n#### Part 3: Fix the Problem')




def remove_html_markup_with_proper_quotes(s):
    tag = False
    quote = ''
    out = ""

    for c in s:
        assert tag or quote == ''

        if c == '<' and quote == '':
            tag = True
        elif c == '>' and quote == '':
            tag = False
        elif (c == '"' or c == "'") and tag and quote == '':
            # beginning of string
            quote = c
        elif c == quote:
            # end of string
            quote = ''
        elif not tag:
            out = out + c

    return out

if __name__ == "__main__":
    assert(remove_html_markup_with_proper_quotes(s) == "foo")


if __name__ == "__main__":
    assert remove_html_markup_with_proper_quotes(
        "Here's some <strong>strong argument</strong>.") == \
        "Here's some strong argument."
    assert remove_html_markup_with_proper_quotes(
        '<input type="text" value="<your name>">') == ""
    assert remove_html_markup_with_proper_quotes('<b>foo</b>') == 'foo'
    assert remove_html_markup_with_proper_quotes('<b>"foo"</b>') == '"foo"'
    assert remove_html_markup_with_proper_quotes('"<b>foo</b>"') == '"foo"'
    assert remove_html_markup_with_proper_quotes('<"b">foo</"b">') == 'foo'

