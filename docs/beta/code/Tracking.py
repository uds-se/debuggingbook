#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Debugging Book".
# Web site: https://www.debuggingbook.org/html/Tracking.html
# Last change: 2021-01-23 19:57:27+01:00
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


# # Tracking Bugs

if __name__ == "__main__":
    print('# Tracking Bugs')




if __name__ == "__main__":
    from bookutils import YouTubeVideo
    # YouTubeVideo("w4u5gCgPlmg")


if __name__ == "__main__":
    # We use the same fixed seed as the notebook to ensure consistency
    import random
    random.seed(2001)


if __package__ is None or __package__ == "":
    import Intro_Debugging
else:
    from . import Intro_Debugging


import os

if __name__ == "__main__":
    assert os.getenv('USER') == 'zeller'


# ## Reporting Issues

if __name__ == "__main__":
    print('\n## Reporting Issues')




# ### What Goes in a Bug Report?

if __name__ == "__main__":
    print('\n### What Goes in a Bug Report?')




# #### Steps to Reproduce (83%)

if __name__ == "__main__":
    print('\n#### Steps to Reproduce (83%)')




# #### Stack Traces (57%)

if __name__ == "__main__":
    print('\n#### Stack Traces (57%)')




if __package__ is None or __package__ == "":
    from ExpectError import ExpectError
else:
    from .ExpectError import ExpectError


def handle_command(s):
    scope = s.index(" in ")

if __name__ == "__main__":
    with ExpectError():
        handle_command("run")


# #### Test Cases (51%)

if __name__ == "__main__":
    print('\n#### Test Cases (51%)')




# #### Observed Behavior (33%)

if __name__ == "__main__":
    print('\n#### Observed Behavior (33%)')




# #### Screenshots (26%)

if __name__ == "__main__":
    print('\n#### Screenshots (26%)')




# #### Expected Behavior (22%)

if __name__ == "__main__":
    print('\n#### Expected Behavior (22%)')




# #### Configuration Information (< 12%)

if __name__ == "__main__":
    print('\n#### Configuration Information (< 12%)')




# ### Reporting Crashes Automatically

if __name__ == "__main__":
    print('\n### Reporting Crashes Automatically')




# ## An Issue Tracker

if __name__ == "__main__":
    print('\n## An Issue Tracker')




# ### Excursion: Setting up Redmine

if __name__ == "__main__":
    print('\n### Excursion: Setting up Redmine')




import subprocess

import os
import sys

def with_ruby(cmd, inp='', timeout=10, show_stdout=False):
    print(f"$ {cmd}")
    shell = subprocess.Popen(['/bin/sh', '-c',
        f'''rvm_redmine=$HOME/.rvm/gems/ruby-2.7.2@redmine; \
rvm_global=$HOME/.rvm/gems/ruby-2.7.2@global; \
export GEM_PATH=$rvm_redmine:$rvm_global; \
export PATH=$rvm_redmine/bin:$rvm_global/bin:$HOME/.rvm/rubies/ruby-2.7.2/bin:$HOME/.rvm/bin:$PATH; \
cd $HOME/lib/redmine && {cmd}'''],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
    try:
        stdout_data, stderr_data = shell.communicate(inp, timeout=timeout)
    except subprocess.TimeoutExpired:
        shell.kill()
#         stdout_data, stderr_data = shell.communicate(inp)
#         if show_stdout:
#             print(stdout_data, end="")
#         print(stderr_data, file=sys.stderr, end="")
        raise

    print(stderr_data, file=sys.stderr, end="")
    if show_stdout:
        print(stdout_data, end="")

def with_mysql(cmd, timeout=2, show_stdout=False):
    print(f"sql>{cmd}")
    sql = subprocess.Popen(["mysql", "-u", "root",
                           "--default-character-set=utf8mb4"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, 
                            universal_newlines=True)
    try:
        stdout_data, stderr_data = sql.communicate(cmd + ';', 
                                                   timeout=timeout)
    except suprocess.TimeoutExpired:
        sql.kill()
#         stdout_data, stderr_data = sql.communicate(inp)
#         if show_stdout:
#             print(stdout_data, end="")
#         print(stderr_data, file=sys.stderr, end="")
        raise

    print(stderr_data, file=sys.stderr, end="")
    if show_stdout:
        print(stdout_data, end="")

if __name__ == "__main__":
    with_ruby("bundle config set without development test")


if __name__ == "__main__":
    with_ruby("bundle install")


if __name__ == "__main__":
    with_ruby("pkill sql; sleep 5")


if __name__ == "__main__":
    try:
        with_ruby("mysql.server start", show_stdout=True)
    except subprocess.TimeoutExpired:
        pass  # Can actually start without producing output


if __name__ == "__main__":
    with_mysql("drop database redmine")


if __name__ == "__main__":
    with_mysql("drop user 'redmine'@'localhost'")


if __name__ == "__main__":
    with_mysql("create database redmine character set utf8")


if __name__ == "__main__":
    with_mysql("create user 'redmine'@'localhost' identified by 'my_password'")


if __name__ == "__main__":
    with_mysql("grant all privileges on redmine.* to 'redmine'@'localhost'")


if __name__ == "__main__":
    with_ruby("bundle exec rake generate_secret_token")


if __name__ == "__main__":
    with_ruby("RAILS_ENV=production bundle exec rake db:migrate")


if __name__ == "__main__":
    with_ruby("RAILS_ENV=production bundle exec rake redmine:load_default_data", '\n')


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ### Excursion: Starting Redmine

if __name__ == "__main__":
    print('\n### Excursion: Starting Redmine')




import os
import time

from multiprocessing import Process

def run_redmine(port):
    with_ruby(f'exec rails s -e production -p {port} > redmine.log 2>&1',
             timeout=3600)

def start_redmine(port=3000):
    process = Process(target=run_redmine, args=(port,))
    process.start()
    time.sleep(5)

    url = f"http://localhost:{port}"
    return process, url

if __name__ == "__main__":
    redmine_process, redmine_url = start_redmine()


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ### Excursion: Remote Control with Selenium

if __name__ == "__main__":
    print('\n### Excursion: Remote Control with Selenium')




from selenium import webdriver

from selenium.webdriver.common.keys import Keys

BROWSER = 'firefox'

if __name__ == "__main__":
    with_ruby("pkill Firefox.app firefox-bin")


if __package__ is None or __package__ == "":
    from bookutils import rich_output
else:
    from .bookutils import rich_output


HEADLESS = False

def start_webdriver(browser=BROWSER, headless=HEADLESS, zoom=1.4):
    if browser == 'firefox':
        options = webdriver.FirefoxOptions()
    if browser == 'chrome':
        options = webdriver.ChromeOptions()

    if headless and browser == 'chrome':
        options.add_argument('headless')
    else:
        options.headless = headless

    # Start the browser, and obtain a _web driver_ object such that we can interact with it.
    if browser == 'firefox':
        # For firefox, set a higher resolution for our screenshots
        profile = webdriver.firefox.firefox_profile.FirefoxProfile()
        profile.set_preference("layout.css.devPixelsPerPx", repr(zoom))
        redmine_gui = webdriver.Firefox(firefox_profile=profile, options=options)

        # We set the window size such that it fits
        redmine_gui.set_window_size(500, 600)  # was 1024, 600

    elif browser == 'chrome':
        redmine_gui = webdriver.Chrome(options=options)
        redmine_gui.set_window_size(1024, 510 if headless else 640)

    return redmine_gui

if __name__ == "__main__":
    redmine_gui = start_webdriver(browser=BROWSER, headless=HEADLESS)


if __name__ == "__main__":
    redmine_gui.get(redmine_url)


if __name__ == "__main__":
    from IPython.display import display, Image


if __name__ == "__main__":
    Image(redmine_gui.get_screenshot_as_png())


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ### Excursion: Screenshots with Drop Shadows

if __name__ == "__main__":
    print('\n### Excursion: Screenshots with Drop Shadows')




import tempfile

def drop_shadow(contents):
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(contents)
        convert = subprocess.Popen(
            ['convert', tmp.name,
            '(', '+clone', '-background', 'black', '-shadow', '50x10+15+15', ')',
            '+swap', '-background', 'none', '-layers', 'merge', '+repage', '-'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_data, stderr_data = convert.communicate()
    
    if stderr_data:
        print(stderr_data.decode("utf-8"), file=sys.stderr, end="")
        
    return stdout_data

def resize(contents, size):
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(contents)
        convert = subprocess.Popen(
            ['convert', tmp.name, '-resize', size, '-'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_data, stderr_data = convert.communicate()
    
    if stderr_data:
        print(stderr_data.decode("utf-8"), file=sys.stderr, end="")
        
    return stdout_data

def screenshot(driver):
    return Image(resize(drop_shadow(redmine_gui.get_screenshot_as_png()), "50%"))

if __name__ == "__main__":
    screenshot(redmine_gui)


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ### Excursion: First Registration at Redmine

if __name__ == "__main__":
    print('\n### Excursion: First Registration at Redmine')




if __name__ == "__main__":
    redmine_gui.get(redmine_url + '/login')


if __name__ == "__main__":
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.find_element_by_id("username").send_keys("admin")
    redmine_gui.find_element_by_id("password").send_keys("admin")
    redmine_gui.find_element_by_name("login").click()


if __name__ == "__main__":
    time.sleep(2)


if __name__ == "__main__":
    if redmine_gui.current_url.endswith('my/password'):
        redmine_gui.get(redmine_url + '/my/password')
        redmine_gui.find_element_by_id("password").send_keys("admin")
        redmine_gui.find_element_by_id("new_password").send_keys("admin001")
        redmine_gui.find_element_by_id("new_password_confirmation").send_keys("admin001")
        display(screenshot(redmine_gui))
        redmine_gui.find_element_by_name("commit").click()


if __name__ == "__main__":
    redmine_gui.get(redmine_url + '/logout')
    redmine_gui.find_element_by_name("commit").click()


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




if __name__ == "__main__":
    redmine_gui.get(redmine_url + '/login')
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.find_element_by_id("username").send_keys("admin")
    redmine_gui.find_element_by_id("password").send_keys("admin001")
    redmine_gui.find_element_by_name("login").click()
    screenshot(redmine_gui)


# ### Excursion: Creating a Project

if __name__ == "__main__":
    print('\n### Excursion: Creating a Project')




if __name__ == "__main__":
    redmine_gui.get(redmine_url + '/projects')
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.get(redmine_url + '/projects/new')
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.get(redmine_url + '/projects/new')
    redmine_gui.find_element_by_id('project_name').send_keys("The Debugging Book")
    redmine_gui.find_element_by_id('project_description').send_keys("A Book on Automated Debugging")
    redmine_gui.find_element_by_id('project_identifier').clear()
    redmine_gui.find_element_by_id('project_identifier').send_keys("debuggingbook")
    redmine_gui.find_element_by_id('project_homepage').send_keys("https://www.debuggingbook.org/")
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.find_element_by_name('commit').click()


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




if __name__ == "__main__":
    redmine_gui.get(redmine_url + '/projects')
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.get(redmine_url + '/projects/debuggingbook')
    screenshot(redmine_gui)


# ## Reporting an Issue

if __name__ == "__main__":
    print('\n## Reporting an Issue')




if __name__ == "__main__":
    redmine_gui.get(redmine_url + '/issues/new')
    screenshot(redmine_gui)


if __name__ == "__main__":
    issue_title = "Does not render correctly on Nokia Communicator"


if __name__ == "__main__":
    issue_description = \
    """The Debugging Book does not render correctly on the Nokia Communicator 9000.

Steps to reproduce:
1. On the Nokia, go to "https://debuggingbook.org/"
2. From the menu on top, select the chapter "Tracking Origins".
3. Scroll down to a place where a graph is supposed to be shown.
4. Instead of the graph, only a blank space is displayed.

How to fix:
* The graphs seem to come as SVG elements, but the Nokia Communicator does not support SVG rendering. Render them as JPEGs instead.
"""


if __name__ == "__main__":
    redmine_gui.get(redmine_url + '/issues/new')

    redmine_gui.find_element_by_id('issue_subject').send_keys(issue_title)
    redmine_gui.find_element_by_id('issue_description').send_keys(issue_description)
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.find_element_by_id('issue_assigned_to_id').click()
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.find_element_by_name('commit').click()
    screenshot(redmine_gui)


# ### Excursion: Adding Some More Issue Reports

if __name__ == "__main__":
    print('\n### Excursion: Adding Some More Issue Reports')




def new_issue(issue_title, issue_description):
    redmine_gui.get(redmine_url + '/issues/new')

    redmine_gui.find_element_by_id('issue_subject').send_keys(issue_title)
    redmine_gui.find_element_by_id('issue_description').send_keys(issue_description)
    redmine_gui.find_element_by_name('commit').click()
    return screenshot(redmine_gui)

if __name__ == "__main__":
    new_issue("Missing a Chapter on Parallel Debugging",
    """I am missing a chapter on (automatic) debugging of parallel and distributed systems,
including how to detect and repair data races, log message passing, and more.
In my experience, almost all programs are parallel today, so you are missing
an important subject.
""")


if __name__ == "__main__":
    new_issue("Missing a PDF version",
    """Your 'book' does not provide a printed version. I think that printed books

* offer a more enjoyable experience for the reader
* allow me to annotate pages with my own remarks
* allow me to set dog-ear bookmatks
* allow me to show off what I'm currently reading (do you have a cover, too?)

Please provide a printed version - or, at least, produce a PDF version
of the debugging book, and make it available for download, such that I can print it myself.
""")


if __name__ == "__main__":
    new_issue("No PDF version",
    """Can I have a printed version of your book? Please!""")


if __name__ == "__main__":
    new_issue("Does not work with Python 2.7 or earlier",
    """I was deeply disappointed that your hew book requires Python 3.6 or later.
There are still several Python 2.x users out here (I, for one, cannot stand having to
type parentheses for every `print` statement), and I would love to run your code on
my Python 2.7 programs.

Would it be possible to backport the book's code such that it would run on Python 3.x
as well as Python 2.x? I would suggest that you add simple checks around your code
such as the following:

```
import sys

if sys.version_info.major >= 3:
    print("The result is", x)
else: 
    print "The result is", x
```

As an alternative, rewrite the book in Python 2 and have it automatically translate to
Python 3. This way, you could address all Python lovers, not just Python 3 ones.
""")


if __name__ == "__main__":
    new_issue("Support for C++",
    """I had lots of fun with your 'debugging book'. Yet, I was somewhat disappointed
to see that all code examples are in and for Python programs only. Is there a chance
to get them to work on a real programming language such as C or C++? This would also
open the way to discuss several new debugging techniques for bugs that occur in these
languages only. A chapter on C++ move semantics, and how to fix them, for instance,
would be highly appreciated.
""")


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ## Managing Issues

if __name__ == "__main__":
    print('\n## Managing Issues')




if __name__ == "__main__":
    redmine_gui.get(redmine_url + "/projects/debuggingbook")
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.get(redmine_url + '/projects/debuggingbook/issues')
    redmine_gui.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.get(redmine_url + "/issues/")


if __name__ == "__main__":
    redmine_gui.find_element_by_xpath("//tr[@id='issue-2']//a[@title='Actions']").click()
    time.sleep(0.25)


if __name__ == "__main__":
    tracker_item = redmine_gui.find_element_by_xpath(
        "//div[@id='context-menu']//a[text()='Tracker']")
    actions = webdriver.ActionChains(redmine_gui)
    actions.move_to_element(tracker_item)
    actions.perform()
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.find_element_by_xpath("//div[@id='context-menu']//a[text()='Feature']").click()


def mark_tracker(issue, tracker):
    redmine_gui.get(redmine_url + "/issues/")
    redmine_gui.find_element_by_xpath(
        f"//tr[@id='issue-{str(issue)}']//a[@title='Actions']").click()
    time.sleep(0.25)
    
    tracker_item = redmine_gui.find_element_by_xpath(
        "//div[@id='context-menu']//a[text()='Tracker']")
    actions = webdriver.ActionChains(redmine_gui)
    actions.move_to_element(tracker_item)
    actions.perform()
    time.sleep(0.25)
    
    redmine_gui.find_element_by_xpath(
        f"//div[@id='context-menu']//a[text()='{tracker}']").click()

if __name__ == "__main__":
    mark_tracker(3, "Feature")
    mark_tracker(4, "Feature")
    mark_tracker(6, "Feature")


if __name__ == "__main__":
    redmine_gui.get(redmine_url + "/issues/")
    redmine_gui.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    screenshot(redmine_gui)


# ## Assigning Priorities

if __name__ == "__main__":
    print('\n## Assigning Priorities')




if __name__ == "__main__":
    redmine_gui.get(redmine_url + "/issues/")


if __name__ == "__main__":
    redmine_gui.find_element_by_xpath("//tr[@id='issue-1']//a[@title='Actions']").click()
    time.sleep(0.25)


if __name__ == "__main__":
    priority_item = redmine_gui.find_element_by_xpath("//div[@id='context-menu']//a[text()='Priority']")
    actions = webdriver.ActionChains(redmine_gui)
    actions.move_to_element(priority_item)
    actions.perform()
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.find_element_by_xpath("//div[@id='context-menu']//a[text()='Urgent']").click()


if __name__ == "__main__":
    redmine_gui.get(redmine_url + "/issues/")
    redmine_gui.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    screenshot(redmine_gui)


# ## Assigning Issues

if __name__ == "__main__":
    print('\n## Assigning Issues')




if __name__ == "__main__":
    redmine_gui.get(redmine_url + "/issues/")


if __name__ == "__main__":
    redmine_gui.find_element_by_xpath("//tr[@id='issue-1']//a[@title='Actions']").click()
    time.sleep(0.25)


if __name__ == "__main__":
    assignee_item = redmine_gui.find_element_by_xpath(
        "//div[@id='context-menu']//a[text()='Assignee']")
    actions = webdriver.ActionChains(redmine_gui)
    actions.move_to_element(assignee_item)
    actions.perform()
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.find_element_by_xpath("//div[@id='context-menu']//a[text()='<< me >>']").click()
    screenshot(redmine_gui)


# ## Resolving Issues

if __name__ == "__main__":
    print('\n## Resolving Issues')




if __name__ == "__main__":
    redmine_gui.get(redmine_url + "/projects/debuggingbook/issues?query_id=1")
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.get(redmine_url + "/issues/1")
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.get(redmine_url + "/issues/1/edit")
    redmine_gui.find_element_by_id("issue_status_id").click()


if __name__ == "__main__":
    redmine_gui.find_element_by_xpath("//option[text()='Resolved']").click()
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    issue_notes = redmine_gui.find_element_by_id("issue_notes")
    issue_notes.send_keys("Will only work for Nokia Communicator Rev B and later; "
        "Rev A is still unsupported")
    screenshot(redmine_gui)


if __name__ == "__main__":
    redmine_gui.find_element_by_name("commit").click()
    screenshot(redmine_gui)


# ## The Life Cycle of an Issue

if __name__ == "__main__":
    print('\n## The Life Cycle of an Issue')




if __package__ is None or __package__ == "":
    from Intro_Debugging import graph
else:
    from .Intro_Debugging import graph


if __name__ == "__main__":
    from IPython.display import display


if __name__ == "__main__":
    life_cycle = graph()
    life_cycle.attr(rankdir='TB')

    life_cycle.node('New', label="<<b>NEW</b>>", penwidth='2.0')
    life_cycle.node('Assigned', label="<<b>ASSIGNED</b>>")

    with life_cycle.subgraph() as res:
        res.attr(rank='same')
        res.node('Resolved', label="<<b>RESOLVED</b>>", penwidth='2.0')
        res.node('Resolution',
                    shape='plain',
                    fillcolor='white',
                    label="""<<b>Resolution:</b> One of<br align="left"/>
• FIXED<br align="left"/>
• INVALID<br align="left"/>
• DUPLICATE<br align="left"/>
• WONTFIX<br align="left"/>
• WORKSFORME<br align="left"/>
>""")
        res.node('Reopened', label="<<b>REOPENED</b>>", style='invis')

    life_cycle.edge('New', 'Assigned', label=r"Assigned\lto developer")
    life_cycle.edge('Assigned', 'Resolved', label="Developer has fixed bug")

    life_cycle.edge('Resolution', 'Resolved', arrowhead='none', style='dashed')

    life_cycle


if __name__ == "__main__":
    life_cycle.node('Unconfirmed', label="<<b>UNCONFIRMED</b>>", penwidth='2.0')
    # life_cycle.node('Verified', label="<<b>VERIFIED</b>>")
    life_cycle.node('Closed', label="<<b>CLOSED</b>>", penwidth='2.0')
    life_cycle.node('Reopened', label="<<b>REOPENED</b>>", style='filled')
    life_cycle.node('New', label="<<b>NEW</b>>", penwidth='1.0')

    life_cycle.edge('Unconfirmed', 'New', label="Confirmed as \"new\"")
    life_cycle.edge('Assigned', 'New', label="Unassigned")
    life_cycle.edge('Resolved', 'Closed', label=r"Quality Assurance\lconfirms fix")
    life_cycle.edge('Resolved', 'Reopened', label=r"Quality Assurance\lnot satisfied")
    life_cycle.edge('Reopened', 'Assigned', label=r"Assigned\lto developer")
    # life_cycle.edge('Verified', 'Closed', label="Bug is closed")
    life_cycle.edge('Closed', 'Reopened', label=r"Bug is\lreopened")

    life_cycle


# ## Cleanup

if __name__ == "__main__":
    print('\n## Cleanup')




import os

if __name__ == "__main__":
    redmine_process.terminate()


if __name__ == "__main__":
    redmine_gui.close()


if __name__ == "__main__":
    os.system("pkill ruby")


# ## Synopsis

if __name__ == "__main__":
    print('\n## Synopsis')




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
    pass


if __name__ == "__main__":
    2 + 2


# ### Exercise 2: _Title_

if __name__ == "__main__":
    print('\n### Exercise 2: _Title_')



