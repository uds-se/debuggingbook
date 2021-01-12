#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This material is part of "The Debugging Book".
# Web site: https://www.debuggingbook.org/html/Tracking.html
# Last change: 2021-01-12 16:23:22+01:00
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


# ## A Change Tracker

if __name__ == "__main__":
    print('\n## A Change Tracker')




# ### Excursion: Setting up Redmine

if __name__ == "__main__":
    print('\n### Excursion: Setting up Redmine')




import subprocess

import os
import sys

def with_ruby(cmd, inp='', show_stdout=False):
    shell = subprocess.Popen(['/bin/sh', '-c', f'''rvm_redmine=$HOME/.rvm/gems/ruby-2.7.2@redmine; \
rvm_global=$HOME/.rvm/gems/ruby-2.7.2@global; \
export GEM_PATH=$rvm_redmine:$rvm_global; \
export PATH=$rvm_redmine/bin:$rvm_global/bin:$HOME/.rvm/rubies/ruby-2.7.2/bin:$HOME/.rvm/bin:$PATH; \
cd $HOME/lib/redmine && {cmd}'''],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
    stdout_data, stderr_data = shell.communicate(inp)
    print(stderr_data, file=sys.stderr, end="")
    if show_stdout:
        print(stdout_data, end="")

def with_mysql(cmd, show_stdout=False):
    sql = subprocess.Popen(["mysql", "-u", "root",
                           "--default-character-set=utf8mb4"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, 
                            universal_newlines=True)
    stdout_data, stderr_data = sql.communicate(cmd + ';')
    print(stderr_data, file=sys.stderr, end="")
    if show_stdout:
        print(stdout_data, end="")

if __name__ == "__main__":
    with_ruby("bundle install --without development test")


if __name__ == "__main__":
    with_ruby("mysql.server start")


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
    with_ruby(f'exec rails s -e production -p {port} > redmine.log 2>&1')

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

if __package__ is None or __package__ == "":
    from bookutils import rich_output
else:
    from .bookutils import rich_output


HEADLESS = not rich_output()

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
        gui_driver = webdriver.Firefox(firefox_profile=profile, options=options)

        # We set the window size such that it fits
        gui_driver.set_window_size(500, 600)  # was 1024, 600

    elif browser == 'chrome':
        gui_driver = webdriver.Chrome(options=options)
        gui_driver.set_window_size(1024, 510 if headless else 640)

    return gui_driver

if __name__ == "__main__":
    gui_driver = start_webdriver(browser=BROWSER, headless=HEADLESS)


if __name__ == "__main__":
    gui_driver.get(redmine_url)


if __name__ == "__main__":
    from IPython.display import display, Image


if __name__ == "__main__":
    Image(gui_driver.get_screenshot_as_png())


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

def screenshot(driver):
    return Image(drop_shadow(gui_driver.get_screenshot_as_png()))

if __name__ == "__main__":
    screenshot(gui_driver)


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




# ### Excursion: First Registration at Redmine

if __name__ == "__main__":
    print('\n### Excursion: First Registration at Redmine')




if __name__ == "__main__":
    gui_driver.get(redmine_url + '/login')


if __name__ == "__main__":
    screenshot(gui_driver)


if __name__ == "__main__":
    gui_driver.find_element_by_id("username").send_keys("admin")
    gui_driver.find_element_by_id("password").send_keys("admin")
    gui_driver.find_element_by_name("login").click()


if __name__ == "__main__":
    time.sleep(2)


if __name__ == "__main__":
    if gui_driver.current_url.endswith('my/password'):
        gui_driver.get(redmine_url + '/my/password')
        gui_driver.find_element_by_id("password").send_keys("admin")
        gui_driver.find_element_by_id("new_password").send_keys("admin001")
        gui_driver.find_element_by_id("new_password_confirmation").send_keys("admin001")
        display(screenshot(gui_driver))
        gui_driver.find_element_by_name("commit").click()


if __name__ == "__main__":
    gui_driver.get(redmine_url + '/logout')
    gui_driver.find_element_by_name("commit").click()


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




if __name__ == "__main__":
    gui_driver.get(redmine_url + '/login')
    screenshot(gui_driver)


if __name__ == "__main__":
    gui_driver.find_element_by_id("username").send_keys("admin")
    gui_driver.find_element_by_id("password").send_keys("admin001")
    gui_driver.find_element_by_name("login").click()
    screenshot(gui_driver)


# ### Excursion: Creating a Project

if __name__ == "__main__":
    print('\n### Excursion: Creating a Project')




if __name__ == "__main__":
    gui_driver.get(redmine_url + '/projects')
    screenshot(gui_driver)


if __name__ == "__main__":
    gui_driver.get(redmine_url + '/projects/new')
    screenshot(gui_driver)


if __name__ == "__main__":
    gui_driver.get(redmine_url + '/projects/new')
    gui_driver.find_element_by_id('project_name').send_keys("The Debugging Book")
    gui_driver.find_element_by_id('project_description').send_keys("A Book on Automated Debugging")
    gui_driver.find_element_by_id('project_identifier').clear()
    gui_driver.find_element_by_id('project_identifier').send_keys("debuggingbook")
    gui_driver.find_element_by_id('project_homepage').send_keys("https://www.debuggingbook.org/")
    screenshot(gui_driver)


if __name__ == "__main__":
    gui_driver.find_element_by_name('commit').click()


# ### End of Excursion

if __name__ == "__main__":
    print('\n### End of Excursion')




if __name__ == "__main__":
    gui_driver.get(redmine_url + '/projects')
    screenshot(gui_driver)


if __name__ == "__main__":
    gui_driver.get(redmine_url + '/projects/debuggingbook')
    screenshot(gui_driver)


# ## Reporting a Bug

if __name__ == "__main__":
    print('\n## Reporting a Bug')




if __name__ == "__main__":
    gui_driver.get(redmine_url + '/issues/new')
    screenshot(gui_driver)


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
* The graphs seem to come as SVG elements, but the Nokia Communicator
does not support SVG rendering. Render them as JPEGs instead.
"""


if __name__ == "__main__":
    gui_driver.get(redmine_url + '/issues/new')

    gui_driver.find_element_by_id('issue_subject').send_keys(issue_title)
    gui_driver.find_element_by_id('issue_description').send_keys(issue_description)
    screenshot(gui_driver)


if __name__ == "__main__":
    gui_driver.find_element_by_id('issue_assigned_to_id').click()
    screenshot(gui_driver)


if __name__ == "__main__":
    gui_driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    screenshot(gui_driver)


if __name__ == "__main__":
    gui_driver.find_element_by_name('commit').click()
    screenshot(gui_driver)


# ## The Life Cycle of a Bug

if __name__ == "__main__":
    print('\n## The Life Cycle of a Bug')




if __package__ is None or __package__ == "":
    from Intro_Debugging import graph
else:
    from .Intro_Debugging import graph


if __name__ == "__main__":
    from IPython.display import display


if __name__ == "__main__":
    life_cycle = graph()
    # life_cycle.node('Unconfirmed')
    life_cycle.node('New')
    life_cycle.node('Assigned')
    life_cycle.node('Resolved')
    # life_cycle.node('Reopen')
    # life_cycle.node('Verified')
    life_cycle.node('Closed')

    life_cycle.edge('New', 'Assigned')
    life_cycle.edge('Assigned', 'Resolved')
    life_cycle.edge('Resolved', 'Closed')

    life_cycle


if __name__ == "__main__":
    life_cycle.node('Unconfirmed')
    life_cycle.node('Reopen')
    life_cycle.node('Verified')

    life_cycle.edge('Unconfirmed', 'New')
    life_cycle.edge('Assigned', 'New')
    life_cycle.edge('Resolved', 'Verified')
    life_cycle.edge('Resolved', 'Reopen')
    life_cycle.edge('Resolved', 'Unconfirmed')
    life_cycle.edge('Closed', 'Unconfirmed')
    life_cycle.edge('Verified', 'Reopen')
    life_cycle.edge('Reopen', 'Assigned')

    life_cycle


# ### Assigning Bug Reports

if __name__ == "__main__":
    print('\n### Assigning Bug Reports')




# ## Prioritizing Bug Reports

if __name__ == "__main__":
    print('\n## Prioritizing Bug Reports')




# ## Cleanup

if __name__ == "__main__":
    print('\n## Cleanup')




import os

if __name__ == "__main__":
    redmine_process.terminate()


if __name__ == "__main__":
    gui_driver.close()


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



