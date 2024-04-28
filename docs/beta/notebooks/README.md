<!-- Debuggingbook README -->

<!-- Badges to be shown on project page -->

[![Code Tests](https://github.com/uds-se/debuggingbook/actions/workflows/check-code.yml/badge.svg)](https://github.com/uds-se/debuggingbook/actions/workflows/check-code.yml)
&nbsp;
[![Notebook Tests](https://github.com/uds-se/debuggingbook/actions/workflows/check-notebooks.yml/badge.svg)](https://github.com/uds-se/debuggingbook/actions/workflows/check-notebooks.yml)
&nbsp;
[![Static Type Checking](https://github.com/uds-se/debuggingbook/actions/workflows/check-types.yml/badge.svg)](https://github.com/uds-se/debuggingbook/actions/workflows/check-types.yml)
&nbsp;
[![Imports](https://github.com/uds-se/debuggingbook/actions/workflows/check-imports.yml/badge.svg)](https://github.com/uds-se/debuggingbook/actions/workflows/check-imports.yml)
&nbsp;
[![Website www.debuggingbook.org](https://img.shields.io/website-up-down-green-red/https/www.debuggingbook.org.svg)](https://www.debuggingbook.org/)

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/uds-se/debuggingbook/master?filepath=docs/notebooks/00_Table_of_Contents.ipynb)
&nbsp;
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
&nbsp;
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)](https://www.jupyter.org/)
&nbsp;
[![License: MIT (Code), CC BY-NC-SA (Book)](https://img.shields.io/badge/License-MIT_(Code),_CC_BY--NC--SA_4.0_(Book)-blue.svg)](https://github.com/uds-se/debuggingbook/blob/master/LICENSE.md)


# About this Book

__Welcome to "The Debugging Book"!__ 
Software has bugs, and finding bugs can involve lots of effort.  This book addresses this problem by _automating_ software debugging, specifically by _locating errors and their causes automatically_.  Recent years have seen the development of novel techniques that lead to dramatic improvements in automated software debugging.  They now are mature enough to be assembled in a book – even with executable code. 
<!--
**<span style="background-color: yellow">
    <i class="fa fa-fw fa-wrench"></i>
This book is work in progress. It will be released to the public in Spring 2021.</span>**
-->


```python
from bookutils import YouTubeVideo
YouTubeVideo("-nOxI6Ev_I4")
```





<a href="https://www.youtube-nocookie.com/embed/-nOxI6Ev_I4" target="_blank">
<img src="https://www.debuggingbook.org/html/PICS/youtube.png" width=640>
</a>
        



## A Textbook for Paper, Screen, and Keyboard

You can use this book in four ways:

* You can __read chapters in your browser__.  Check out the list of chapters in the menu above, or start right away with the [introduction to debugging](https://www.debuggingbook.org/html/Intro_Debugging.html) or [how debuggers work](https://www.debuggingbook.org/html/Debugger.html).  All code is available for download.

* You can __interact with chapters as Jupyter Notebooks__ (beta).  This allows you to edit and extend the code, experimenting _live in your browser._  Simply select "Resources → Edit as Notebook" at the top of each chapter. <a href="https://mybinder.org/v2/gh/uds-se/debuggingbook/master?filepath=docs/notebooks/Debugger.ipynb" target=_blank>Try interacting with the introduction to interactive debuggers.</a>

* You can __use the code in your own projects__.  You can download the code as Python programs; simply select "Resources → Download Code" for one chapter or "Resources → All Code" for all chapters.  These code files can be executed, yielding (hopefully) the same results as the notebooks.  Once the book is out of beta, you can also [install the Python package](https://www.debuggingbook.org/html/Importing.html).

* You can __present chapters as slides__.  This allows for presenting the material in lectures.  Just select "Resources → View slides" at the top of each chapter. <a href="https://www.debuggingbook.org/slides/Debugger.slides.html" target=_blank>Try viewing the slides for how debuggers work.</a>

## Who this Book is for

This work is designed as a _textbook_ for a course in software debugging; as _supplementary material_ in a software testing or software engineering course; and as a _resource for software developers_. We cover fault localization, program slicing, input reduction, automated repair, and much more, illustrating all techniques with code examples that you can try out yourself.

## News

This book is _work in progress._  All chapters planned are out now, but we keep on refining text and code with [minor and major releases.](https://www.debuggingbook.org/html/ReleaseNotes.html) To get notified on updates, <a href="https://mastodon.social/invite/P27cijZH">follow us on Mastodon</a>.

<!--
