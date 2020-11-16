# About this Book

__Welcome to "The Debugging Book"!__ 
Software has bugs, and finding bugs can involve lots of effort.  This book addresses this problem by _automating_ software debugging, specifically by _locating errors and their causes automatically_.  Recent years have seen the development of novel techniques that lead to dramatic improvements in automated software debugging.  They now are mature enough to be assembled in a book – even with executable code. 

**<span style="background-color: yellow">
    <i class="fa fa-fw fa-wrench"></i>
This book is work in progress. It will be released to the public in the beginning of 2021.</span>**

## A Textbook for Paper, Screen, and Keyboard

You can use this book in four ways:

* You can __read chapters in your browser__.  Check out the list of chapters in the menu above, or start right away with the [introduction to debugging](https://www.debuggingbook.org/html/Intro_Debugging.html) or [how debuggers work](https://www.debuggingbook.org/html/Debugger.html).  All code is available for download.

* You can __interact with chapters as Jupyter Notebooks__ (beta).  This allows you to edit and extend the code, experimenting _live in your browser._  Simply select "Resources $\rightarrow$ Edit as Notebook" at the top of each chapter. <a href="https://mybinder.org/v2/gh/uds-se/debuggingbook/master?filepath=docs/notebooks/Debugger.ipynb" target=_blank>Try interacting with the introduction to interactive debuggers.</a>

* You can __use the code in your own projects__.  You can download the code as Python programs; simply select "Resources $\rightarrow$ Download Code" for one chapter or "Resources $\rightarrow$ All Code" for all chapters.  These code files can be executed, yielding (hopefully) the same results as the notebooks.  Once the book is out of beta, you can also [install the Python package](https://www.debuggingbook.org/html/Importing.html).

* You can __present chapters as slides__.  This allows for presenting the material in lectures.  Just select "Resources $\rightarrow$ View slides" at the top of each chapter. <a href="https://www.debuggingbook.org/slides/Debugger.slides.html" target=_blank>Try viewing the slides for how debuggers work.</a>

## Who this Book is for

This work is designed as a _textbook_ for a course in software debugging; as _supplementary material_ in a software testing or software engineering course; and as a _resource for software developers_. We cover fault localization, program slicing, input reduction, automated repair, and much more, illustrating all techniques with code examples that you can try out yourself.

## News

This book is _work in progress_, with new chapters being added every week. To get notified on updates, attend one of our courses or <a href="https://twitter.com/Debugging_Book?ref_src=twsrc%5Etfw" data-show-count="false">follow us on Twitter</a>.

<a class="twitter-timeline" data-width="500" data-chrome="noheader nofooter noborders transparent" data-link-color="#A93226" data-align="center" href="https://twitter.com/Debugging_Book?ref_src=twsrc%5Etfw" data-dnt="true">News from @Debugging_Book</a> 


## About the Authors

This book is written by _Andreas Zeller_, a long-standing expert in automated debugging, software analysis and software testing.  Andreas is happy to share his expertise and making it accessible to the public.

## Frequently Asked Questions

### Troubleshooting

#### Why does it take so long to start an interactive notebook?

The interactive notebook uses the [mybinder.org](https://mybinder.org) service, which runs notebooks on their own servers.  Starting Jupyter through mybinder.org normally takes about 30 seconds, depending on your Internet connection. If, however, you are the first to invoke binder after a book update, binder recreates its environment, which will take a few minutes.  Reload the page occasionally.

#### The interactive notebook does not work!

mybinder.org imposes a [limit of 100 concurrent users for a repository](https://mybinder.readthedocs.io/en/latest/user-guidelines.html).  Also, as listed on the [mybinder.org status and reliability page](https://mybinder.readthedocs.io/en/latest/reliability.html),

> As mybinder.org is a research pilot project, the main goal for the project is to understand usage patterns and workloads for future project evolution. While we strive for site reliability and availability, we want our users to understand the intent of this service is research and we offer no guarantees of its performance in mission critical uses.

There are alternatives to mybinder.org; see below.

#### Do I have alternatives to the interactive notebook?

If mybinder.org does not work or match your needs, you have a number of alternatives:

1. **Download the Python code** (using the menu at the top) and edit and run it in your favorite environment.  This is easy to do and does not require lots of resources.
2. **Download the Jupyter Notebooks** (using the menu at the top) and open them in Jupyter.  Here's [how to install jupyter notebook on your machine](https://www.dataquest.io/blog/jupyter-notebook-tutorial/).

#### Can I run the code on my Windows machine?

We try to keep the code as general as possible, but occasionally, when we interact with the operating system, we assume a Unix-like environment (because that is what Binder provides).  To run these examples on your own Windows machine, you can install a Linux VM or a [Docker environment](https://github.com/uds-se/fuzzingbook/blob/master/deploy/README.md).

#### Can't you run your own dedicated cloud service?

Technically, yes; but this would cost money and effort, which we'd rather spend on the book at this point.  If you'd like to host a [JupyterHub](http://jupyter.org/hub) or [BinderHub](https://github.com/jupyterhub/binderhub) instance for the public, please _do so_ and let us know.

### Content

#### Can I use your code in my own programs?

Yes!  See the [installation instructions](https://www.debuggingbook.org/html/Importing.html) for details.

#### Which content has come up?

See the [release notes](https://www.debuggingbook.org/html/ReleaseNotes.html) for details.

#### How do I cite your work?

Thanks for referring to our work!  Once the book is complete, you will be able to cite it in the traditional way.  In the meantime, just click on the "cite" button at the bottom of the Web page for each chapter to get a citation entry.

#### Can you cite my paper?  And possibly write a chapter about it?

We're always happy to get suggestions!  If we missed an important reference, we will of course add it.  If you'd like specific material to be covered, the best way is to _write a notebook_ yourself; see our [Guide for Authors](https://www.debuggingbook.org/html/Guide_for_Authors.html) for instructions on coding and writing.  We can then refer to it or even host it.

### Teaching and Coursework

#### Can I use your material in my course?

Of course!  Just respect the [license](https://github.com/uds-se/debuggingbook/blob/master/LICENSE.md) (including attribution and share alike).  If you want to use the material for commercial purposes, contact us.

#### Can I extend or adapt your material?

Yes!  Again, please see the [license](https://github.com/uds-se/debuggingbook/blob/master/LICENSE.md) for details.

#### How can I run a course based on the book?

We have successfully used the material in various courses.  

* Initially, we used the slides and code and did _live coding_ in lectures to illustrate how a technique works. 

* Now, the goal of the book is to be completely self-contained; that is, it should work without additional support.  Hence, we now give out completed chapters to students in a _flipped classroom_ setting, with the students working on the notebooks at their leisure.  We would meet in the classroom to discuss experiences with past notebooks and discuss future notebooks.

* We have the students work on exercises from the book or work on larger (fuzzing) projects.  We also have students who use the book as a base for their research; indeed, it is very easy to prototype in Python for Python.

When running a course, [do not rely on mybinder.org](#Troubleshooting) – it will not provide sufficient resources for a larger group of students.  Instead, [install and run your own hub.](#Do-I-have-alternatives-to-the-interactive-notebook?)

#### Are there specific subsets I can focus on?

We will compile a number of [tours through the book](https://www.debuggingbook.org/html/Tours.html) for various audiences.  Our [Sitemap](https://www.debuggingbook.org/html/00_Table_of_Contents.html) lists the dependencies between the individual chapters.

#### How can I extend or adapt your slides?

Download the Jupyter Notebooks (using the menu at the top) and adapt the notebooks at your leisure (see above), including "Slide Type" settings.  Then,

1. Download slides from Jupyter Notebook; or
2. Use the RISE extension ([instructions](http://www.blog.pythonlibrary.org/2018/09/25/creating-presentations-with-jupyter-notebook/)) to present your slides right out of Jupyter notebook.

#### Do you provide PDFs of your material?

At this point, we do not provide support for PDF versions.  We will be producing PDF and print versions after the book is complete.

### Other Issues

#### I have a question, comment, or a suggestion.  What do I do?

You can [tweet to @debugging_book on Twitter](https://twitter.com/debugging_book), allowing the community of readers to chime in.  For bugs you'd like to get fixed, report an issue on the [development page](https://github.com/uds-se/debuggingbook/issues).

#### I have reported an issue two weeks ago.  When will it be addressed?

We prioritize issues as follows:

1. Bugs in code published on fuzzingbook.org
2. Bugs in text published on fuzzingbook.org
3. Writing missing chapters
4. Issues in yet unpublished code or text
5. Issues related to development or construction
6. Things marked as "beta"
7. Everything else

#### How can I solve problems myself?

We're glad you ask that.  The [development page](https://github.com/uds-se/debuggingbook/) has all sources and some supplementary material.  Pull requests that fix issues are very welcome.

#### How can I contribute?

Again, we're glad you're here!  We are happy to accept 

* **Code fixes and improvements.**  Please place any code under the MIT license such that we can easily include it.
* **Additional text, chapters, and notebooks** on specialized topics.  We plan to set up a special folder for third-party contributions.

See our [Guide for Authors](https://www.debuggingbook.org/html/Guide_for_Authors.html) for instructions on coding and writing.
