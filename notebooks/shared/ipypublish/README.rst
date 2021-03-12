ipypublish
==========

**Project**: https://github.com/chrisjsewell/ipypublish

**Documentation**: http://ipypublish.readthedocs.io

.. image:: https://travis-ci.org/chrisjsewell/ipypublish.svg?branch=master
    :target: https://travis-ci.org/chrisjsewell/ipypublish

A workflow for creating and editing publication ready scientific
reports and presentations, from one or more Jupyter Notebooks, without leaving the
browser!

.. image:: https://github.com/chrisjsewell/ipypublish/raw/master/example_workflow.gif

For an example of the potential input/output, see `Example.ipynb <https://github.com/chrisjsewell/ipypublish/raw/master/example/notebooks/Example.ipynb>`__ , `Example.pdf <https://chrisjsewell.github.io/ipypublish/Example.view_pdf.html>`__, `Example.html <https://chrisjsewell.github.io/ipypublish/Example.html>`__ and `Example.slides.html <https://chrisjsewell.github.io/ipypublish/Example.slides.html#/>`__.

Design Philosophy
-----------------

In essence, the dream is to have the ultimate hybrid of Jupyter
Notebook, WYSIWYG editor (e.g. MS Word) and document preparation system
(e.g. `TexMaker <http://www.xm1math.net/texmaker/>`__), being able to:

-  Dynamically (and reproducibly) explore data, run code and output the
   results
-  Dynamically edit and visualise the basic components of the document
   (text, math, figures, tables, references, citations, etc).
-  Have precise control over what elements are output to the final
   document and how they are layed out and typeset.

   -  Also be able to output the same source document to different
      layouts and formats (pdf, html, presentation slides, etc).

Workflow
--------

1. Create a notebook with some content!
2. optionally create a .bib file and logo image
3. Adjust the notebook and cell metadata.
4. Clone the ipypublish `GitHub
   repository <https://github.com/chrisjsewell/ipypublish>`__ and run
   the nbpublish.py script for either the specific notebook, or a folder
   containing multiple notebooks.
5. A converted folder will be created, into which final .tex .pdf and
   \_viewpdf.html files will be output, named by the notebook or folder
   input

The default latex template outputs all markdown cells (unless tagged
latex\_ignore), and then only code and output cells with `latex metadata
tags <#latex-metadata-tags>`__. See
`Example.ipynb <https://github.com/chrisjsewell/ipypublish/blob/master/example/notebooks/Example.ipynb>`__, 
`Example.pdf <https://chrisjsewell.github.io/ipypublish/Example.view_pdf.html>`__ and `Example.slides.html <https://chrisjsewell.github.io/ipypublish/Example.slides.html#/>`__
for an example of the potential input and output.

**See the project site for more info!**

Acknowledgements
----------------

I took strong influence from:

-  `Julius
   Schulz <http://blog.juliusschulz.de/blog/ultimate-ipython-notebook>`__
-  `Dan
   Mackinlay <https://livingthing.danmackinlay.name/jupyter.html>`__
-  Notebook concatenation was adapted from `nbconvert
   issue#253 <https://github.com/jupyter/nbconvert/issues/253>`__

