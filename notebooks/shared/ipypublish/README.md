[![Build Status](https://travis-ci.org/chrisjsewell/ipypublish.svg?branch=master)](https://travis-ci.org/chrisjsewell/ipypublish)
[![Coverage Status](https://coveralls.io/repos/github/chrisjsewell/ipypublish/badge.svg?branch=master)](https://coveralls.io/github/chrisjsewell/ipypublish?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/243d0038a2f543e7a9c47a781ca3cbf5)](https://www.codacy.com/app/chrisj_sewell/ipypublish?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=chrisjsewell/ipypublish&amp;utm_campaign=Badge_Grade)
[![PyPI](https://img.shields.io/pypi/v/ipypublish.svg)](https://pypi.python.org/pypi/ipypublish/)

# ipypublish
A workflow for creating and editing publication ready scientific reports and presentations, from one or more Jupyter Notebooks, without leaving the browser!

![WorkFlow Example](/example_workflow.gif)

See [Example.ipynb](example/notebooks/Example.ipynb), [Example.pdf](https://chrisjsewell.github.io/ipypublish/Example.view_pdf.html),
[Example.html](https://chrisjsewell.github.io/ipypublish/Example.html) and 
[Example.slides.html](https://chrisjsewell.github.io/ipypublish/Example.slides.html#/) for an example of the potential input/output.

Or, for a practical example of the ipypublish capability, see these documents on Atomic 3D Visualisation: 
[Notebook](https://github.com/chrisjsewell/chrisjsewell.github.io/blob/master/3d_atomic/3D%20Atomic%20Visualisation.ipynb), [PDF](https://chrisjsewell.github.io/3d_atomic/converted/3D%20Atomic%20Visualisation.view_pdf.html), [HTML](https://chrisjsewell.github.io/3d_atomic/converted/3D%20Atomic%20Visualisation.html) or [Reveal.JS slideshow](https://chrisjsewell.github.io/3d_atomic/converted/3D%20Atomic%20Visualisation.slides.html).

- [Design Philosophy](#design-philosophy)
- [Workflow](#worklow)
- [Setting up the environment](#setting-up-the-environment)
- [Setting up a Notebook ](#setting-up-a-notebook)
- [Converting Notebooks](#converting-notebooks)
    - [Creating a bespoke converter](#creating-a-bespoke-converter)
- [Metadata Tags](#metadata-tags)
    - [Object Output Formats](#object-output-formats)
    - [Captions in a Markdown cell](#captions-in-a-markdown-cell)
	- [Embedding Interactive HTML](#embedding-interactive-html)
- [Citations and Bibliography](#citations-and-bibliography)
- [Live Slideshows](#live-slideshows)
- [Dealing with external data](#dealing-with-external-data)
- [Miscellaneous](#miscellaneous)
- [Acknowledgements](#acknowledgements)


## Design Philosophy

In essence, the dream is to have the ultimate hybrid of Jupyter Notebook, WYSIWYG editor (e.g. MS Word) and document preparation system (e.g. [TexMaker](http://www.xm1math.net/texmaker/)), being able to:

- Dynamically (and reproducibly) explore data, run code and output the results
- Dynamically edit and visualise the basic components of the document (text, math, figures, tables, references, citations, etc).
- Have precise control over what elements are output to the final document and how they are layed out and typeset.
     - Also be able to output the same source document to different layouts and formats (pdf, html, presentation slides, etc).

## Workflow

1. Create a notebook with some content!
2. optionally create a .bib file and external images
3. Adjust the notebook and cell metadata. 
4. install ipypublish and run the `nbpublish` for either the specific notebook, or a folder containing multiple notebooks. 
5. A converted folder will be created, into which final .tex .pdf and _viewpdf.html files will be output, named by the notebook or folder input

The default latex template outputs all markdown cells (unless tagged latex_ignore), and then only code and output cells with [latex metadata tags](#latex-metadata-tags). 
See [Example.ipynb](https://github.com/chrisjsewell/ipypublish/blob/master/example/notebooks/Example.ipynb), [Example.pdf](https://chrisjsewell.github.io/ipypublish/Example.view_pdf.html),
[Example.html](https://chrisjsewell.github.io/ipypublish/Example.html) and [Example.slides.html](https://chrisjsewell.github.io/ipypublish/Example.slides.html#/) for examples of the potential input and output.

## Setting up the environment

Using [Conda](https://conda.io/docs/) is recommended for package management, 
in order to create self contained environments with specific versions of packages. 
The main external packages required are the Jupyter notebook, and [Pandoc](http://pandoc.org) (for conversion of file formats):

	conda create --name ipyreport -c conda-forge jupyter pandoc
	
ipypublish can then be installed into this environment:

	source activate ipyreport
	pip install ipypublish
	
For converting to PDF, the TeX document preparation ecosystem is required (an in particular [latexmk](http://mg.readthedocs.io/latexmk.html)), which can be installed from:

- Linux: [TeX Live](http://tug.org/texlive/)
- macOS (OS X): [MacTeX](http://tug.org/mactex/)
- Windows: [MikTex](http://www.miktex.org/)

ipypublish is automatically **tested** on update against **python 2.7 and 3.6**, for both **Linux and OSX**, using [Travis CI](https://en.wikipedia.org/wiki/Travis_CI). Therefore, to troubleshoot any installation/run issues, 
it is best to look at the [travis config](https://github.com/chrisjsewell/ipypublish/blob/master/.travis.yml) 
and [travis test runs](https://travis-ci.org/chrisjsewell/ipypublish) for working configurations.

For helpful extensions to the notebooks core capabilities (like a toc sidebar):

	conda install --name ipyreport jupyter_contrib_nbextensions
	
A more extensive setup of useful packages (used to create the example) 
are listed in ![conda_packages.txt](conda_packages.txt) 
and an environment can be created directly from this using conda:

	conda create --name ipyreport -c conda-forge -c matsci --file conda_packages.txt
	
## Setting up a Notebook 

For improved latex/pdf output, `ipynb_latex_setup.py` contains import and setup code for the notebook and a number of common packages and functions, including:

- numpy, matplotlib, pandas, sympy, ...
- `images_hconcat`, `images_vconcat` and `images_gridconcat` functions, which use the PIL/Pillow package to create a single image from multiple images (with specified arrangement)

To use this script, in the first cell of a notebook, insert:

```python
from ipypublish.scripts.ipynb_latex_setup import *
```

It is recommended that you also set this cell as an initialisation cell (i.e. have `"init_cell": true` in the metadata)

For existing notebooks: the **nb_ipypublish_all** and **nb_ipypublish_nocode** converters (see below) can be helpful for outputting a notebook, with identical content to that input, but with default metatags defining how content is to be output.

## Converting Notebooks

The nbpublish script handles parsing the notebooks to nbconvert, with the appropriate converter. To see all options for this script:

	nbpublish -h

For example, to convert the Example.ipynb notebook directly to pdf:

	nbpublish -pdf example/notebooks/Example.ipynb

If a folder is input, then the .ipynb files it contains are processed and combined in 'natural' sorted order, i.e. 2_name.ipynb before 10_name.ipynb. By default, notebooks beginning '_' are ignored.

All available converters are also listed by `nbpublish -h`. Three of note are:

- **latex_ipypublish_main** is the **default** and converts cells to latex according to metadata tags on an 'opt in' basis. Note that, for this converter, **no code cells or output** will appear in the final tex/pdf document unless they have a suitable [ipub metadata tag](#latex-metadata-tags).
- **html_ipypublish_main** converts the entire notebook(s) to html and adds a table of contents sidebar and a button to toggle input code and output cells visible/hidden, with latex citations and references resolved. 
- **slides_ipypublish_main** converts the notebook to [reveal.js](http://lab.hakim.se/reveal-js/#/) slides, with latex citations and references resolved and slide partitioning by markdown headers. See the [Live Slideshows](#live-slideshows) section for using `nbpresent` to serve these slides to a webbrowser. 
- The **all** and **nocode** variants of these converters preprocess a copy of the notebook, to add default metadata tags to the notebook and all cells, such that all output is rendered (with or without the code)

The current `nbconvert --to pdf` does not correctly resolve references and citations (since it copies the files to a temporary directory). Therefore nbconvert is only used for the initial `nbconvert --to latex` phase, followed by using `latexmk` to create the pdf and correctly resolve everything. **To convert your own notebook to PDF** for the first time, a good route would be to use:

	nbpublish -f latex_ipypublish_all -pdf --pdf-debug path/to/YourNotebook.ipynb
	
**To raise any issues** please include the converted/YourNotebook.nbpub.log file.

### The ipypublish defaults

The ipypublish 'main' converters are designed with the goal of creating a single notebook, which may contain lots of exploratory code/outputs, mixed with final output, and that can be output as both a document (latex/pdf or html) and a presentation (reveal.js). The logic behind the default output is then:

- For documents: all headings and body text is generally required, but only a certain subset of code/output
- For slides: all headings are required, but most of the body text will be left out and sustituted with 'abbreviated' versions, and only a certain subset of code/output.

This leads to the following logic flow (discussed further in the [Metadata Tags](#metadata-tags) section):

**latex_ipypublish_main**/**html_ipypublish_main**:

- all cells: bypass "ignore" and "slideonly" tags
- markdown cells: include all
- code cells (input): only include if the "code" tag is present
- code cells (output): only include if the following tags are present
    - "figure" for png/svg/pdf/jpeg or html (html only)
	- "table" or "equation" for latex or html (html only)
	- "mkdown" for markdown text
	- "text" for plain text

**slides_ipypublish_main**:

- all cells: bypass "ignore"
- markdown cells: are first split into header (beggining #)/non-header components
    - headers: include all
	- non-headers: only include if "slide" tag
- code cells (input): only include if the "code" tag is present
- code cells (output): only include if the following tags are present
    - "figure" for png/svg/pdf/jpeg/html
	- "table" or "equation" for latex/html
	- "mkdown" for markdown text
	- "text" for plain text

Packages, such as pandas and matplotlib, use jupyter notebooks [rich representation](http://ipython.readthedocs.io/en/stable/config/integrating.html#rich-display) mechanics to store a single output in multiple formats. nbconvert (and hence ipypublish) then selects only the highest priority (compatible) format to be output. This allows, for example, for pandas DataFrames to be output as 
latex tables in latex documents and html tables in html documents/slides.

### Creating a bespoke converter

On instatiation, ipypublish loads all converter plugins in its internal [export_plugins](https://github.com/chrisjsewell/ipypublish/tree/master/ipypublish/export_plugins) folder. Additionally, when `nbpublish` or `nbpresent` are called, if a folder named **ipypublish_plugins** is present in the current working directory, they will load all plugins in this folder. Programatically, it is the `ipypublish.export_plugins.add_directory` function which is being called and adding modules to an internal dictionary.

The simplest application of this, would be to copy the [latex_ipypublish_all.py](https://github.com/chrisjsewell/ipypublish/blob/master/ipypublish/export_plugins/latex_ipypublish_all.py) file (or the html/slides variants) and make changes to the `cell_defaults` and `nb_defaults` dictionaries to suit your output needs.

A plugin is a python (.py) file with at least the following four variables (i.e. it's interface spec):

1. a **docstring** describing its output format
2. an **oformat** string,  specifying a base exporter prefix (for any of the exporters listed [here](https://nbconvert.readthedocs.io/en/latest/api/exporters.html#specialized-exporter-classes))
3. a **config** dictionary, containing any configuration option (as a string) listed [here](https://nbconvert.readthedocs.io/en/latest/api/exporters.html#specialized-exporter-classes). This is mainly to supply preprocessors (which act on the notbook object before it is parsed) or filters (which are functions supplied to the jinja template).
4. a **template** string, specifying the [Jinja templates](https://jinja2.readthedocs.io/en/latest/intro.html), which contains rules for how each element of the notebook should be converted, and also what each section of the latex file should contain. 
5. It is not required, but recommended, to also include the version number of ipypublish which the plugin was written for.

So a simple plugin would look like this (create_tplx will be explained below) 

```python
"""this exporter exports a .tex file with nothing in it
"""
from ipypublish.latex.create_tplx import create_tplx
oformat = 'Latex'
config = {}
template = create_tplx()

```

This is similar to how nbconvert works, except for one key difference, 
the plugin must specify the entire jinja template (rather than using a default one).
The advantage of this, is that the plugin has complete control over the look of the final document.

To aid in the creation of the jinja template, the `create_tplx` (for latex) and `create_tpl` (for html) functions
work by creating an inital *skeleton* template, with placeholders in all the relevant [structural blocks](https://nbconvert.readthedocs.io/en/latest/customizing.html#Template-structure). 
They then take a list of *fragment* dictionaries which progressively append input to the relevant blocks. 
So, for instance:   

```python
"""exports a .tex file containing 
some latex setup and
only input markdown cells from the notebook 
"""
from ipypublish.latex.create_tplx import create_tplx
oformat = 'Latex'
config = {}

doc_dict = {
	'document_docclass':r'\documentclass[11pt]{article}',
	'document_packages':r"""
	\usepackage{caption}
	 \usepackage{amsmath}
	"""
}

mkdown_dict = {
  'notebook_input_markdown':r"""
 	((( cell.source | citation2latex | strip_files_prefix | convert_pandoc('markdown', 'json',extra_args=[]) | resolve_references | convert_pandoc('json','latex') )))
	"""
}

template = create_tplx([doc_dict,mkdown_dict])

```

This approach allows independant aspects of the document to be 
stored separately then pieced together in the desired manner. 
ipypublish stores all of the standard fragments in separate modules,
for instance the latex_standard_article plugin looks like this: 

```python
"""latex article in the standard nbconvert format
"""

from ipypublish.latex.create_tplx import create_tplx
from ipypublish.latex.standard import standard_article as doc
from ipypublish.latex.standard import standard_packages as package
from ipypublish.latex.standard import standard_definitions as defs
from ipypublish.latex.standard import standard_contents as content
from ipypublish.latex.standard import in_out_prompts as prompts

oformat = 'Latex'
template = create_tplx(
    [package.tplx_dict,defs.tplx_dict,doc.tplx_dict,
    content.tplx_dict,prompts.tplx_dict])

config = {}

```

Now, if you wanted mainly the same output format but without input and output prompts shown,
simply copy this plugin but remove the prompts.tplx_dict.

By default, sections are appended to, so;

```python
dict1 = {'notebook_input':'a'}
dict2 = {'notebook_input':'b'}
template = create_tplx([dict1,dict2])
```

would show a, then b. But, if you want to redefine a particular section(s);

```python
dict1 = {'notebook_input':'a'}
dict2 = {
	'overwrite':['notebook_input'],
	'notebook_input':'b'}
template = create_tplx([dict1,dict2])
```

will only show b.

Note that, the `create_tpl` template additionally has *pre* and *post* placeholder. 
This is helpful for wrapping cells in extra html tags. For instance:

```python

dict1 = {
  'notebook_input_markdown_pre':r"<div class="inner">",
  'notebook_input_markdown':"test",
  'notebook_input_markdown_post':r"</div>",
}
dict2 = {
  'notebook_input_markdown_pre':r"<div class="outer">",
  'notebook_input_markdown_post':r"</div>",
}

template = create_tpl([dict1,dict2])

```

will result in a template containing:

```html
<div class="outer">
<div class="inner">
test
</div>
</div>
```

## Metadata Tags

All information additional information, used to specify how a particular notebook/cell/output will be represented, when converted, is stored in the metadata under:

```json
{
		"ipub": {}
}
```

There are three levels of metadata:

- For notebook level: in the Jupyter Notebook Toolbar go to Edit -> Edit Notebook Metadata
- For cell level: in the Jupyter Notebook Toolbar go to View -> Cell Toolbar -> Edit Metadata and a button will appear above each cell.
- For output level: using `IPython.display.display(obj,metadata={"ipub":{}})`, you can set metadata specific to a certain output. Options set at the output level will override options set at the cell level. for an example of this, run the [MultiOutput_Example.ipynb](example/notebooks/MultiOutput_Example.ipynb).

**Please note**, setting a value to `"value":{}` is the same as `"value":false` so,
if you are not setting additional options, use `"value":true`.

### Document Tags

To change the **language** of the document:

```json
{
"ipub": {
	"language" : "french"
	}
}
```

where the language can be any specified in the 
[babel](https://people.phys.ethz.ch/~ihn/latex/babel.pdf) package.

To specify where the **bibliography** is:

```json
{
"ipub": {
	"bibliography" : "path/to/bibliograph.bib"
	}
}
```

The path can be absolute or relative.

For **titlepage**, enter in notebook metadata:

```json
{
"ipub": {
  "titlepage": {
	"author": "Authors Name",
	"email": "authors@email.com",
	"supervisors": [
	  "First Supervisor",
	  "Second Supervisor"
	],
	"title": "Main-Title",
	"subtitle": "Sub-Title",
	"tagline": "A tagline for the report.",
	"institution": [
	  "Institution1",
	  "Institution2"
	],
	"logo": "path/to/logo_example.png"
	}
  }
}
```

- all keys are optional
- if there is no title, then the notebook filename will be used
- if nbpublish.py is called on a folder, then the meta data from the first notebook will be used
- logo should be the path (absolute or relative) to a logo image file

To control the output of **contents tables**:

```json
{
"ipub": {
  "toc": true,
  "listfigures": true,
  "listtables": true,
  "listcode": true,
  }
}
```

To override the default **placement of figures and tables**:

```json
{
"ipub": {
    "figure": {
      "placement": "!bp"
      },
    "table": {
      "placement": "!bp"
      }
  }
}
```

See [Positioning_images_and_tables](https://www.sharelatex.com/learn/Positioning_images_and_tables) for placement options.

### Cell Tags

To **ignore any cell** for all outputs:

```json
{
"ipub": {
	"ignore" : true
	}
}
```

To mark any cell as for output to **slides only**:

```json
{
"ipub": {
	"slideonly" : true
	}
}
```

To  **output a code block**:

```json
{
"ipub": {
  "code": {
	"format" : {},
    "asfloat": true,
    "caption": "",
    "label": "code:example_sym",
    "widefigure": false,
    "placement": "H"
    }
  }
}
```

all extra tags are optional:

- `format` can contain any keywords related to the latex [Listings](https://en.wikibooks.org/wiki/LaTeX/Source_Code_Listings) package (such as syntax highlighting colors)
- `asfloat` contitutes whether the code is wrapped in a codecell (float) environment or is inline.
- all other tags work the same as figure (below).

To  **output text produced by the code** (e.g. *via* the `print` command):

```json
{
"ipub": {
  "text": {
      "format": {
       "basicstyle": "\\small"
      },
    "asfloat": true,
    "caption": "",
    "label": "code:example_sym",
    "widefigure": false,
    "placement": "H",
	"use_ansi": false
    }
  }
}
```

all extra tags are optional:

- `format` can contain any keywords related to the latex [Listings](https://en.wikibooks.org/wiki/LaTeX/Source_Code_Listings) package (such as syntax highlighting colors). N.B. in place of `\` use `\\`.
- `asfloat` contitutes whether the code is wrapped in a codecell (float) environment or is inline.
- if `use_ansi` is true then, instead of stripping ansi colors in latex output, they will be converted to latex, wrapped in % characters and the listings option escapechar=\% set. 
- all other tags work the same as figure (below).


For **figures** (i.e. any graphics output by the code), enter in cell metadata:

```json
{
"ipub": {
  "figure": {
    "caption": "Figure caption.",
    "label": "fig:flabel",
    "placement": "H",
	"height":0.4,
    "widefigure": false,
    }
  }
}
```

- all tags are optional
- height/width correspond to the fraction of the page height/width, only one should be used (aspect ratio will be maintained automatically)
- `placement` is optional and constitutes using a placement arguments for the figure (e.g. \begin{figure}[H]). See [Positioning_images_and_tables](https://www.sharelatex.com/learn/Positioning_images_and_tables).
- `widefigure` is optional and constitutes expanding the figure to the page width (i.e. \begin{figure*}) (placement arguments will then be ignored)

For  **tables** (e.g. those output by `pandas`), enter in cell metadata:

```json
{
"ipub": {
     "table": {
	    "caption": "Table caption.",
	    "label": "tbl:tlabel",
	    "placement": "H",
            "alternate": "gray!20"
	  }
   }
}
```

- `caption` and `label` are optional
- `placement` is optional and constitutes using a placement arguments for the table (e.g. \begin{table}[H]). See [Positioning_images_and_tables](https://www.sharelatex.com/learn/Positioning_images_and_tables).
- `alternate` is optional and constitutes using alternating colors for the table rows (e.g. \rowcolors{2}{gray!25}{white}). See (https://tex.stackexchange.com/a/5365/107738)[https://tex.stackexchange.com/a/5365/107738].
- if tables exceed the text width, in latex, they will be shrunk to fit 


For  **equations** (e.g. thos output by `sympy`), enter in cell metadata:

```json
{
  "ipub": {
	  "equation": {
        "environment": "equation",
	    "label": "eqn:elabel"
	  }
  }
}
```

- environment is optional and can be 'none' or any of those available in [amsmath](https://www.sharelatex.com/learn/Aligning_equations_with_amsmath); 'equation', 'align','multline','gather', or their \* variants. Additionaly, 'breqn' or 'breqn\*' will select the experimental [breqn](https://ctan.org/pkg/breqn) environment to *smart* wrap long equations. 
- label is optional and will only be used if the equation is in an environment

For **slide output**:

```json
{
  "ipub": {
	  "slide": true
  }
}
```

- the value of slide can be true, "new" (to indicate the start of a new slide) or "notes"

### Object Output Formats

The format of the Jupyter Notebook (.ipynb) file allows for the storage of a single output in multiple formats. This is taken advantage of by packages such as matplotlib and pandas, etc to store a figure/table in both latex and html formats, which can then be selected by ipypublish based on the document type required. 

Sometimes a user may wish to have greater control over the output format
and/or which output types are to be stored. It it possible to achieve this *via* the Jupyter `display` function. For example, if we wanted to display a
pandas.DataFrame table without the index column, such that it can be output
to both a pdf and html document:

```python
from IPython.display import display
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.random((3,3)))
latex = df.to_latex(index=False)
html = df.to_html(index=False)
display({'text/latex':latex,
         'text/html':html},raw=True)
```

If you wish to create your own object with multiple output formats,
you should create a class with multiple `_repr_*_()` methods (as described [here](http://ipython.readthedocs.io/en/stable/config/integrating.html#rich-display)):

```python
class MyObject(object):
    def __init__(self, text):
        self.text = text

    def _repr_latex_(self):
        return "\\textbf{" + self.text + "}"

    def _repr_html_(self):
        return "<b>" + self.text + "</b>"
```

### Captions in a Markdown cell

Especially for long captions, it would be prefered that they can be viewed and edited in a notebook Markdown cell, rather than hidden in the metadata. This can be achieved using the default ipypublish converters:

If a **markdown cell** or **code cell with latex/text output** has the metadata tag:

```json
{
 "ipub": {
	"caption": "fig:example_mpl"
	}
}
```

Then, during the the postprocessor stage, this cell will be removed from the notebook object, and its text stored as a *resource*;

- the cell's text is the first paragraph of the markdown string, i.e. nothing after a newline (\n) 
- if there are multiple instance of the same cation name, then only the last instance will be stored

During the jinja templating, if a **figure, table or code** cell has a label matching any stored caption name, for example:

```json
{
"ipub": {
	"figure": {
	  "caption": "",
	  "label": "fig:example_mpl"
	}
  }
}
```

Then its caption will be overriden with the stored text. 

### Embedding Interactive HTML

Packages built on [IPywidgets](http://ipywidgets.readthedocs.io), 
like [PythreeJS](https://github.com/jovyan/pythreejs), 
[Pandas3JS](https://github.com/chrisjsewell/pandas3js) 
and the excellent [IPyvolume](https://ipyvolume.readthedocs.io/en/latest/), 
are making it increasingly easier to render complex, interactive html in the notebook. 
IPywidgets offers a [save notebook with widgets](http://ipywidgets.readthedocs.io/en/latest/embedding.html) feature, however, this can greatly increase the size of the notebook.


A better solution, recently offered, is to save a [html snippet](http://ipywidgets.readthedocs.io/en/latest/embedding.html#embeddable-html-snippet) 
of the current widget state to file and embed it into the html/slides output as an iframe. This is also particularly useful in reveal.js slides, 
since the iframe content can be [*lazy loaded*](https://github.com/hakimel/reveal.js/#lazy-loading).
To embed html, use the `embed_html` tag:

```json
{
  "ipub": {
    "embed_html": {
      "filepath": "path/to/file.html",
      "other_files": ["path/to/file.js"],
	  "url": "https//path/to/url.html",
	  "width":0.5,
	  "height":0.5
    },
    "figure": {
      "caption": "An example of embedded html"
    }
  }
}
```

If the cell already contains an output, then this tag will create/overwrite the first output's "text/html" type. 
This allows for a single notebook cell with a static image of the widget in the output, and a path to the embed html in the metadata so that a) if you export to latex/pdf, you get the static image or b) if you export to html/reveal slides, you get the html.

- use either filepath or url
- other_files are files required by the html file (e.g. javascript libraries). These files will be copied to the the same folder as the html
- width/height refers to the fraction of the viewspace used (e.g. 0.5 width -> 50vw and 0.5 height -> 50vh)

An example of how this works is in the [Example.ipynb](example/notebooks/Example.pdf), and the 
[Example.html](https://chrisjsewell.github.io/ipypublish/Example.html#Embedded-HTML-6) and 
[Example.slides.html](https://chrisjsewell.github.io/ipypublish/Example.slides.html#/9) outputs. 

## Citations and Bibliography

Using Zotero's Firefox plugin and [Zotero Better Bibtex](https://github.com/retorquere/zotero-better-bibtex/releases/tag/1.6.100) for;

- automated .bib file updating 
- drag and drop cite keys \cite{kirkeminde_thermodynamic_2012}
- `latexmk -bibtex -pdf` (in nbpublish.py) handles creation of the bibliography
- \usepackage{doi} turns the DOI numbers into url links

    - in Zotero-Better-Bibtex I have the option set to only export DOI, if both DOI and URL are present.
	
Please note, at the time of writing, Better BibTeX does not support Zotero 5.0 ([issue#555](https://github.com/retorquere/zotero-better-bibtex/issues/555)). For now I have turned off auto-updates of Zotero, though this is probably not wise for long ([Zotero 5 Discussion](https://forums.zotero.org/discussion/comment/277434/#Comment_277434)).

Can use: 

```html
<cite data-cite="kirkeminde_thermodynamic_2012">(Kirkeminde, 2012)</cite> 
```

to make it look better in html, but not specifically available for drag and drop in Zotero 
	
## Live Slideshows

The **nbpresent** script handles serving [reveal.js](http://lab.hakim.se/reveal-js/#/) slides to a webbrowser. To see all options for this script:

	nbpresent -h
	
Note that, for offline use, simply download the latest version of reveal.js [here](https://github.com/hakimel/reveal.js/releases), rename the entire folder to reveal.js and place it in the same folder as the converted .slides.html file. The slides can also be save to PDF my appending `pdf-export` to the url (see [here](https://github.com/hakimel/reveal.js#pdf-export) for details).


Additionally, the [Reveal.js - Jupyter/IPython Slideshow Extension (RISE)](https://github.com/damianavila/RISE) notebook extension offers rendering as a Reveal.js-based slideshow, where you can execute code or show to the audience whatever you can show/do inside the notebook itself! Click on the image to see a demo:

[![RISE Demo](https://img.youtube.com/vi/sXyFa_r1nxA/0.jpg)](https://www.youtube.com/watch?v=sXyFa_r1nxA) 

## Dealing with external data

A goal for scientific publishing is automated reproducibility of analyses, which the Jupyter notebook excels at. But, more than that, it should be possible to efficiently reproduce the analysis with different data sets. This entails having **one point of access** to a data set within the notebook, rather than having copy-pasted data into variables, i.e. this:

```python
data = read_in_data('data_key')
variable1 = data.key1
variable2 = data.key2
...
```

rather than this:

```python
variable1 = 12345
variable2 = 'something'
...
```

The best-practice for accessing heirarchical data (in my opinion) is to use the JSON format (as long as the data isn't [relational](http://www.sarahmei.com/blog/2013/11/11/why-you-should-never-use-mongodb/)), because it is:

- applicable for any data structure
- lightweight and easy to read and edit
- has a simple read/write mapping to python objects (using [json](https://docs.python.org/3.6/library/json.html))
- widely used (especially in web technologies)

A good way to store multiple bits of JSON data is in a [mongoDB](https://docs.mongodb.com/manual/administration/install-community/) and accessing it via [pymongo](https://api.mongodb.com/python/current/). This will also make it easy to move all the data to a cloud server at a later time, if required.

    conda install pymongo

But, if the data is coming from files output from different simulation or experimental code, where the user has no control of the output format. Then writing JSON parsers may be the way to go, and this is where [jsonextended](https://github.com/chrisjsewell/jsonextended) comes in, which implements:

- a lightweight plugin system to define bespoke classes for parsing different file extensions and data types.
- a 'lazy loader' for treating an entire directory structure as a nested dictionary.

For example:

```python
from jsonextended import plugins, edict
plugins.load_plugins_dir('path/to/folder_of_parsers','parsers')
data = edict.LazyLoad('path/to/data')
variable1 = data.folder1.file1_json.key1
variable2 = data[['folder1','file1.json','key2']]
variable3 = data[['folder1','file2.csv','key1']]
variable4 = data[['folder2','subfolder1','file3.other','key1']]
...    
```

If you are dealing with numerical data arrays which are to large to be loaded directly in to memory, 
then the [h5py](http://docs.h5py.org/en/latest/index.html) interface to the [HDF5](http://hdfgroup.org/) binary data format,
allows for the manipultion of even multi-terabyte datasets stored on disk, as if they were real NumPy arrays. 
These files are also supported by [jsonextended](https://github.com/chrisjsewell/jsonextended) lazy loading.

## Miscellaneous

I also use the Firefox Split Pannel extension to view the {name}_viewpdf.html page and monitor changes to the pdf.

[bookbook](https://github.com/takluyver/bookbook) is another package with some conversion capabilities.

## Acknowledgements

I took strong influence from:

- [Julius Schulz](http://blog.juliusschulz.de/blog/ultimate-ipython-notebook)
- [Dan Mackinlay](https://livingthing.danmackinlay.name/jupyter.html)
- Notebook concatination was adapted from [nbconvert issue#253](https://github.com/jupyter/nbconvert/issues/253)


