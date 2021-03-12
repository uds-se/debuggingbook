# Controlling Content Output

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

## The IPyPublish Defaults

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

## Creating a Bespoke Converter

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
