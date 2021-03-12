# Metadata Tags

## Introduction

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

## Document Level

### Language 

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

### Bibliography

To specify where the **bibliography** is:

```json
{
"ipub": {
	"bibliography" : "path/to/bibliograph.bib"
	}
}
```

The path can be absolute or relative.

### Title Page

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

### Contents Tables

To control the output of **contents tables**:

```json
{
"ipub": {
  "toc": true,
  "listfigures": true,
  "listtables": true,
  "listcode": true
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

## Cell/Output Level

### Ignore

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
### Code Block

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

### Output Text

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

### Output Figures

For **figures** (i.e. any graphics output by the code), enter in cell metadata:

```json
{
"ipub": {
  "figure": {
    "caption": "Figure caption.",
    "label": "fig:flabel",
    "placement": "H",
	"height":0.4,
    "widefigure": false
    }
  }
}
```

- all tags are optional
- height/width correspond to the fraction of the page height/width, only one should be used (aspect ratio will be maintained automatically)
- `placement` is optional and constitutes using a placement arguments for the figure (e.g. \begin{figure}[H]). See [Positioning_images_and_tables](https://www.sharelatex.com/learn/Positioning_images_and_tables).
- `widefigure` is optional and constitutes expanding the figure to the page width (i.e. \begin{figure*}) (placement arguments will then be ignored)

### Output Tables

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

### Output Equations

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

### Controlling Slides

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
df = pd.DataFrame(np.random.random((3, 3)))
latex = df.to_latex(index=False)
html = df.to_html(index=False)
display({'text/latex': latex,
         'text/html': html}, raw=True)
         
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
