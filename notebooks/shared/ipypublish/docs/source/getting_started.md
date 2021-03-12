# Getting Started

## Installation

Using [Conda](https://conda.io/docs/) is recommended for package management, 
in order to create self contained environments with specific versions of packages. 
The main external packages required are the Jupyter notebook and [Pandoc](http://pandoc.org) (for conversion between file formats):

	conda create --name ipyreport -c conda-forge jupyter pandoc
	
ipypublish can then be installed into this environment:

	source activate ipyreport
	pip install ipypublish
	
For converting to PDF, the TeX document preparation ecosystem is required, 
in particular [latexmk](http://mg.readthedocs.io/latexmk.html)), which can be installed from:

- Linux: [TeX Live](http://tug.org/texlive/)
- macOS (OS X): [MacTeX](http://tug.org/mactex/)
- Windows: [MikTex](http://www.miktex.org/)

For helpful extensions to the notebooks core capabilities, see the 
[Jupiter Notebook Extensions package](http://jupyter-contrib-nbextensions.readthedocs.io/en/latest/):

	conda install --name ipyreport jupyter_contrib_nbextensions
	
Additionally, a more extensive setup of useful packages (used to create the examples) 
are listed in [conda_packages.txt](https://github.com/chrisjsewell/ipypublish/blob/master/conda_packages.txt) 
and an environment can be created directly from this file using conda:

	conda create --name ipyreport -c conda-forge -c matsci --file conda_packages.txt

## Basic Conversion

The **nbpublish** script handles parsing the notebooks to nbconvert, with the appropriate converter.

    nbpublish -h
    nbpublish -pdf -f latex_ipypublish_nocode path/to/notebook.ipynb

For a more detailed explanation see the [Controlling Content Output](content_output.html) section.
    
The **nbpresent** script handles serving [reveal.js](http://lab.hakim.se/reveal-js/#/) slides to a webbrowser.

    nbpresent -h
	nbpresent -f slides_ipypublish_nocode path/to/notebook.ipynb
	
Note that, for offline use, simply download the latest version of reveal.js [here](https://github.com/hakimel/reveal.js/releases), 
rename the entire folder to reveal.js and place it in the same folder as the converted .slides.html file. 
The slides can also be save to PDF my appending `pdf-export` to the url (see [here](https://github.com/hakimel/reveal.js#pdf-export) for details).

## Troubleshooting

For installation issues, [Travis CI](https://en.wikipedia.org/wiki/Travis_CI) is used to automatically test updates against 
python 2.7 and 3.6, for both Linux and OSX, Therefore, to troubleshoot any installation/run issues, 
it is best to first look at the [travis config](https://github.com/chrisjsewell/ipypublish/blob/master/.travis.yml) 
and [travis test runs](https://travis-ci.org/chrisjsewell/ipypublish) for working configurations.

For conversion issues, for both `nbpublish` and `nbpresent`, detailed log messages of the run are output to 
both the console and file (default path: converted/notebook_name.nbpub.log). 
To debug conversions, use the `--log-level debug` and `--pdf-debug` flags. If there is still an error, please raise an
issue on the [GitHub repository](https://github.com/chrisjsewell/ipypublish/issues), including the run environment and
the log file.

# Examples

See [Example.ipynb](example/notebooks/Example.ipynb), [Example.pdf](https://chrisjsewell.github.io/ipypublish/Example.view_pdf.html),
[Example.html](https://chrisjsewell.github.io/ipypublish/Example.html) and 
[Example.slides.html](https://chrisjsewell.github.io/ipypublish/Example.slides.html#/) 
for an example of the potential input/output.

Or, for a practical example of the ipypublish capability, see these documents on Atomic 3D Visualisation: 
[Notebook](https://github.com/chrisjsewell/chrisjsewell.github.io/blob/master/3d_atomic/3D%20Atomic%20Visualisation.ipynb), 
[PDF](https://chrisjsewell.github.io/3d_atomic/converted/3D%20Atomic%20Visualisation.view_pdf.html), 
[HTML](https://chrisjsewell.github.io/3d_atomic/converted/3D%20Atomic%20Visualisation.html) 
or [Reveal.JS slideshow](https://chrisjsewell.github.io/3d_atomic/converted/3D%20Atomic%20Visualisation.slides.html).
