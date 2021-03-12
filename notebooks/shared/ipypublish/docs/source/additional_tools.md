# Additional Tools

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

The [Reveal.js - Jupyter/IPython Slideshow Extension (RISE)](https://github.com/damianavila/RISE) notebook extension 
offers rendering as a Reveal.js-based slideshow, where you can execute code or show to the audience whatever 
you can show/do inside the notebook itself! Click on the image to see a demo:

[![RISE Demo](https://img.youtube.com/vi/sXyFa_r1nxA/0.jpg)](https://www.youtube.com/watch?v=sXyFa_r1nxA) 

## Working With External Data

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
