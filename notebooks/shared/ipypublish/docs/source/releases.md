# Releases

## v0.10.9 - Improve ipubpandoc filter: conversion of equations 

- Ensure equations that are already wrapped in a math environment are not wrapped twice. 
- For RST output, ensure multiline equations are correctly indented 

## v0.10.8 - Fix Binder Environment 

Up-date the Conda requirements to working versions 

## v0.10.7 - Add sphinx options to add buttons for toggling all input/output cells 

 

## v0.10.6 - Added option to toggle show/hide code cells with Sphinx 

 

## v0.10.5 - Remove restriction on Sphinx<2.0 

Changes include: 
 
- implement `HTML2JSONParser` and `pytest-regressions` for testing HTML documents 
 
Rather then searching for specific bits of text, we convert the document to a json object, stripping out sections irrelevant for testing (like the header, footer and scripts that can change between versions). and test with the `data_regression` fixture 
 
- Add pre_commit configuration 
- Update .travis.yml to test python 3.7 rather than 3.4 (which is deprecated) 
- Update RTD configuration 
- version bump 
 

## v0.10.4 - Fix image reference clashes in rst 

 

## v0.10.3 - Minor Improvements to `ipypublish.sphinx.notebook` 

- remove `sphinx.ext.imgconverter` from sphinx auto-builds 
- add additional known sphinx roles 

## v0.10.2 - Update Requirements 

- Only require backport dependencies for python version older than their implementation 
- use `ordered-set`, instead of `oset` dependency, since it is better maintained 

## v0.10.1 - Minor Improvements to `ipypublish.sphinx.notebook` 

- Formatting of the execution_count is now inserted by: `ipysphinx_input_prompt.format(count=execution_count)` 
- use "Code Cell Output" as placeholder for output image caption 

## v0.10.0 -  Add Sphinx extension for glossary referencing: `ipypublish.sphinx.gls` 

- Added Sphinx extension for glossary referencing: `ipypublish.sphinx.gls`. 
  See :ref:`sphinx_ext_gls` 
 
- Added `ConvertBibGloss` post-processor, 
  to convert a bibglossary to the required format 
 
- Added notebook-level metadata options for `bibglossary` and `sphinx` 
  (see :ref:`meta_doclevel_schema`) 
 
- Large refactoring and improvements for test suite, particularly for testing 
  of Sphinx extensions (using the Sphinx pytest fixtures) and creation of the 
  `IpyTestApp` fixture 
 
- fixes #71  
 
Back-compatibility breaking changes: 
 
- renamed Sphinx notebook extension from 
  `ipypublish.ipysphinx` to `ipypublish.sphinx.notebook` 
  (see :ref:`sphinx_ext_notebook`) 
 
- `ipypublish.postprocessors.base.IPyPostProcessor.run_postprocess` 
  input signature changed (and consequently it has changes for all post-processors) 
 
`v0.9`: 
 
```python 
   def run_postprocess(self, stream, filepath, resources): 
      output_folder = filepath.parent 
``` 
 
`v0.10`: 
 
``` python 
   def run_postprocess(self, stream, mimetype, filepath, resources): 
      output_folder = filepath.parent 
``` 

## v0.9.4 - Bug Fix 

Bug fix for widefigures 
(see `issue <https://github.com/chrisjsewell/ipypublish/issues/68>`_), 
thanks to @katie-jones 
 
fixes #68  

## v0.9.3 - Added sdist to pypi release (for use by Conda) 

 

## v0.9.2 - minor bug fix 

remove blank line between: 
 
   .. nboutput:: rst 
       :class: rendered_html 

## v0.9.1 - minor bug fix 

- fix newline between directive and options 

## v0.9.0 - Major Improvements 

- Added ``ipubpandoc`` (see :ref:`markdown_cells`) 
- Refactored conversion process to 
  :py:class:`ipypublish.convert.main.IpyPubMain` configurable class 
- Added postprocessors (see :ref:`post-processors`) 
- Added Sphinx extension (see :ref:`sphinx_extension`) 
- Added Binder examples to documentation (see :ref:`code_cells`) 

## v0.8.3 - Handle Cell Attachments 

Images can also be embedded in the notebook itself. Just drag an image file into the Markdown cell you are just editing or copy and paste some image data from an image editor/viewer. 
 
