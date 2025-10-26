#!/usr/bin/env python3
# Patch Jupyter Book HTML

import argparse
import os.path
import sys

example_content = """
<div class="dropdown dropdown-download-buttons">
  <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false" aria-label="Download this page">
    <i class="fas fa-download"></i>
  </button>
  <ul class="dropdown-menu">
      
      
      
      <li><a href="../_sources/sphinx/index.md" target="_blank"
   class="btn btn-sm btn-download-source-button dropdown-item"
   title="Download source file"
   data-bs-placement="left" data-bs-toggle="tooltip"
>
  

<span class="btn__icon-container">
  <i class="fas fa-file"></i>
  </span>
<span class="btn__text-container">.md</span>
</a>
</li>
      
      
      
      
      <li>
<button onclick="window.print()"
  class="btn btn-sm btn-download-pdf-button dropdown-item"
  title="Print to PDF"
  data-bs-placement="left" data-bs-toggle="tooltip"
>
  

<span class="btn__icon-container">
  <i class="fas fa-file-pdf"></i>
  </span>
<span class="btn__text-container">.pdf</span>
</button>
</li>
      
  </ul>
</div>
"""

# We are at
# https://www.fuzzingbook.org/html/Fuzzer.html
# We refer to
# https://www.fuzzingbook.org/code/Fuzzer.py

def convert(content, basename="index"):
    # Replace "Download source file"
    content = content.replace('title="Download source file"', 'title="Download notebook"')

    # Add .py download button at top
    start = content.find('dropdown-download-buttons')
    start = content.find('<ul class="dropdown-menu">', start)
    start = content.find('\n', start)
    
    py_ref = f"../code/{basename}.py"
    
    py_button_html = f'''
    <li><a href="{py_ref}" target="_blank"
       class="btn btn-sm btn-download-source-button dropdown-item"
       title="Download Python code"
       data-bs-placement="left" data-bs-toggle="tooltip"
    >
  

    <span class="btn__icon-container">
      <i class="fas fa-file"></i>
      </span>
    <span class="btn__text-container">.py</span>
    </a>
    </li>
    '''
    
    content = content[:start] + py_button_html + content[start:]

    # Add "all downloads" download button at end of menu
    start = content.find('</ul>', start)
    
    importing_ref = "Importing.html"
    py_downloads_html = f'''
    <li><a href="{importing_ref}" target="_blank"
       class="btn btn-sm btn-download-source-button dropdown-item"
       title="All downloads"
       data-bs-placement="left" data-bs-toggle="tooltip"
    >
  

    <span class="btn__icon-container">
      <i class="fas fa-file"></i>
      </span>
    <span class="btn__text-container">All downloads</span>
    </a>
    </li>
    '''

    content = content[:start] + py_downloads_html + content[start:]
    
    return content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='*', help="jupyter-book HTML files")
    
    args = parser.parse_args()
    
    if len(args.files) == 0:
        print(convert(example_content))
        sys.exit(0)

    for html_file in args.files:
        content = open(html_file).read()
        if 'title="All downloads"' in content:
            print(f'{html_file}: already patched')
            continue

        basename = os.path.splitext(os.path.basename(html_file))[0]
        new_content = convert(content, basename=basename)
        # open(html_file + '~', 'w').write(content)
        open(html_file, 'w').write(new_content)
        print(f'{html_file}: patched')

        
        
    
        
