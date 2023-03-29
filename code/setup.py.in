#!/usr/bin/env python

import setuptools

with open("README.md", "rb") as fh:
    long_description = fh.read().decode('utf-8')
    
# See https://packaging.python.org/tutorials/packaging-projects/ for details
setuptools.setup(
    name="debuggingbook",
    version="1.1.2",
    author="Andreas Zeller",
    author_email="zeller@cispa.de",
    description="Code for 'The Debugging Book' (https://www.debuggingbook.org/)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.debuggingbook.org/",
    packages=['debuggingbook', 'debuggingbook.bookutils'],
    python_requires='>=3.9',  # Mostly for ast.unparse()
    # packages=setuptools.find_packages(),
    
    # From requirements.txt
    install_requires=[
        'beautifulsoup4>=4.9.3',
        'diff_match_patch>=20200713',
        'easyplotly>=0.1.3',
        'enforce>=0.3.4',
        'fuzzingbook>=0.8.1',
        'graphviz>=0.14.2',
        'ipython>=7.16.1',
        'lxml>=4.5.1',
        'Markdown>=3.3.4',
        'matplotlib>=3.3.2',
        'mypy>=0.800',
        'nbconvert>=6.0.7',
        'nbformat>=5.0.8',
        'networkx>=2.5',
        'numpy>=1.16.5',
        'pydriller>=2.3',
        'pyparsing==2.4.7',  # newer versions conflict with bibtexparser
        'Pygments>=2.7.1',
        'python-magic>=0.4.18',
        'scikit_learn>=0.23.2',
        'selenium>=3.141.0',
        'showast>=0.2.4',
        'types-Markdown>=3'
    ],

    # See https://pypi.org/classifiers/
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Debuggers",
        "Topic :: Software Development :: Testing",
        "Topic :: Education :: Testing",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Security"
    ],
)
