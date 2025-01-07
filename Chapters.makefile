# Debuggingbook Makefile
# This file defines the chapter files to be included

# Name of the project
PROJECT = debuggingbook

# Some metadata
BOOKTITLE = The Debugging Book
AUTHORS = Andreas Zeller
TWITTER = @Debugging_Book
MASTODON = @TheDebuggingBook

# Chapters to include in the book, in this order.
# * Chapters in `..._PART` get published.
# * Chapters in `..._PART_READY` only get published as beta, with a disclaimer.
# * Chapters in `..._PART_TODO` only get published as beta, with a disclaimer.
#      and a "todo" (wrench) marker in the menu.

# Chapter(s) to be marked as "new" in menu
NEW_CHAPTERS = \
	Alhazen.ipynb

# Introduction
INTRO_PART = \
	01_Intro.ipynb \
	Tours.ipynb \
	Intro_Debugging.ipynb
INTRO_PART_READY = 
INTRO_PART_TODO =

# Observing Executions
OBSERVING_PART = \
	02_Observing.ipynb \
	Tracer.ipynb \
	Debugger.ipynb \
	Assertions.ipynb
OBSERVING_PART_READY = 
OBSERVING_PART_TODO = 

# Flows and Dependencies
DEPENDENCIES_PART = \
	03_Dependencies.ipynb \
	Slicer.ipynb
DEPENDENCIES_PART_READY =
DEPENDENCIES_PART_TODO = 

# Simplifying Failures
EXPERIMENTING_PART = \
	04_Reducing.ipynb \
	DeltaDebugger.ipynb \
	ChangeDebugger.ipynb
EXPERIMENTING_PART_READY = 
EXPERIMENTING_PART_TODO =
	# ThreadDebugger.ipynb

# Abstracting Failures
ABSTRACTING_PART = \
	05_Abstracting.ipynb \
	StatisticalDebugger.ipynb \
	DynamicInvariants.ipynb \
	DDSetDebugger.ipynb \
	Alhazen.ipynb \
	PerformanceDebugger.ipynb
ABSTRACTING_PART_READY =
ABSTRACTING_PART_TODO =

# Repairing Failures
REPAIRING_PART = \
	06_Repairing.ipynb \
	Repairer.ipynb
REPAIRING_PART_READY =
REPAIRING_PART_TODO =

# Debugging in the Large
IN_THE_LARGE_PART = \
	07_In_the_Large.ipynb \
	Tracking.ipynb \
	ChangeCounter.ipynb
IN_THE_LARGE_PART_READY =
IN_THE_LARGE_PART_TODO =

# Appendices for the book
APPENDICES = \
	99_Appendices.ipynb \
	ExpectError.ipynb \
	Timer.ipynb \
	Timeout.ipynb \
	ClassDiagram.ipynb \
	StackInspector.ipynb

# Additional notebooks for special pages (not to be included in distributions)
FRONTMATTER = \
	index.ipynb
EXTRAS = \
	ReleaseNotes.ipynb \
	Importing.ipynb \
	Guide_for_Authors.ipynb \
	Template.ipynb \
	404.ipynb \
	Time_Travel_Debugger.ipynb \
	Reducing_Code.ipynb \
	Repairing_Code.ipynb \
	Project_of_your_choice.ipynb \
	IllustratedCode.ipynb

# These chapters will show up in the "public" version
PUBLIC_CHAPTERS = \
	$(INTRO_PART) \
	$(OBSERVING_PART) \
	$(DEPENDENCIES_PART) \
	$(EXPERIMENTING_PART) \
	$(ABSTRACTING_PART) \
	$(REPAIRING_PART) \
	$(IN_THE_LARGE_PART)

# These chapters will show up in the "beta" version
CHAPTERS = \
	$(INTRO_PART) \
	$(INTRO_PART_READY) \
	$(INTRO_PART_TODO) \
	$(OBSERVING_PART) \
	$(OBSERVING_PART_READY) \
	$(OBSERVING_PART_TODO) \
	$(DEPENDENCIES_PART) \
	$(DEPENDENCIES_PART_READY) \
	$(DEPENDENCIES_PART_TODO) \
	$(EXPERIMENTING_PART) \
	$(EXPERIMENTING_PART_READY) \
	$(EXPERIMENTING_PART_TODO) \
	$(ABSTRACTING_PART) \
	$(ABSTRACTING_PART_READY) \
	$(ABSTRACTING_PART_TODO) \
	$(REPAIRING_PART) \
	$(REPAIRING_PART_READY) \
	$(REPAIRING_PART_TODO) \
	$(IN_THE_LARGE_PART) \
	$(IN_THE_LARGE_PART_READY) \
	$(IN_THE_LARGE_PART_TODO)

READY_CHAPTERS = \
	$(INTRO_PART_READY) \
	$(OBSERVING_PART_READY) \
	$(DEPENDENCIES_PART_READY) \
	$(EXPERIMENTING_PART_READY) \
	$(ABSTRACTING_PART_READY) \
	$(REPAIRING_PART_READY) \
	$(IN_THE_LARGE_PART_READY)

TODO_CHAPTERS = \
	$(INTRO_PART_TODO) \
	$(OBSERVING_PART_TODO) \
	$(DEPENDENCIES_PART_TODO) \
	$(EXPERIMENTING_PART_TODO) \
	$(ABSTRACTING_PART_TODO) \
	$(REPAIRING_PART_TODO) \
	$(IN_THE_LARGE_PART_TODO)

## Specific settings

# No timeouts; debuggingbook/Tracing can take up to 15 minutes to render
EXECUTE_TIMEOUT = 900
TIME = time

# Default target
web:

# No type checking for IllustratedCode
mypy/.IllustratedCode.py.out:
	echo $(PY_SUCCESS_MAGIC) >> $@ 