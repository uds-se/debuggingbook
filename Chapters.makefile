# Debuggingbook Makefile
# This file defines the chapter files to be included

# Name of the project
PROJECT = debuggingbook

# Some metadata
BOOKTITLE = The Debugging Book
AUTHORS = Andreas Zeller
TWITTER = @Debugging_Book

# Where the shared files are
SHARED = ../fuzzingbook/

# Chapter(s) to be marked as "new" in menu
NEW_CHAPTERS = 

# Chapters to include in the book, in this order

# Introduction
INTRO_PART = 
INTRO_PART_READY =
INTRO_PART_TODO = \
	01_Intro.ipynb \
	Tours.ipynb \
	Intro_Debugging.ipynb

# Tracking Failures (also: where the bugs are)
TRACKING_PART = 
TRACKING_PART_READY =
TRACKING_PART_TODO = 
	# 02_Tracking.ipynb \
	# Tracking.ipynb

# Observing Executions
OBSERVING_PART = 
OBSERVING_PART_READY = 
OBSERVING_PART_TODO = \
	03_Observing.ipynb \
	Tracer.ipynb \
	Debugger.ipynb \
	Assertions.ipynb \
	Slicer.ipynb \
	StatisticalDebugger.ipynb

# Simplifying Failures
EXPERIMENTING_PART = 
EXPERIMENTING_PART_READY =
EXPERIMENTING_PART_TODO = \
	04_Inputs.ipynb \
	DeltaDebugger.ipynb \
	ChangeDebugger.ipynb
	# Grammars.ipynb \
	# GrammarReducer.ipynb

# Mining Specifications
ABSTRACTING_PART =
ABSTRACTING_PART_READY =
ABSTRACTING_PART_TODO = 
	# 05_Abstracting.ipynb \
	# DynamicInvariants.ipynb \
	# DDSet.ipynb \
	# Alhazen.ipynb
	
# Repairing Failures
REPAIRING_PART =
REPAIRING_PART_READY =
REPAIRING_PART_TODO = 
	# 05_Repairing.ipynb \
	# Repair.ipynb

# Appendices for the book
APPENDICES = \
	99_Appendices.ipynb \
	ExpectError.ipynb \
	Timer.ipynb \
	ClassDiagram.ipynb \
	ControlFlow.ipynb \
	RailroadDiagrams.ipynb

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
	Reducing_Code.ipynb
	
# These chapters will show up in the "public" version
PUBLIC_CHAPTERS = \
	$(INTRO_PART) \
	$(TRACKING_PART) \
	$(OBSERVING_PART) \
	$(EXPERIMENTING_PART) \
	$(ABSTRACTING_PART) \
	$(REPAIRING_PART)

# These chapters will show up in the "beta" version
CHAPTERS = \
	$(INTRO_PART) \
	$(INTRO_PART_READY) \
	$(INTRO_PART_TODO) \
	$(TRACKING_PART) \
	$(TRACKING_PART_READY) \
	$(TRACKING_PART_TODO) \
	$(OBSERVING_PART) \
	$(OBSERVING_PART_READY) \
	$(OBSERVING_PART_TODO) \
	$(EXPERIMENTING_PART) \
	$(EXPERIMENTING_PART_READY) \
	$(EXPERIMENTING_PART_TODO) \
	$(ABSTRACTING_PART) \
	$(ABSTRACTING_PART_READY) \
	$(ABSTRACTING_PART_TODO) \
	$(REPAIRING_PART) \
	$(REPAIRING_PART_READY) \
	$(REPAIRING_PART_TODO)

READY_CHAPTERS = \
	$(INTRO_PART_READY) \
	$(TRACKING_PART_READY) \
	$(OBSERVING_PART_READY) \
	$(EXPERIMENTING_PART_READY) \
	$(ABSTRACTING_PART_READY) \
	$(REPAIRING_PART_READY)

TODO_CHAPTERS = \
	$(INTRO_PART_TODO) \
	$(TRACKING_PART_TODO) \
	$(OBSERVING_PART_TODO) \
	$(EXPERIMENTING_PART_TODO) \
	$(ABSTRACTING_PART_TODO) \
	$(REPAIRING_PART_TODO)
