# Debuggingbook Makefile
# This file defines the chapter files to be included

# Name of the project
PROJECT = debuggingbook

# Chapter(s) to be marked as "new" in menu
NEW_CHAPTERS = 

# Chapters to include in the book, in this order
INTRO_PART = 
INTRO_PART_READY =
INTRO_PART_TODO = \
	01_Intro.ipynb \
	Tours.ipynb \
	Intro_Debugging.ipynb

INPUTS_PART = 
INPUTS_PART_READY = \
	02_Inputs.ipynb \
	Grammars.ipynb \
	GrammarFuzzer.ipynb
	Reducer.ipynb \
INPUTS_PART_TODO = \
	DDSet.ipynb \
	Alhazen.ipynb
	
CODE_PART =
CODE_PART_READY =
CODE_PART_TODO = \
	03_Code.ipynb \
	Checkers.ipynb \
	DynamicInvariants.ipynb \
	Statistical.ipynb \
	Slices.ipynb \
	Repair.ipynb

MANAGEMENT_PART = 
MANAGEMENT_PART_READY = 
MANAGEMENT_PART_TODO = \
	04_Managing.ipynb \
	Tracker.ipynb \
	WhereTheBugsAre.ipynb

# Appendices for the book
APPENDICES = \
	99_Appendices.ipynb \
	ExpectError.ipynb \
	Timer.ipynb \
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
	404.ipynb
	
# These chapters will show up in the "public" version
PUBLIC_CHAPTERS = \
	$(INTRO_PART) \
	$(INPUTS_PART) \
	$(CODE_PART) \
	$(MANAGEMENT_PART)

# These chapters will show up in the "beta" version
CHAPTERS = \
	$(INTRO_PART) \
	$(INTRO_PART_READY) \
	$(INTRO_PART_TODO) \
	$(INPUTS_PART) \
	$(INPUTS_PART_READY) \
	$(INPUTS_PART_TODO) \
	$(CODE_PART) \
	$(CODE_PART_READY) \
	$(CODE_PART_TODO) \
	$(MANAGEMENT_PART) \
	$(MANAGEMENT_PART_READY) \
	$(MANAGEMENT_PART_TODO)
	
READY_CHAPTERS = \
	$(INTRO_PART_READY) \
	$(INPUTS_PART_READY) \
	$(CODE_PART_READY) \
	$(MANAGEMENT_PART_READY)

TODO_CHAPTERS = \
	$(INTRO_PART_TODO) \
	$(INPUTS_PART_TODO) \
	$(CODE_PART_TODO) \
	$(MANAGEMENT_PART_TODO)
