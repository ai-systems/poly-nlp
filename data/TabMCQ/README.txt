------------
TabMCQ v 1.0
------------

-------
SUMMARY
-------
This package contains a copy of the Aristo Tablestore (Nov 2015 Snapshot) from the Allen Institute for Artificial Intelligence (http://allenai.org/content/data/AristoTablestore-Nov2015Snapshot.zip), which is a collection of curated facts in the form of tables. These facts are collected using a mixture of manual and semi-automated techniques. The package additionally contains a large set of crowd-sourced multiple-choice questions covering the facts in the tables. Through the setup of the crowd-sourced annotation task, implicit alignment information between questions and tables is also collected and included in this package. For further information, see "TabMCQ: A Dataset of General Knowledge Tables and Multiple-choice Questions" (pdf included in this package)

-------
LICENCE
-------
This package is distributed under the Creative Commons Attribution-ShareAlike 4.0 International License (http://creativecommons.org/licenses/by-sa/4.0/legalcode)

This means you are free to:
1) Share — copy and redistribute the material in any medium or format
2) Adapt — remix, transform, and build upon the material
for any purpose, even commercially.
The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:
1) Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
2) ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
3) No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

----------------
PACKAGE CONTENTS
----------------
The contents of this package are the following:

- MCQs.tsv: a tab-seprated file containing the Multiple-Choice Questions and their alignment information to tables.
- README.txt: this file.
- TableIndex.xlsx: a spreadsheet containing a topical index of all the tables.
- Tables: a folder containing all the tables, sub-divided into three categories.
- TabMCQ.pdf: a report containing information on the dataset and how it was created.

-------------------
MCQS AND ALIGNMENTS
-------------------
The MCQs and their alignments are contained in the file MCQs.tsv. The file is tab-separated and its header is explanatory of the various data columns. Here's a summary:

- QUESTION: the actual text of the question.
- QUESTION-ALIGNMENT: the cells in the aligned row (RELEVANT TABLE + RELEVANT ROW; see below) that were used to construct the question (0 indexed).
- CHOICE 1 through 4: the different choices for the question.
- CORRECT CHOICE: the correct answer to the question (1 indexed)
- RELEVANT TABLE: the table needed to answer this question (see TableIndex.xlsx for mapping).
- RELEVANT ROW: the row in the RELEVANT TABLE needed to answer the question (0 indexed, including header).
- RELEVANT COL: the column in the RELEVANT ROW of the RELEVANT TABLE containing the answer to the question (0 indexed).

An additional piece of potential alignment information is left unmentioned because it is implicit from the existing alignments. These are the header-cell alignments. For every MCQ if a cell (<table>,<row>,<col>) is a relevant cell -- either because it contains the answer, or was used to construct the question -- then (<table>,<header>,<col>) MIGHT also be relevant.

-----------
TABLE INDEX
-----------
The table index contains a mapping between table names and their topics. It also lists the number of rows (including header) for each table. The tables are divided into 3 broad categories:

1) REGENTS - Tables constructed manually from studying the REGENTS MCQ dataset (http://allenai.org/content/data/Regents.zip).
2) MONARCH - Tables constructed manually from studying an additional in-house MCQ dataset at AI2.
3) AUTO - Tables constructed semi-automatically.

The table names match the ones found in the MCQs.tsv file, and are an index into the Tables/ folder. The part of the name before the dash refers to the subfolder of Tables/ in which a specific table is to be found, while the part after refers to the number in that subfolder.

It may be noted that, unlike REGENTS and MONARCH tables, the AUTO tables are not topically divided. They contain a heterogenous mix of data and are rather divided on the bsis of how they were generated and from what source of information.

------
TABLES
------
The tables are divided into subfolders as per the description in TABLE INDEX. Each table contains a header row followed by a number of fact rows. Each fact row is a sentence. However a table contains recurring filler patterns that divide each fact sentence into meaningful columns. These rows and columns are the ones indexed into by the alignment data in MCQs.tsv.

---------
REFERENCE
---------
If you use this data in your research please cite:

Sujay Kumar Jauhar, Peter Turney and Eduard Hovy. (2016). TabMCQ: A Dataset of General Knowledge Tables and Multiple-choice Questions. arXiv preprint arXiv:1602.03960 [cs.CL].