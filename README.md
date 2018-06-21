# Concept Sequence Tagging for a Movie Domain
First project of Language Understanding Systems, Fall Semester 2017, University of Trento.

The project consist in building a model for concept sequence tagging for a Movie Domain.

The task is described in the [task](tasdk.pdf) file

The results are discussed in the [report](report.pdf) file.

## Repository Structure
This repository contains:
 - three folders, `code`, `data`, `result`; 
 - two .pdf, `task` and `report`; and
 - two .md files, `README` and `LICENCE`
 
In the `task` pdf is reported the task of the assignment from the professor, in the report my explain of what I did, how and, with which results.

In the `data` folder is reported the dataset used. The dataset is divided in train and test and for each of them the `.data` file contains the tuple, (word concepts) and the `.feats.txt` file contains the tuples (word, Pos, Lemma).
 
In the `code` folder is contained the `main.py` file, in which is contained the implementation of the algorithm discussed in the `report.pdf`; in the `sandbox` folder the main will dump all the working file; and the `conlleval.pl` script that allows the evaluation of the performances of the algorithm automatically.

Finally in the `result` folder will be written the `performances.txt` 

## Running the algorithm

The [code/main.py](code/main.py) can be launched as is without setting any parameters since a working configurations is already given by default. 

```bash
cd code
chmod +x conlleval.pl
python2.7 main.py
```

Please note that the default configuration will clean up the sandbox when done.

For any further information about eh working of the algorithm do not hesitate to contact me or read the [report.pdf](report.pdf)
