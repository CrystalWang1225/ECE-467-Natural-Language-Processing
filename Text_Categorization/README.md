## Text categorization Project

##### Professor Carl Sable
##### Yuecen Wang

#### Introduction
This is the text categorization project using Na&#x00EF;ve Bayes approach based on Bayes' Theorem. In this project, I have read and then stored the test and training file into list in python for tokenizing training and test files. Psuedo-count smoothing is used here.

#### Usage
```
python project1.py [TRAINING_PATH] [TEST_PATH]
```
The training_path should contain the list of labeled training documents, the test path should contain the list of testing documents.
After the prediction has made, the program will ask the user to specify the output name

### Testing phase
The Perl script can compare the output of tet categorization to a file with the actual labels for the set

### Requirement
* nltk
* scikit-learn

