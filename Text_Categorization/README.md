## Text categorization Project

##### Professor Carl Sable
##### Yuecen Wang

#### Introduction
This is the text categorization project using Na&#x00EF;ve Bayes approach based on Bayes' Theorem. In this project, I have read and then stored the test and training file into list in python for tokenizing training and test files. Psuedo-count smoothing is used here. I experienced with POS tagging, lemmatization, case sensitivity, stop lists. For my program, stop lists has made a significant difference. POS tagging actually made my program worse in terms of the overall accuracy. In the end, I chose to use all lower cases, lemmatization, and a stop list for the final system. 

#### Usage
```
python project1.py [TRAINING_PATH] [TEST_PATH]
```
The training_path should contain the list of labeled training documents, the test path should contain the list of testing documents.
After the prediction has made, the program will ask the user to specify the output name.

### Testing phase
The Perl script can compare the output of text categorization to a file with the actual labels for the set. I have also written a scoring function inside of the model class. It calculates the accuracy - the number of correct predictions / All predictions. I have also used the tools of scikit-learn model_selection that outputs a similar confusion matrix as the Perl script.For second and the third data sets that lack actual testing labels, I split the whole dataset into a 8:2 training and testing set. The program has a higher accuracy for corpus3 compared to corpus2, a binary dataset. 

### Requirement
* nltk
* scikit-learn

