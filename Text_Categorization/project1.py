import os
import sys
import numpy as np
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
#from sklearn import model_selection


def preprocess(train_doc, test_doc):
    X_train = []
    for each_doc in train_doc:
        with open(each_doc, "r") as file:
            X_train.append((each_doc, file.read()))

    X_test = []
    for each_doc in test_doc:
        with open(each_doc, "r") as file:
            X_test.append((each_doc, file.read()))

    stopwords = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone',
                 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount',
                 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around',
                 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before',
                 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both',
                 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de',
                 'describe', 'detail', 'did', 'do', 'does', 'doing', 'don', 'done', 'down', 'due', 'during', 'each',
                 'eg',
                 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every',
                 'everyone',
                 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first',
                 'five', 'for',
                 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give',
                 'go', 'had',
                 'has', 'hasnt', 'have', 'having', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein',
                 'hereupon',
                 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc',
                 'indeed',
                 'interest', 'into', 'is', 'it', 'its', 'itself', 'just', 'keep', 'last', 'latter', 'latterly', 'least',
                 'less',
                 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most',
                 'mostly',
                 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next',
                 'nine',
                 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on',
                 'once',
                 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over',
                 'own',
                 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed',
                 'seeming',
                 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty',
                 'so',
                 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such',
                 'system',
                 't', 'take', 'ten', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence',
                 'there',
                 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin',
                 'third', 'this',
                 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top',
                 'toward',
                 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was',
                 'we',
                 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas',
                 'whereby',
                 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole',
                 'whom',
                 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours',
                 'yourself',
                 'yourselves']
    # Building the Vocabulary of the words for the given documents
    vocab = {}
    for i in range(len(train_doc)):
        word_list = []
        for word in X_train[i][1].split():
            # remove any punctuation and change all the words into lower case character
            word_new = word.strip(string.punctuation).strip('\n').strip('\t').lower()
            #             # Using Potter Steemming
            #             ps = PorterStemmer()
            #             word_new = ps.stem(word_new)
            # Using Lemmatization
            wordnet_lemmatizer = WordNetLemmatizer()
            word_new = wordnet_lemmatizer.lemmatize(word_new)
            if (len(word_new) > 2) and (word_new not in stopwords):
                if word_new in vocab:
                    vocab[word_new] += 1
                else:
                    vocab[word_new] = 1
    num_words = [0 for i in range(max(vocab.values()) + 1)]
    frequency = [i for i in range(max(vocab.values()) + 1)]
    for key in vocab:
        num_words[vocab[key]] += 1

    cutoff_freq = 20
    num_words_above_cutoff = len(vocab) - sum(num_words[0:cutoff_freq])

    print("Number of words with frequency higher than cutoff frequency({}) :".format(cutoff_freq),
          num_words_above_cutoff)

    features = []
    for key in vocab:
        if vocab[key] >= cutoff_freq:
            features.append(key)

    X_train_dataset = np.zeros((len(X_train), len(features)))

    # The construction of the word vector
    for i in range(len(X_train)):
        word_list = [word.strip(string.punctuation).strip('\n').strip('\t').lower() for word in X_train[i][1].split()]
        for word in word_list:
            if word in features:
                X_train_dataset[i][features.index(word)] += 1

    X_test_dataset = np.zeros((len(X_test), len(features)))

    for i in range(len(X_test)):
        word_list = [word.strip(string.punctuation).strip('\n').strip('\t').lower() for word in X_test[i][1].split()]
        for word in word_list:
            if word in features:
                X_test_dataset[i][features.index(word)] += 1

    return X_train_dataset, X_test_dataset


# Implementing Naive Bayes from scratch
class NaiveBayesModel:

    def __init__(self):
        # count is a dictionary which stores several dictionaries corresponding to each category
        # each value in the subdictionary represents the freq of the key corresponding to that category
        self.table = {}

        self.categories = None

    def fit(self, X_train, Y_train):
        self.categories = set(Y_train)
        for cat in self.categories:
            self.table[cat] = {}
            for i in range(len(X_train[0])):
                self.table[cat][i] = 0
            self.table[cat]['total_words'] = 0
            self.table[cat]['total_counts'] = 0
        self.table['total_counts'] = len(X_train)

        for i in range(len(X_train)):
            for j in range(len(X_train[0])):
                self.table[Y_train[i]][j] += X_train[i][j]
                self.table[Y_train[i]]['total_words'] += X_train[i][j]
            self.table[Y_train[i]]['total_counts'] += 1

    def calculate_prob(self, test, category):
        # Avoid Numberical Underlow using Log probability
        total_words = len(test)
        # Log (A/B) = Log(A) - Log(B)
        log_prob = np.log(self.table[category]['total_counts']) - np.log(self.table['total_counts'])

        for i in range(len(test)):
            current_priori = test[i] * (
                        np.log(self.table[category][i] + 1) - np.log(self.table[category]['total_words'] + total_words))
            log_prob += current_priori

        return log_prob

    def predict_each(self, test):

        global_max = None
        initialized = True
        global_category = None

        # Prediction for each document
        for cat in self.categories:
            current_log_prob = self.calculate_prob(test, cat)
            if (initialized == True) or (current_log_prob > global_max):
                # Updating current best probability
                global_max = current_log_prob
                global_category = cat
                initialized = False
        return global_category

    def predict(self, X_test):
        Y_prediction = []
        for i in range(len(X_test)):
            Y_prediction.append(self.predict_each(X_test[i]))
        return Y_prediction

    def score(self, Y_pred, Y_test):
        count = 0
        print(len(Y_pred))
        print(len(Y_test))
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y_test[i]:
                count += 1
        accuracy = count / len(Y_pred)
        return accuracy


def main():
    if len(sys.argv) < 2:
        print('Please specify the path to be listed')
        sys.exit()

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    if not os.path.isfile(test_path) or not os.path.isfile(train_path):
        print("The file you specify does not exit !")
        sys.exit()

    train_doc = []
    train_cat = []
    test_doc = []
    Y_test = []

    with open(train_path, "r") as file:
        for line in file:
            line = line.strip()
            doc, category = line.split(' ')
            train_doc.append(doc)
            train_cat.append(category)

    # Testing corpus 2 and corpus 3 accuracy using model selection split
    # if corpus2 in train_path or corpus3 in train_path:
    #     train_doc, test_doc, train_cat, Y_test = model_selection.train_test_split(train_doc, train_cat, test_size=0.2,
    #                                                                               random_state=0)

    # with open(corpus1_path, "r") as file:
    #     for line in file:
    #         line = line.strip()
    #         doc, category = line.split(' ')
    #         Y_test.append(category)

    with open(test_path, "r") as file:
        for line in file:
            line = line.strip('\n')
            test_doc.append(line)

    X_train, X_test = preprocess(train_doc, test_doc)

    model = NaiveBayesModel()
    print("Start Training...")
    model.fit(X_train, train_cat)
    print("Predicting...")
    Y_test_pred = model.predict(X_test)
   # our_score_test = model.score(Y_test_pred, Y_test)
    output_path = input('Enter a output file path here:')
    with open(output_path, "w") as file:
        for i in range(len(test_doc)):
            file.write('%s %s\n' % (test_doc[i], Y_test_pred[i]))
    print("Output Done")
    # For my own evaluation
    # print("Our score on testing data :", our_score_test)
    # print("Classification report for testing data :-")
    # print(classification_report(Y_test, Y_test_pred))


if __name__ == "__main__":
    main()