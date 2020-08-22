# Ghazal Sadeghian(9533054)-Project3-AI

import numpy as np
import random
import operator
import string

import bs4 as bs
import urllib.request
import re
import nltk
from nltk.tokenize import word_tokenize

################# Reading Test Data ###############
###################################################
foundLabels = []
labelfile = open('labels.txt', 'r')

tText = ""
labels = []
Lines = labelfile.readlines()
for i in Lines:
    i = i.lower()
    label = re.sub(r'[^A-Za-z.$]', '', i)
    labels.append(label)
labelfile.close()

################# Reading Test Data ###############
###################################################

testfile = open('Test_data.txt', 'r')

tText = ""
sentences = []
Lines = testfile.readlines()
del Lines[0]
for i in Lines:
    i = i.lower()
    sen = re.sub(r'[^A-Za-z.$ ]', '', i)
    sentences.append(sen)
testfile.close()

#################   Reading Train Data ###############
######################################################
trainfile = open('Train_data.txt', 'r')

trainText = ''
count = 0

for i in range(500):
    senT = trainfile.readline()
    senT = senT.lower()
    senT = re.sub(r'[^A-Za-z ]', '', senT)
    senT = "/s/ " + senT + " //s/" + " "
    trainText += senT
    # trainText += (trainfile.readline()) + " "

trainfile.close()

# trainText = trainText.lower()
# trainText = re.sub(r'[^A-Za-z. ]', '', trainText)
wordsTokens = word_tokenize(trainText)
################# TRIGRAM ###############
#########################################

trigram = {}
words = 2


def Trigram():
    global trigram
    global words
    for i in range(len(wordsTokens) - words):
        seq = ' '.join(wordsTokens[i:i + words])
        if seq not in trigram.keys():
            trigram[seq] = {}
        if wordsTokens[i + words] not in (trigram[seq]).keys():
            (trigram[seq])[wordsTokens[i + words]] = 1
        else:

            prev = (trigram[seq])[wordsTokens[i + words]]
            (trigram[seq])[wordsTokens[i + words]] = prev + 1

    for n in trigram.keys():
        countForEach = 0
        for j in trigram[n]:
            countForEach = 0
            for i in (trigram[n]).keys():
                countForEach = countForEach + (trigram[n])[i]

            for k in (trigram[n]).keys():
                # print(countForEach)
                (trigram[n])[k] = float((trigram[n])[k] / countForEach)

    # for printing trigram model of traning set just call Trigram function
    # print("Trigram Model:")
    # for n in trigram.keys():
    #
    #     for i in (trigram[n]).keys():
    #         print(i, "|", n, ":", (trigram[n])[i])


################# BIGRAM ###############
########################################

twograms = {}
duwords = 1


def Bigram():
    global twograms
    global duwords
    for i in range(len(wordsTokens) - duwords):
        duseq = ' '.join(wordsTokens[i:i + duwords])
        if duseq not in twograms.keys():
            twograms[duseq] = {}
        if wordsTokens[i + duwords] not in (twograms[duseq]).keys():
            (twograms[duseq])[wordsTokens[i + duwords]] = 1
        else:

            duprev = (twograms[duseq])[wordsTokens[i + duwords]]
            (twograms[duseq])[wordsTokens[i + duwords]] = duprev + 1

    for n in twograms.keys():
        countForEachdu = 0
        for j in twograms[n]:
            countForEachdu = 0
            for i in (twograms[n]).keys():
                countForEachdu = countForEachdu + (twograms[n])[i]

            for k in (twograms[n]).keys():
                # print(countForEach)
                (twograms[n])[k] = float((twograms[n])[k] / countForEachdu)

    # for printing bigram model of traning set just call Bigram function
    # print("Bigram Model:")
    # for n in twograms.keys():
    #
    #     for i in (twograms[n]).keys():
    #         print(i,"|",n,":",(twograms[n])[i])


################# UNIGRAM ###############
#########################################

unigram = {}
uniwords = 0


def Unigram():
    global unigram
    global uniwords
    for i in range(len(wordsTokens) - uniwords):
        uniseq = ' '.join(wordsTokens[i:i + uniwords])
        if uniseq not in unigram.keys():
            unigram[uniseq] = {}
        if wordsTokens[i + uniwords] not in (unigram[uniseq]).keys():
            (unigram[uniseq])[wordsTokens[i + uniwords]] = 1
        else:

            uniprev = (unigram[uniseq])[wordsTokens[i + uniwords]]
            (unigram[uniseq])[wordsTokens[i + uniwords]] = uniprev + 1

    for n in unigram.keys():
        countForEachuni = 0
        for j in unigram[n]:
            countForEachuni = 0
            for i in (unigram[n]).keys():
                countForEachuni = countForEachuni + (unigram[n])[i]

            for k in (unigram[n]).keys():
                # print(countForEach)
                (unigram[n])[k] = float((unigram[n])[k] / countForEachuni)

    # for printing unigram model of traning set just call Unigram function
    # print("Unigram Model:")
    # for n in unigram.keys():
    #
    #     for i in (unigram[n]).keys():
    #         print(i,":",(unigram[n])[i])


#################   Filling Test Data Sentences with Trigram    ###############
######################################################


# fillNum=0
def fill(foundTri, foundTwo):
    global twograms
    global trigram
    # global fillNum
    if foundTri not in trigram.keys():
        # if foundTwo not in twograms.keys():
            return "NOT FOUND"
        # possible_wordsdu = twograms[foundTwo]
        # return max(possible_wordsdu.items(), key=operator.itemgetter(1))[0]

    possible_words = trigram[foundTri]
    nextW = max(possible_words.items(), key=operator.itemgetter(1))[0]
    # fillNum+=1
    # nextW = weighted_random_by_dct(possible_words)
    # print("filNum:", fillNum)
    return nextW


def weighted_random_by_dct(dct):
    rand_val = random.random()
    total = 0
    for k, v in dct.items():
        total += v
        if rand_val <= total:
            return k


def fillSentencesTrigram():
    global sentences
    global foundLabels
    count = 1
    for se in sentences:
        tri = ""
        du = ""
        wordsInSen = se.lower().split()
        tri += wordsInSen[wordsInSen.index('$') - 2]
        tri += " "
        tri += wordsInSen[wordsInSen.index('$') - 1]
        du = wordsInSen[wordsInSen.index('$') - 1]
        filll = fill(tri, du)
        foundLabels.append(filll)
        print(count, ",", filll)
        count += 1


#################   Filling Test Data Sentences with Bigram ###############
######################################################


# fillNum=0
def fillBi(foundTwo):
    global twograms
    if foundTwo not in twograms.keys():
        return "NOT FOUND"
    possible_wordsdu = twograms[foundTwo]
    return max(possible_wordsdu.items(), key=operator.itemgetter(1))[0]


def fillSentencesBigram():
    global sentences
    global foundLabels
    count = 1
    for se in sentences:
        du = ""
        wordsInSen = se.lower().split()
        du = wordsInSen[wordsInSen.index('$') - 1]
        filll = fillBi(du)
        foundLabels.append(filll)
        print(count, ",", filll)
        count += 1


#


#################   Filling Test Data Sentences with Bigram ###############
######################################################

def fillSentencesUnigram():
    global sentences
    global unigram
    global foundLabels
    count = 1
    for se in sentences:
        du = ""
        wordsInSen = se.lower().split()
        filll = max(unigram[''].items(), key=operator.itemgetter(1))[0]
        foundLabels.append(filll)
        print(count, ",", filll)
        count += 1


################ Backoff (alpha1,2,3 are parameters)###############
########################################

backoff = {}


def Backoff():
    global backoff
    global trigram
    global twograms
    global unigram

    alpha3 = 0.33
    alpha2 = 0.33
    alpha1 = 0.34

    for i in trigram.keys():
        backoff[i] = trigram[i]
        for tri in backoff[i].keys():
            backoff[i][tri] = alpha3 * backoff[i][tri]
        for j in twograms.keys():
            if j == i.split()[1]:
                for k in twograms[j].keys():

                    if k in backoff[i].keys():
                        backoff[i][k] = alpha2 * twograms[j][k]+backoff[i][k]
                    else:
                        backoff[i][k] = twograms[j][k]
                        backoff[i][k] = alpha2 * backoff[i][k]
        for uni in unigram[''].keys():
            if uni in backoff[i].keys():
                backoff[i][uni] = alpha1 * unigram[''][uni] + backoff[i][uni]
            else:
                backoff[i][uni] = alpha1 * unigram[''][uni]

    # for printing backoff model of traning set just call Backoff function
    # print("Backoff Model:")
    # for n in backoff.keys():
    #
    #     for i in (backoff[n]).keys():
    #         print(i, "|", n, ":", (backoff[n])[i])


#################   Filling Test Data Sentences with BackoffModel    ###############
######################################################


def fillBackoff(foundTri):
    global backoff
    if foundTri not in backoff.keys():
        return "NOT FOUND"
    possible_words = backoff[foundTri]

    nextW = max(possible_words.items(), key=operator.itemgetter(1))[0]
    return nextW


def fillSentencesBackoff():
    global sentences
    global foundLabels
    count = 1
    for se in sentences:
        tri = ""
        wordsInSen = se.lower().split()
        tri += wordsInSen[wordsInSen.index('$') - 2]
        tri += " "
        tri += wordsInSen[wordsInSen.index('$') - 1]
        filll = fillBackoff(tri)
        foundLabels.append(filll)
        print(count, ",", filll)
        count += 1


#######################################################


# Unigram()
Bigram()
# Trigram()
# Backoff()
fillSentencesBigram()
matchedCount = 0

for i in range(len(labels)):
    if labels[i] == foundLabels[i]:
        matchedCount += 1
print("Num of matched: ", matchedCount)
print("Accuracy: ", (matchedCount / len(labels)) * 100, "%")
