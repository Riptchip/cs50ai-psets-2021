import nltk
import sys
import os
import math
import string
import glob

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    # Given a directory name, return a dictionary mapping the filename of each
    # `.txt` file inside that directory to the file's contents as a string.

    files = dict()
    
    for textFile in glob.glob(os.path.join(directory, '*.txt')):
        with open(textFile, 'r', encoding='utf8') as t:
            files[textFile[len(directory) + len(os.sep):]] = t.read()

    return files


def tokenize(document):
    # Given a document (represented as a string), return a list of all of the
    # words in that document, in order.

    # Process document by coverting all words to lowercase, and removing any
    # punctuation or English stopwords.

    document = document.lower()

    words = list()

    for word in nltk.word_tokenize(document):
        for char in word:
            if char in string.punctuation:
                word = word.replace(char, '')
        
        if word in string.punctuation or word in nltk.corpus.stopwords.words("english"):
            continue

        words.append(word)
    
    return words


def compute_idfs(documents):
    # Given a dictionary of `documents` that maps names of documents to a list
    # of words, return a dictionary that maps words to their IDF values.

    # Any word that appears in at least one of the documents should be in the
    # resulting dictionary.

    wordsIDF = dict()
    words = dict()

    for document in documents:
        for word in documents[document]:
            if not word in words:
                words[word] = [document]
            elif not document in words[word]:
                words[word].append(document)

    for word in words:
        wordsIDF[word] = math.log(len(documents) / len(words[word]))

    return wordsIDF


def top_files(query, files, idfs, n):
    # Given a `query` (a set of words), `files` (a dictionary mapping names of
    # files to a list of their words), and `idfs` (a dictionary mapping words
    # to their IDF values), return a list of the filenames of the the `n` top
    # files that match the query, ranked according to tf-idf.

    tf_idfs = dict()

    for document in files:
        wordCounter = dict()

        for word in files[document]:
            if word in query:
                if word in wordCounter:
                    wordCounter[word] += 1
                else:
                    wordCounter[word] = 1
            
        tf_idfs[document] = 0

        for word in wordCounter:
            tf_idfs[document] += wordCounter[word] * idfs[word]

    return list({k: v for k, v in sorted(tf_idfs.items(), key=lambda item: item[1], reverse=True)}.keys())[:n]


def top_sentences(query, sentences, idfs, n):
    # Given a `query` (a set of words), `sentences` (a dictionary mapping
    # sentences to a list of their words), and `idfs` (a dictionary mapping words
    # to their IDF values), return a list of the `n` top sentences that match
    # the query, ranked according to idf. If there are ties, preference should
    # be given to sentences that have a higher query term density.

    ranking = dict()

    for sentence in sentences:
        sentenceHasWordInQuery = False

        for word in sentences[sentence]:
            if word in query:
                sentenceHasWordInQuery = True
                if sentence in ranking:
                    ranking[sentence][0] += idfs[word]
                    ranking[sentence][1] += 1
                else:
                    ranking[sentence] = [idfs[word], 1]
        
        if sentenceHasWordInQuery:
            ranking[sentence][1] /= len(sentences[sentence])

    ranking = {k: v for k, v in sorted(ranking.items(), key=lambda item: item[1][0], reverse=True)}

    topSentences = dict()
    rankingKeys = list(ranking.keys())

    nTopSentences = n
    while n > 0:
        currentPos = abs(n - nTopSentences)
        ties = {rankingKeys[currentPos]: ranking[rankingKeys[currentPos]]}
        
        for i in range(currentPos, len(rankingKeys)):
            if i == len(rankingKeys) - 1:
                break

            if ranking[rankingKeys[i]] == ranking[rankingKeys[i + 1]]:
                ties[rankingKeys[i + 1]] = ranking[rankingKeys[i + 1]]
            else:
                break
        
        if len(ties) == 0:
            n -= 1
            topSentences[rankingKeys[currentPos]] = ranking[rankingKeys[currentPos]]
            continue
        
        topSentences.update({k: v for k, v in sorted(ties.items(), key=lambda item: item[1][1], reverse=True)})

        n -= len(ties)
        
    return list(topSentences.keys())[:nTopSentences]


if __name__ == "__main__":
    main()
