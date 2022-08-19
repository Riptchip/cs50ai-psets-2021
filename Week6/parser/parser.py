import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP C NPorNA VPorNA NPorNA
NA -> 
C -> Conj | NA
NP -> N | Det NP | P NP | Adj NP
VP -> V | V NP | NP VP | Conj VP | Adv VP | V NP Adv
NPorNA -> NP | NA
VPorNA -> VP | NA
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))        


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    
    b = sentence.lower()

    b = nltk.word_tokenize(b)

    # Iterate through each word in sentence to remove non alpha words
    for word in b:
        haveAlpha = False
        for char in word:
            if char.isalpha():
                haveAlpha = True
                break
        if not haveAlpha:
            b.remove(word)

    return b


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """

    nounPhraseChunks = list()

    # Iterate through all subtrees of sentence that has the 'NP' label
    for t in tree.subtrees(lambda i: i.label() == 'NP'):
        count = 0
        # Count how much subtrees labeled 'NP' this subtree have
        for s in t.subtrees():
            if s.label() == 'NP':
                count += 1
    
        # If subtree hasn't another subtree of label 'NP', this subtree is a noun phrase chunk
        # The if statement should be "if count == 0", but as "t.subtrees" return itself too the statement is like that
        if count <= 1:
            nounPhraseChunks.append(t)

    return nounPhraseChunks


if __name__ == "__main__":
    main()
