import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Create dictionary to store probabilities
    probability = dict()

    for site in list(corpus.keys()):
        probability[site] = 0.0
    
    # If there is no link, all pages have the same probability to be visited
    if len(page) == 0:
        for site in probability:
            probability[site] = 1 / len(corpus)
        
        return probability
    
    # Else distribuite the probabilities
    for link in page:  # damping factor divided between all links that have in the page
        probability[link] = damping_factor / len(page)

    for site in probability:  # 1 - damping factor divided between all pages in corpus
        probability[site] += (1 - damping_factor) / len(corpus)

    return probability


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Create dictionary to store probabilities
    probability = dict()
        
    # Create dictionary to store Transition Model probabilities
    tmp = dict()
    
    # Fill the probability and tmp dictionaries
    for site in corpus:
        probability[site] = 1.0 / len(corpus)
        tmp[site] = transition_model(corpus, corpus[site], damping_factor)

    # Choose randomly the first page that the surfer will visited
    page = random.choice(list(probability.keys()))

    remainingSamples = n

    while remainingSamples > 0:
        # Distribute the tmp probabilities on the actual page for each page 
        for site in probability:
            probability[site] += tmp[page][site]

        # Choose a new page to visit based on the damping factor
        if random.random() <= damping_factor and len(page) > 0:
            page = random.choice(list(corpus[page]))
        else:
            page = random.choice(list(corpus.keys()))
        
        remainingSamples -= 1

    # After take all samples, divide the probabilities by the number samples to normalize them
    for site in probability:
        probability[site] /= n

    return probability


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Create dictionary to store probabilities
    probability = dict()

    # Fill the probability dictionary
    for site in corpus:
        if len(corpus[site]) == 0:  # If the page has no links, add a link to all the pages in corpus
            for page in corpus:
                corpus[site].add(page)

        probability[site] = 1.0 / len(corpus)

    continueIterating = True  # Variable to turn off loop

    # Loop until all diferences between the new and last probabilities be less than 0.0001
    while continueIterating:
        continueIterating = False

        temp = probability.copy()

        # Iterate trough all the  pages in corpus
        for p in corpus:
            totalSum = 0.0
            # Iterate trough all the pages in corpus and their links 
            for i in corpus:
                for j in corpus[i]:
                    """
                    If there is a link to the current page (p) update the total sum of probability
                    divided by the number of links of all pages that links to current page
                    """
                    if j == p:
                        totalSum += probability[i] / len(corpus[i])

            probability[p] = ((1 - damping_factor) / len(corpus)) + (damping_factor * totalSum)  # PageRank formula

            if abs(probability[p] - temp[p]) >= 0.0001:
                continueIterating = True

    return probability


if __name__ == "__main__":
    main()
