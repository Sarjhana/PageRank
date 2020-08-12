import os
import random
import re
import sys
import numpy as np
from random import choice

DAMPING = 0.85
SAMPLES = 10000


def main():
    '''if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])'''
    
    corpus =  crawl('corpus2')
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
    
    values = corpus.get(page)
    probability_dist = {k:0.0 for k in corpus.keys()}
    if  len(values) != 0:
        for value in values:
            probability_dist[value] = DAMPING/len(values)
        for key in corpus:
            probability_dist[key] += (1-DAMPING)/len(corpus)
    else:
        for k in probability_dist:
            probability_dist[k] = 1/len(corpus)

    #print(probability_dist)
    return probability_dist
        
        
    #raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    visited = dict()
    start = choice(list(corpus))
    visited[start] = 1
    
    for i in range(n):
        transition = transition_model(corpus,start,damping_factor)
        
        keys = []
        for page in transition.keys():
            keys.append(page)
        
        prob_distribution = []
        for value in transition.values():
            prob_distribution.append(value)


        sample = np.random.choice(keys, p = prob_distribution)
        start = sample

        if sample in visited:
            visited[sample] += 1
        else:
            visited[sample] = 1

    for page in visited:
        visited[page] = visited[page]/n

    return visited

    
        
    #raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    currentDist = {k:0.0 for k in corpus.keys()}
    N = len(corpus)
    finalDist = dict()
    for k in corpus.keys():
        currentDist[k] = 1/N
        finalDist[k] = 1/N
    count = False
    while(True):
        for p in corpus:
            summation = 0.0
            for i in corpus:
                if i != p and p in corpus[i]:
                    numLink = len(corpus[i])
                    if numLink == 0:
                        numLink = N
                    pagerank_i = currentDist[i]
                    summation += (pagerank_i/numLink)
            pagerank_p = ((1-damping_factor)/N) + damping_factor*summation
            finalDist[p] = pagerank_p
        for p in corpus:
            if abs(finalDist[p] - currentDist[p]) > 0.001:
                count = True
        if count == False:
            break
        count = False
        for p in corpus:
            currentDist[p] = finalDist[p]
    return finalDist
    
    #raise NotImplementedError


if __name__ == "__main__":
    main()
