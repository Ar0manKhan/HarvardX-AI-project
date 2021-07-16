import os
import random
import re
import sys
from collections import Counter
from math import isclose

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
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor=DAMPING):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = {}
    page_list = []
    weight_list = []
    page_links = corpus[page]
    try:
        page_links_probablity = damping_factor / len(page_links)
    except:
        page_links_probablity = 0
    corpus_page = corpus.keys()
    corpus_page_probablity = (1 - damping_factor) / len(corpus_page)
    for i in page_links:
        page_list.append(i)
        weight_list.append(page_links_probablity + corpus_page_probablity)
    for i in corpus_page - page_links:
        page_list.append(i)
        weight_list.append(corpus_page_probablity)
    return page_list, weight_list


def sample_pagerank(corpus, damping_factor=DAMPING, n=SAMPLES):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_visit = []
    page = random.choice(list(corpus.keys()))
    for _ in range(n):
        page_visit.append(page)
        page_visit_prob = transition_model(corpus, page, damping_factor)
        page = random.choices(page_visit_prob[0], weights=page_visit_prob[1])[0]

    page_visit_stat = Counter(page_visit)
    page_probablity = {}
    for i in page_visit_stat:
        page_probablity[i] = page_visit_stat[i] / n
    return page_probablity


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    corpus_pages = set(corpus.keys())
    random_probablity = (1 - damping_factor) / len(corpus_pages)
    # Adding links to the page which don't has any link
    for page in corpus:
        if not corpus[page]:
            corpus[page] = corpus_pages

    # Making default page rank ie. 1/N to each page
    page_rank = {}
    current_rank = {}  # New page rank for comparing two ranks
    for page in corpus:
        page_rank[page] = 1 / len(corpus_pages)
        current_rank[page] = 0

    # Creating dictionary in which key is each page and value will be list of
    # pages in which link of page (key) is found
    page_linked = {}
    for page in corpus:
        page_linked[page] = []
    for page in corpus:
        for link in corpus[page]:
            page_linked[link].append(page)

    while True:
        for page in corpus:
            sum_probab_linked = 0
            for parent in page_linked[page]:
                sum_probab_linked += page_rank[parent] / len(corpus[parent])
            current_rank[page] = random_probablity + damping_factor * sum_probab_linked
        if similar_pagerank(page_rank, current_rank):
            break
        page_rank = current_rank.copy()
    return page_rank


def similar_pagerank(a, b):
    """
    This is the method which will return true if every page of both ranks has
    difference of 0.001
    """
    for page in a:
        if not isclose(a[page], b[page], rel_tol=0.001):
            return False
    return True


if __name__ == "__main__":
    main()
