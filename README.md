# PageFinder

A two-stage web crawler that finds task-relevant pages within a seed domain's ecosystem. Stage 1 uses Personalized PageRank to discover and rank related domains. Stage 2 (optional) uses those domains as a whitelist for semantic search.

## How it works

**Stage 1 - Domain Discovery (`related_domains.py`)**

Crawls pages within your seed domain (and its subdomains), extracts all hyperlinks, and builds a domain-level graph. External domains show up as nodes but aren't actually visited. Then runs Personalized PageRank with a restart vector biased toward the seed, so domains that are structurally close to your starting point rank higher than random external sites.

The output is a ranked list of domains by importance relative to the seed. Subdomains typically rank highest, followed by partner organizations, with social media and ad networks (and possibly malware links) pushed to the bottom even if they appear everywhere.

This modified Personalized PageRank algorithm is described in more detail in our research paper, currently in preprint as the file *paper.pdf* in this repo.

**Stage 2 - Task-Relevant Search (`task_pages.py`)**

Takes the top-k domains from Stage 1 as a whitelist, then crawls within that boundary looking for pages matching your natural language query. Uses OpenAI embeddings to score page content against your task, with a priority queue that explores promising pages first.

Pages scoring above your thresholds get returned as results. The whitelist acts as a hard constraint - the crawler won't wander off to random external sites.

## Key design decisions

- **Custom restart vector**: Instead of uniform teleportation, random walks restart at the seed domain and its subdomains. This measures importance *from the seed's perspective* rather than global popularity.

- **Template filtering**: Navigation, footers, and sidebars get stripped before link extraction. Otherwise every page would have edges to the same set of nav links, drowning out the actual content links.

- **WWW canonicalization**: `www.example.com` and `example.com` are treated as the same host, but only for seed-related domains. External sites keep their original form.

- **Dangling node handling**: External domains have no outgoing links (we don't crawl them), so they'd become rank sinks. Instead, they teleport via the restart vector. In the Markov transition matrix, we replace the column corresponding to the external domains with restart vector **v**.

- **Task-gated expansion**: In Stage 2, pages that don't score well against your task don't get their links followed. Saves a lot of wasted crawling.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-proj-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

The embedding model is only needed for Stage 2. Stage 1 works without any API keys.

## Usage

### Stage 1: Rank related domains

```bash
# Basic usage
python related_domains.py cmu.edu

# Crawl more pages, show more results
python related_domains.py cmu.edu --max-pages-total 500 --max-depth 3 --top-k 30

# Treat an external domain as first-party (for sites with multiple apex domains)
python related_domains.py troycolts.org --alias troy.k12.mi.us
```

### Stage 2: Find task-relevant pages

```bash
# Run both stages together
python task_pages.py cmu.edu "Find tuition and fees information"

# Adjust thresholds for more/fewer results
python task_pages.py stanford.edu "PhD admission requirements" --keep-threshold 0.3 --expand-threshold 0.4

# Use more domains from Stage 1
python task_pages.py mit.edu "faculty directory" --whitelist-top-k 15
```

## Parameters

### related_domains.py

| Flag | Default | What it does |
|------|---------|--------------|
| `--max-pages-total` | 2000 | Total pages to crawl across all hosts |
| `--max-pages-per-host` | 200 | Max pages from any single subdomain |
| `--max-depth` | 3 | BFS depth limit |
| `--damping` | 0.85 | PageRank damping factor (probability of following a link vs teleporting) |
| `--seed-weight` | 0.9 | How much of the restart vector goes to the seed vs subdomains |
| `--top-k` | 50 | Number of domains to display |
| `--alias` | - | Additional domains to treat as first-party (repeatable) |
| `--verbose` | off | Show per-page crawl progress |

### task_pages.py

| Flag | Default | What it does |
|------|---------|--------------|
| `--whitelist-top-k` | 10 | How many domains from Stage 1 to include in whitelist |
| `--max-pages` | 100 | Max pages to crawl in Stage 2 |
| `--keep-threshold` | 0.25 | Min similarity to keep a page as a candidate result |
| `--expand-threshold` | 0.35 | Min similarity to follow links from a page |
| `--top-results` | 10 | Number of results to show |
| `--chunk-size` | 5000 | Characters per embedding chunk |
| `--max-chunks` | 2 | Max chunks to embed per page (for speed) |

## Files

```
related_domains.py   # Stage 1: Domain ranking via Personalized PageRank
task_pages.py        # Stage 2: Task-relevant page search with embeddings
ppr.py               # Personalized PageRank algorithm (imported by related_domains.py)
requirements.txt     # Python dependencies
.env                 # API keys (create this yourself)
```

## Changes
- Modified PPR algorithm to collapse urls to their domains when creating nodes, instead of building the graph by treating each page in a domain as distinct nodes.
- Changed the 3 stage process to 2. After whitelisting domains, instead of running BFS on the whole website before running tasks as a separate stage, running best-first search each time a task is given as one stage is a lot more accurate for large networks, although it loses efficiency when queuing a large number of tasks.
- The best-first search is done using a priority queue that sorts based on 3 factors:
    - Parent page similarity (default 0.4 weighting)
    - Anchor text relevance (default 0.35 weighting)
    - URL token relevance (default 0.25 weighting)

## Next Steps

- Use LLM to verify that top page is relevant to task

- Then find path to that page using existing graph from Pass 2