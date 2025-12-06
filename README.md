# PageFinder

Lightweight crawler + PageRank scorer to find relevant pages for user-defined tasks as part of Accessibility Testing Agent - https://github.com/justanothernoob4648/at-agent

## Setup
- Install dependencies using `pip install -r requirements.txt`
- Add your API key for the LLM scorer using `OPENAI_API_KEY=sk-...` in a `.env` file in the proj root (auto-loaded)
- Pick an embedding model if you want and put it in the `.env` file as `OPENAI_EMBEDDING_MODEL=text-embedding-ur-model`

## Usage
- Two-pass crawl with external domain whitelist discovery:
  - Pass 1: crawl the root domain, add edges to external domains as nodes, run PageRank, and whitelist the top 5 domains.
    - Additional layer added to let user interactively select what to whitelist from the top domains
  - Pass 2: crawl across the root + whitelisted domains (BFS depth 2), collect text/snippets.
- After crawl, user provides tasks for agent. Using provided LLM, ranks pages collected in Pass 2 by semantic similarity to task.
- Example (LLM + hybrid):  
  `python main.py https://example.edu --use-llm --hybrid-expand "Help me find tuition fees"`
- Interactive mode (no goals provided):  
  `python main.py https://example.edu --use-llm --hybrid-expand`

### Flags
- `--use-llm` enable OpenAI embeddings for semantic scoring. (Recommended)
- `--llm-model` override embedding model name.
- `--pagerank-weight` blend PageRank with semantic similarity (0-1). (default 0)
- `--hybrid-expand` if similarity is low, expand crawl from top candidates and retry ranking.
- `--hybrid-threshold` score threshold to trigger expansion; lower is stricter.
- `--hybrid-max-new` max new pages to fetch per expansion.
- `--hybrid-frontier-k` use outbound links from the top-k candidates as frontier seeds.
- `--whitelist-depth` pass1 depth when discovering external domains (default 2).
- `--whitelist-top-k` how many external domains to whitelist (default 5).
- `--crawl-depth` pass2 depth across allowed domains (default 2).


## Next Steps
- Frontier expansion can be more intelligent, since we already have the task
  - Compute semantic similarity during crawl and only visit sites that pass certain threshold
  - Crawl with more depth than breadth
- Use LLM to verify that top page is relevant to task
- Then find path to that page using existing graph from Pass 2