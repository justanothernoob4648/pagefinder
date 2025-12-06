# PageRank Plus

Lightweight crawler + PageRank scorer with optional OpenAI-backed semantic ranking.

## Setup
- Install dependencies: `pip install -r requirements.txt`
- Add your API key (LLM scorer): either put `OPENAI_API_KEY=sk-...` in a `.env` file in the project root (auto-loaded) or export it in your shell: `export OPENAI_API_KEY=sk-...`
- (Optional) pick an embedding model: `export OPENAI_EMBEDDING_MODEL=text-embedding-3-small`

## Usage
- Two-pass crawl with external whitelist discovery (default):
  - Pass 1: crawl the root domain, add edges to external domains as nodes, run PageRank, and whitelist the top 5 domains.
  - Pass 2: crawl across the root + whitelisted domains (depth 2), collect text/snippets, and rank by semantic similarity.
- Example (LLM + hybrid):  
  `python main.py https://example.edu --use-llm --hybrid-expand "Help me find tuition fees"`
- Interactive mode (no goals provided):  
  `python main.py https://example.edu --use-llm --hybrid-expand`

### Flags
- `--use-llm` enable OpenAI embeddings for semantic scoring.
- `--llm-model` override the embedding model name.
- `--pagerank-weight` blend PageRank with semantic similarity (0-1).
- `--hybrid-expand` if similarity is low, expand crawl from top candidates and retry ranking.
- `--hybrid-threshold` score threshold to trigger expansion; lower is stricter.
- `--hybrid-max-new` max new pages to fetch per expansion.
- `--hybrid-frontier-k` use outbound links from the top-k candidates as frontier seeds.
- `--whitelist-depth` pass1 depth when discovering external domains (default 2).
- `--whitelist-top-k` how many external domains to whitelist (default 5).
- `--crawl-depth` pass2 depth across allowed domains (default 2).
