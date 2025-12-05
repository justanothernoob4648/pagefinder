# PageRank Plus

Lightweight crawler + PageRank scorer with optional OpenAI-backed semantic ranking.

## Setup
- Install dependencies: `pip install -r requirements.txt`
- Add your API key (LLM scorer): either put `OPENAI_API_KEY=sk-...` in a `.env` file in the project root (auto-loaded) or export it in your shell: `export OPENAI_API_KEY=sk-...`
- (Optional) pick an embedding model: `export OPENAI_EMBEDDING_MODEL=text-embedding-3-small`

## Usage
- Basic (bag-of-words similarity):  
  `python main.py https://example.edu "Find admission requirements"`
- Enable LLM semantic ranking with embeddings:  
  `python main.py https://example.edu --use-llm --pagerank-weight 0.2 "Help me find tuition fees"`
- Interactive mode (no goals provided):  
  `python main.py https://example.edu --use-llm`
- Hybrid expand (if scores are low, crawl best candidate links and retry):  
  `python main.py https://example.edu --use-llm --hybrid-expand --hybrid-threshold 0.35`

### Flags
- `--use-llm` enable OpenAI embeddings for semantic scoring.
- `--llm-model` override the embedding model name.
- `--pagerank-weight` blend PageRank with semantic similarity (0-1).
- `--hybrid-expand` if similarity is low, expand crawl from top candidates and retry ranking.
- `--hybrid-threshold` score threshold to trigger expansion; lower is stricter.
- `--hybrid-max-new` max new pages to fetch per expansion.
- `--hybrid-frontier-k` use outbound links from the top-k candidates as frontier seeds.
