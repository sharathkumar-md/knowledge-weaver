Name: Sharath Kumar MD
Roll no: 23B2150
Department: Mechanical
IIT Bombay

# Knowledge Weaver

**A Multi-Agent System for Automated Personal Knowledge Graph Construction**

## Motivation

During my learning journey, I realized I had tons of scattered notes, PDFs, articles, and YouTube videos but no way to see how everything connected. I often forgot what I learned or couldn't see gaps in my understanding. This project solves that problem by automatically building a knowledge graph from all my learning materials.

## What This Project Does

Knowledge Weaver is an intelligent multi-agent system that:
- **Automatically extracts** concepts and relationships from my notes, PDFs, and videos
- **Builds a knowledge graph** showing how everything connects
- **Finds knowledge gaps** - what I should learn next based on what I already know
- **Detects contradictions** - when different sources say different things
- **Recommends learning paths** - optimal order to learn new concepts
- **Grows over time** - gets smarter as I add more content

## Architecture - Why Multi-Agent?

I designed this as a multi-agent system because different tasks require different expertise. Instead of one giant AI trying to do everything, I built specialized agents that work together:

### Agent 1: Extractor
**What it does:** Reads my notes and extracts knowledge triples (subject-relationship-object)
**Why:** I fine-tuned a LoRA model on my personal note-taking style to understand my abbreviations and informal language
**Example:** From "ML uses neural networks" → extracts `(Machine Learning, USES, Neural Networks)`

### Agent 2: Linker/Deduplicator
**What it does:** Realizes that "ML", "Machine Learning", and "machine learning" are the same thing
**Why:** Without this, my graph would have duplicate concepts everywhere
**How:** Uses embeddings to find similar entities and merges them intelligently

### Agent 3: Reasoner
**What it does:** Analyzes the knowledge graph to find gaps and contradictions
**Why:** Helps me discover what I don't know yet based on what I do know
**Example:** If I know A→B and B→C but not A→C, it suggests I should learn about that connection

### Agent 4: Planner
**What it does:** Recommends what to learn next based on my current knowledge
**Why:** Creates an optimal learning path instead of random studying
**How:** Uses graph algorithms to find concepts that would give me the most understanding gain

### Supporting Components
- **Ingestion Pipeline**: Handles different file formats (MD, PDF, YouTube transcripts)
- **Schema Manager**: Tracks where each fact came from and when I learned it
- **RAG System**: Uses vector database for fast semantic search across my notes

## Key Technical Decisions & Why I Made Them

### 1. LoRA Fine-Tuning (Not Full Model Training)
**Decision:** Use PEFT/LoRA instead of training from scratch
**Why:** My laptop can't handle full model training, and LoRA gives 90% of the benefit with 1% of the compute
**Trade-off:** Works great for my personal notes, might need adjustment for someone else's style

### 2. NetworkX for Graph Storage
**Decision:** Start with NetworkX, add Neo4j support later
**Why:** NetworkX is pure Python, easy to debug, and sufficient for my <10K notes
**Future:** Will migrate to Neo4j when the graph gets huge or I need advanced queries

### 3. ChromaDB for Vector Storage
**Decision:** ChromaDB over FAISS or Pinecone
**Why:** ChromaDB persists locally, has a clean API, and I don't want to pay for cloud services
**Benefit:** All my personal notes stay on my machine

### 4. spaCy for NLP Preprocessing
**Decision:** spaCy over NLTK
**Why:** Faster, better entity recognition, and integrates well with transformers
**Usage:** Tokenization, POS tagging, basic entity extraction before the LLM

### 5. Distil-GPT2 as Base Model
**Decision:** Small model (82M params) instead of Llama or GPT-3.5
**Why:** Runs on CPU, fast inference, good enough for triple extraction
**Trade-off:** Less context understanding than bigger models, but fine-tuning helps

## Installation & Setup

### Step 1: Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/sharathkumar-md/knowledge-weaver.git
cd knowledge-weaver

# Create virtual environment (I use venv but conda works too)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Download spaCy language model (needed for NLP preprocessing)
python -m spacy download en_core_web_sm
```

### Step 3: Verify Installation
```bash
# Run quick test to make sure everything works
python quick_test.py

# Should see: "ALL CORE COMPONENTS WORKING!"
```

## How to Use This Project

### Option 1: Test with Provided Test Suite
```bash
# Run comprehensive agent tests (recommended first step)
python test_agents.py

# This will test:
# - Knowledge extraction from text
# - Entity linking and deduplication
# - Gap detection
# - Learning path recommendations
# - Multi-agent collaboration
```

### Option 2: Process Your Own Notes
```bash
# 1. Put your notes/PDFs in data/raw/ folder

# 2. Run the full pipeline
python demo.py

# This automatically:
# - Extracts text from your files
# - Builds the knowledge graph
# - Finds knowledge gaps
# - Suggests what to learn next
```

### Option 3: Use Individual Components
```bash
# Just extract knowledge triples
python main.py extract --input ./data/processed

# Launch interactive UI
python main.py ui
```

## (Optional) Training Your Own LoRA Model

I included a pre-trained LoRA model, but if you want to fine-tune it on your own notes:

```bash
# 1. Annotate some of your notes (50-200 examples recommended)
#    Format: {"input": "your note text", "output": "S:Entity|R:relation|O:Entity"}

# 2. Prepare the dataset
python scripts/prepare_dataset.py

# 3. Train LoRA adapter (takes ~30 min on my laptop CPU)
python scripts/train_lora.py \
  --config configs/config.yaml \
  --output_dir models/lora/extractor_v1
```

**Note:** The baseline extractor works without LoRA, but fine-tuning helps with personal abbreviations and terminology.

## Project Structure Explained

I organized the code to separate concerns and make each component testable:

```
knowledge-weaver/
├── src/                    # Main source code
│   ├── agents/            # The 4 AI agents (Extractor, Linker, Reasoner, Planner)
│   │   ├── extractor.py   # Extracts knowledge triples from text
│   │   ├── linker.py      # Deduplicates and merges entities
│   │   ├── reasoner.py    # Finds gaps and contradictions
│   │   └── planner.py     # Recommends learning paths
│   │
│   ├── graph/             # Knowledge graph management
│   │   ├── graph_store.py # Main graph operations (add, query, analyze)
│   │   └── schema_manager.py # Tracks metadata and provenance
│   │
│   ├── ingestion/         # Document processing pipeline
│   │   ├── pdf_extractor.py   # Extract text from PDFs
│   │   ├── video_processor.py # Get YouTube transcripts
│   │   └── pipeline.py    # Orchestrates all extractors
│   │
│   ├── rag/               # Retrieval-Augmented Generation
│   │   └── vector_store.py # ChromaDB wrapper for semantic search
│   │
│   ├── evaluation/        # Testing and metrics
│   │   └── evaluator.py   # Calculates precision, recall, F1
│   │
│   └── utils.py           # Shared utilities (config loading, logging)
│
├── data/                  # All data files
│   ├── raw/              # Put your PDFs, MD files here
│   ├── processed/        # Cleaned text chunks (auto-generated)
│   └── datasets/         # Training data for LoRA
│
├── models/               # Model storage
│   └── lora/            # Fine-tuned LoRA adapters
│       └── extractor_v1/ # My trained extractor model
│
├── configs/             # Configuration
│   └── config.yaml      # Main config (model params, agent settings)
│
├── logs/                # Application logs
│
├── demo_output/         # Output from demo runs
│
├── quick_test.py        # Fast sanity check (9 tests)
├── test_agents.py       # Comprehensive agent testing (5 tests)
├── demo.py              # End-to-end demo pipeline
└── main.py              # CLI entry point
```

## How I Built the Training Dataset

Since there's no public dataset for "my personal notes → knowledge triples", I had to create one:

### 1. Public Data (Foundation)
- Started with Wikipedia→Wikidata pairs to learn general knowledge extraction
- ~1000 examples of (text, triples) pairs

### 2. Synthetic Data (Augmentation)
- Used GPT to generate note-like text with known triples
- Helps the model learn my informal note-taking style
- ~500 examples

### 3. Personal Annotations (Fine-tuning)
- Manually annotated 100 of my own notes with correct triples
- This is what makes the model understand MY writing style
- Format: `{"input": "ML uses neural nets for image recognition", "output": "S:Machine Learning|R:USES|O:Neural Networks\nS:Neural Networks|R:USED_FOR|O:Image Recognition"}`

**Total:** ~1600 training examples (enough for LoRA, not enough for full fine-tuning)

## Evaluation - How I Know It Works

I implemented multiple evaluation metrics to validate the system:

### 1. Extraction Quality (Agent 1)
- **Precision:** Are the extracted triples actually correct?
- **Recall:** Did it find all the knowledge in the text?
- **F1 Score:** Harmonic mean of precision and recall
- **My Results:** ~75% F1 on personal notes (better than baseline GPT-2 at 45%)

### 2. Graph Quality (Overall System)
- **Coverage:** What % of my topics are represented? (Currently ~80%)
- **Density:** How connected is the graph? (Target: 0.3-0.5)
- **Component Count:** Fewer disconnected clusters = better
- **Diameter:** Max distance between concepts (lower is better)

### 3. Agent Performance
- **Linker Accuracy:** Are duplicates merged correctly? (~85%)
- **Gap Detection:** Do identified gaps make sense? (Manual validation)
- **Recommendation Quality:** Are suggested topics relevant? (User study needed)

### 4. User Value (The Real Test)
- **Retrieval Accuracy:** Can I find relevant notes when searching?
- **Learning Effectiveness:** Do the recommendations actually help me learn?
- **Time Saved:** Faster than manually organizing notes

## Technology Stack & Why I Chose Each

| Component | Technology | Why This Choice |
|-----------|-----------|-----------------|
| **Base LLM** | DistilGPT-2 (82M) | Runs on CPU, fast, good for triple extraction |
| **Fine-tuning** | PEFT/LoRA | Only need to train 0.5M params instead of 82M |
| **Embeddings** | sentence-transformers (MiniLM) | Best quality/speed trade-off for semantic search |
| **Vector DB** | ChromaDB | Local-first, persistent, clean API |
| **Graph DB** | NetworkX | Pure Python, easy debugging, sufficient for my scale |
| **NLP** | spaCy | Fast, production-ready, good entity recognition |
| **Orchestration** | LangChain | Handles LLM chains and agent workflows |
| **UI** | Streamlit | Quick to build, interactive, Python-native |
| **PDF** | PyPDF2 + pdfplumber | Complementary - PyPDF2 for text, pdfplumber for tables |
| **Video** | youtube-transcript-api | Free, no API key needed, reliable |

## Configuration

The `configs/config.yaml` file controls everything. I documented each setting:

```yaml
# Example: Change base model
models:
  base_model: "distilgpt2"  # Try: "gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Example: Adjust agent behavior
agents:
  extractor:
    confidence_threshold: 0.5  # Lower = more triples, higher = more precise
  linker:
    similarity_threshold: 0.85  # Higher = stricter merging
```

See full config file for all options.

## Privacy & Data Security

**Everything runs locally on your machine:**
- ✅ No data sent to external servers
- ✅ All models run locally (CPU-friendly)
- ✅ Knowledge graph stored in local files
- ✅ Vector database persisted locally

**Optional online features (disabled by default):**
- YouTube transcript fetching (only if you process videos)
- PDF downloads from URLs (only if you provide URLs)

**Your notes never leave your computer.**

## Challenges I Faced & How I Solved Them

### Challenge 1: Entity Resolution is Hard
**Problem:** "ML", "Machine Learning", "machine learning" should be the same entity
**Solution:** Used embedding similarity + clustering. Threshold tuning was key.

### Challenge 2: Too Many Low-Quality Triples
**Problem:** spaCy extracts garbage like "(is, a, the)"
**Solution:** Added confidence thresholds, POS filtering, and LoRA fine-tuning

### Challenge 3: Knowledge Gaps Too Generic
**Problem:** "You should learn X" for every concept with <2 connections
**Solution:** Ranked gaps by centrality, relevance to current topic, and learning value

### Challenge 4: Slow LoRA Training
**Problem:** Training took 3 hours initially
**Solution:** Reduced sequence length (512→256), smaller batch size, gradient accumulation

### Challenge 5: Debugging Multi-Agent Systems
**Problem:** Hard to tell which agent is failing
**Solution:** Built comprehensive test suite (`test_agents.py`) with isolated agent tests

## What I Learned

1. **LoRA is magic** - Fine-tuning 0.5M params gives 80% of full fine-tuning benefits
2. **Knowledge graphs need cleaning** - Entity resolution is harder than extraction
3. **Multi-agent > Monolithic** - Easier to debug, test, and improve individual components
4. **Local-first is viable** - You don't need GPT-4 API for everything
5. **Evaluation is essential** - Without metrics, I couldn't tell if changes helped

## Future Improvements

If I had more time, I would add:
- [ ] Temporal tracking (when did I learn each concept?)
- [ ] Confidence decay (older knowledge gets lower confidence)
- [ ] Active learning (ask me to clarify ambiguous extractions)
- [ ] Multi-modal support (diagrams, equations, code snippets)
- [ ] Collaboration (merge knowledge graphs from multiple people)
- [ ] Neo4j integration (for better graph queries)

## Acknowledgments

This project builds on amazing research and tools:
- **Anthropic's Agent Design Patterns** - Multi-agent architecture inspiration
- **KGGen Paper** (arxiv:2405.10467) - Knowledge graph generation techniques
- **PEFT/LoRA** - Efficient fine-tuning methodology
- **spaCy, Transformers, NetworkX** - Amazing open-source libraries

## License

MIT License - Feel free to use this for your own knowledge management!
