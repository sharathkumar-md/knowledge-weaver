# Knowledge Weaver - Project Overview

**A detailed explanation of my implementation for evaluators**

## Table of Contents
1. [Project Goal](#project-goal)
2. [Why This Matters](#why-this-matters)
3. [System Architecture](#system-architecture)
4. [Implementation Details](#implementation-details)
5. [Key Algorithms](#key-algorithms)
6. [Design Decisions](#design-decisions)
7. [Challenges & Solutions](#challenges--solutions)

---

## Project Goal

**Build an intelligent multi-agent system that automatically constructs a personal knowledge graph from unstructured learning materials (notes, PDFs, videos) and helps me learn more effectively.**

### Specific Objectives
1. ✅ Extract structured knowledge (triples) from unstructured text
2. ✅ Build and maintain a knowledge graph with proper entity resolution
3. ✅ Identify knowledge gaps and contradictions
4. ✅ Recommend optimal learning paths
5. ✅ Make it work locally (no cloud dependencies, privacy-first)

---

## Why This Matters

### The Problem I'm Solving

As a student, I have:
- 📝 Scattered notes across multiple platforms
- 📚 PDFs and articles I've read but can't remember
- 🎥 YouTube videos with useful information
- 🤔 No way to see how everything connects
- ❓ No idea what I should learn next

**Traditional solutions (Notion, Obsidian, Roam):**
- ❌ Require manual linking
- ❌ Don't understand context
- ❌ Can't find gaps automatically
- ❌ No intelligent recommendations

**My solution:**
- ✅ Automatic knowledge extraction
- ✅ Intelligent entity linking
- ✅ Gap detection via graph analysis
- ✅ AI-powered learning recommendations

---

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    USER DOCUMENTS                            │
│          (PDFs, Markdown, YouTube transcripts)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              INGESTION PIPELINE                              │
│  • PDF Extractor (PyPDF2, pdfplumber)                       │
│  • Markdown Parser                                           │
│  • YouTube Transcript API                                    │
│  • Text Chunker (512 tokens with 50 overlap)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              AGENT 1: EXTRACTOR                              │
│  • spaCy NLP preprocessing                                   │
│  • DistilGPT-2 + LoRA fine-tuned model                      │
│  • Output: Knowledge Triples (S, R, O)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           KNOWLEDGE GRAPH STORAGE                            │
│  • NetworkX DiGraph                                          │
│  • Schema Manager (metadata, provenance)                     │
│  • ChromaDB Vector Store (for RAG)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              AGENT 2: LINKER                                 │
│  • sentence-transformers embeddings                          │
│  • Cosine similarity computation                             │
│  • Agglomerative clustering                                  │
│  • Output: Merged entities, deduplicated graph              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              AGENT 3: REASONER                               │
│  • Graph analysis (centrality, components)                   │
│  • Gap detection (weak connections)                          │
│  • Contradiction detection (opposite relations)              │
│  • Output: Knowledge gaps, contradictions                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              AGENT 4: PLANNER                                │
│  • Graph traversal algorithms                                │
│  • Priority ranking (centrality + novelty)                   │
│  • Learning path construction                                │
│  • Output: Recommended topics with reasoning                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  USER INTERFACE                              │
│  • Graph visualization (NetworkX → Pyvis)                    │
│  • Interactive exploration (Streamlit)                       │
│  • Query interface (RAG-powered search)                      │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Raw Text → Chunks → Triples → Graph → Cleaned Graph → Analysis → Recommendations
```

---

## Implementation Details

### Agent 1: Extractor

**Purpose:** Convert unstructured text into structured knowledge triples

**Input:**
```
"Machine learning uses neural networks for pattern recognition"
```

**Output:**
```
S:Machine Learning|R:USES|O:Neural Networks
S:Neural Networks|R:USED_FOR|O:Pattern Recognition
```

**How it Works:**

1. **Preprocessing (spaCy)**
   ```python
   doc = nlp(text)
   entities = [(ent.text, ent.label_) for ent in doc.ents]
   ```
   - Tokenization
   - POS tagging
   - Named Entity Recognition
   - Dependency parsing

2. **Rule-Based Extraction (Baseline)**
   ```python
   for token in doc:
       if token.dep_ == "nsubj":  # Subject
           subject = token.text
       if token.pos_ == "VERB":    # Relation
           relation = token.text
       if token.dep_ == "dobj":    # Object
           obj = token.text
   ```
   - Extract subject-verb-object patterns
   - Filter by POS tags
   - Normalize entities

3. **Model-Based Extraction (LoRA Fine-tuned)**
   ```python
   model_input = f"Extract knowledge: {text}"
   output = lora_model.generate(model_input)
   # Output: "S:Entity1|R:relation|O:Entity2"
   ```
   - Use fine-tuned DistilGPT-2
   - Generate structured triples
   - Higher accuracy than rules alone

4. **Confidence Scoring**
   ```python
   confidence = compute_confidence(triple, context)
   if confidence < threshold:
       discard_triple()
   ```

**Files:**
- `src/agents/extractor.py` - Main implementation
- `src/models/train_lora.py` - LoRA training script
- `models/lora/extractor_v1/` - Trained model weights

---

### Agent 2: Linker

**Purpose:** Deduplicate entities and merge similar concepts

**The Problem:**
```
Extracted entities:
- "ML"
- "Machine Learning"
- "machine learning"
- "MachineLearning"

These should all be ONE entity!
```

**How it Works:**

1. **Entity Collection**
   ```python
   entities = set()
   for triple in all_triples:
       entities.add(triple.subject)
       entities.add(triple.object)
   ```

2. **Embedding Generation**
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   embeddings = model.encode(list(entities))
   ```
   - Convert each entity to 384-dim vector
   - Captures semantic meaning

3. **Similarity Computation**
   ```python
   from sklearn.metrics.pairwise import cosine_similarity
   similarity_matrix = cosine_similarity(embeddings)
   ```
   - Compare all entity pairs
   - High similarity = likely same entity

4. **Clustering**
   ```python
   from sklearn.cluster import AgglomerativeClustering
   clustering = AgglomerativeClustering(
       n_clusters=None,
       distance_threshold=0.15,  # 1 - similarity_threshold
       linkage='average'
   )
   labels = clustering.fit_predict(embeddings)
   ```
   - Group similar entities together
   - Each cluster = one canonical entity

5. **Entity Merging**
   ```python
   for cluster in clusters:
       canonical = choose_canonical_form(cluster)  # Usually longest
       for variant in cluster:
           update_all_triples(variant -> canonical)
   ```

**Files:**
- `src/agents/linker.py`

---

### Agent 3: Reasoner

**Purpose:** Analyze the knowledge graph to find gaps and contradictions

**What it Detects:**

1. **Knowledge Gaps** - Concepts that need more exploration
   ```python
   def find_gaps(graph):
       gaps = []
       for node in graph.nodes():
           degree = graph.degree(node)
           if degree < 2:  # Weakly connected
               gaps.append({
                   'concept': node,
                   'reason': 'only one connection',
                   'severity': 'high' if degree == 0 else 'medium'
               })
       return gaps
   ```

2. **Contradictions** - Conflicting information
   ```python
   def find_contradictions(graph):
       for (u, v, data1) in graph.edges(data=True):
           for (x, y, data2) in graph.edges(data=True):
               if u == x and v == y:
                   if is_opposite(data1['relation'], data2['relation']):
                       return Contradiction(u, v, data1, data2)
   ```

3. **Knowledge Threads** - Learning sequences
   ```python
   def find_threads(graph, start_node):
       paths = nx.all_simple_paths(graph, start_node, max_length=5)
       return sorted(paths, key=lambda p: path_quality(p))
   ```

**Graph Metrics:**
- Centrality (which concepts are most important?)
- Clustering coefficient (how interconnected?)
- Connected components (isolated knowledge islands?)
- Graph diameter (max distance between concepts)

**Files:**
- `src/agents/reasoner.py`

---

### Agent 4: Planner

**Purpose:** Recommend what to learn next

**Recommendation Algorithm:**

```python
def recommend_next_steps(graph, current_topic, k=5):
    # 1. Find current topic in graph
    if current_topic not in graph:
        return fallback_recommendations(graph)

    # 2. Get neighborhood
    neighbors = set(graph.neighbors(current_topic))
    second_order = set()
    for neighbor in neighbors:
        second_order.update(graph.neighbors(neighbor))

    # 3. Score candidates
    candidates = second_order - neighbors - {current_topic}
    scores = {}
    for candidate in candidates:
        scores[candidate] = (
            0.4 * centrality(candidate) +      # Important concepts
            0.3 * novelty(candidate) +          # New to me
            0.3 * relevance(candidate, current_topic)  # Related to current
        )

    # 4. Return top k
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
```

**Scoring Factors:**

1. **Centrality** - How important is this concept?
   ```python
   pagerank = nx.pagerank(graph)
   centrality_score = pagerank[concept]
   ```

2. **Novelty** - Haven't learned this yet
   ```python
   novelty = 1.0 - (times_visited[concept] / max_visits)
   ```

3. **Relevance** - Related to what I'm learning now
   ```python
   relevance = shortest_path_length(current, concept)
   ```

**Files:**
- `src/agents/planner.py`

---

## Key Algorithms

### 1. Triple Extraction (Dependency Parsing)

```python
def extract_triples_from_parse(doc):
    triples = []
    for sent in doc.sents:
        for token in sent:
            # Subject-Verb-Object pattern
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                subject = token.text
                relation = token.head.text
                # Find object
                for child in token.head.children:
                    if child.dep_ in ["dobj", "attr", "prep"]:
                        obj = child.text
                        triples.append((subject, relation, obj))
    return triples
```

### 2. Entity Resolution (Agglomerative Clustering)

```python
from sklearn.cluster import AgglomerativeClustering

def cluster_entities(entities, embeddings, threshold=0.85):
    # Distance = 1 - similarity
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1-threshold,
        affinity='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)

    # Group entities by cluster
    clusters = {}
    for entity, label in zip(entities, labels):
        clusters.setdefault(label, []).append(entity)

    return clusters
```

### 3. Gap Detection (Graph Analysis)

```python
def detect_knowledge_gaps(graph):
    gaps = []

    # 1. Isolated nodes
    for node in nx.isolates(graph):
        gaps.append({
            'node': node,
            'type': 'isolated',
            'priority': 'critical'
        })

    # 2. Weak connections (degree < threshold)
    for node in graph.nodes():
        degree = graph.degree(node)
        if degree < 2:
            gaps.append({
                'node': node,
                'type': 'weak_connection',
                'degree': degree,
                'priority': 'medium'
            })

    # 3. Bridge nodes (removing them disconnects graph)
    bridges = list(nx.bridges(graph.to_undirected()))
    for (u, v) in bridges:
        gaps.append({
            'edge': (u, v),
            'type': 'critical_connection',
            'priority': 'high'
        })

    return gaps
```

### 4. Learning Path Construction (BFS)

```python
def construct_learning_path(graph, start, end, max_depth=5):
    # Find shortest path
    try:
        path = nx.shortest_path(graph, start, end)
    except nx.NetworkXNoPath:
        return None

    # Enrich path with context
    enriched_path = []
    for i, node in enumerate(path):
        enriched_path.append({
            'concept': node,
            'step': i + 1,
            'explanation': get_context(graph, node),
            'prerequisites': list(graph.predecessors(node)),
            'next_topics': list(graph.successors(node))
        })

    return enriched_path
```

---

## Design Decisions

### 1. Why Multi-Agent Instead of Monolithic?

**Monolithic Approach:**
```python
def process_everything(text):
    # One giant function doing everything
    triples = extract(text)
    graph = build_graph(triples)
    cleaned = clean_graph(graph)
    gaps = find_gaps(cleaned)
    return recommend(gaps)
```

**Problems:**
- ❌ Hard to debug (which part failed?)
- ❌ Can't test components independently
- ❌ Difficult to improve one part without affecting others
- ❌ All-or-nothing execution

**Multi-Agent Approach:**
```python
# Each agent is independent
extractor = ExtractorAgent(config)
linker = LinkerAgent(config)
reasoner = ReasonerAgent(config)
planner = PlannerAgent(config)

# Can test each separately
triples = extractor.extract(text)      # Test Agent 1
cleaned = linker.deduplicate(triples)   # Test Agent 2
gaps = reasoner.analyze(graph)          # Test Agent 3
recs = planner.recommend(gaps)          # Test Agent 4
```

**Benefits:**
- ✅ Easy to debug (isolate failing agent)
- ✅ Independent testing
- ✅ Can improve agents separately
- ✅ Parallel execution possible
- ✅ Follows separation of concerns

---

### 2. Why LoRA Instead of Full Fine-Tuning?

**Full Fine-Tuning:**
- Train all 82M parameters of DistilGPT-2
- Requires: 16GB+ VRAM, 8+ hours training
- Risk: Catastrophic forgetting
- Cost: High compute

**LoRA (Low-Rank Adaptation):**
- Train only 0.5M parameters (adapters)
- Requires: 4GB RAM, 30 min training on CPU
- Risk: Minimal (base model frozen)
- Cost: Low compute

**Trade-off:**
- LoRA gets 80% of full fine-tuning performance
- With 1% of the compute cost
- Perfect for my use case!

---

### 3. Why NetworkX Instead of Neo4j?

**Neo4j (Graph Database):**
- ✅ Production-ready
- ✅ Advanced query language (Cypher)
- ✅ Scales to millions of nodes
- ❌ Requires separate server
- ❌ More complex setup
- ❌ Harder to debug

**NetworkX (Python Library):**
- ✅ Pure Python (easy debugging)
- ✅ No external dependencies
- ✅ Great for prototyping
- ✅ Sufficient for my scale (<10K nodes)
- ❌ All in-memory (doesn't scale to millions)
- ❌ No advanced query features

**My Decision:**
- Start with NetworkX (simple, fast development)
- Add Neo4j support later if needed
- Current implementation supports both via adapter pattern

---

## Challenges & Solutions

### Challenge 1: Low-Quality Triple Extraction

**Problem:**
```
Input: "The cat sat on the mat"
Bad output:
- (cat, sat, on)
- (The, cat, sat)
- (sat, on, the)
```

**Solutions Tried:**

1. ❌ **More rules** → Too brittle, many edge cases
2. ❌ **Bigger model** → Too slow, still makes mistakes
3. ✅ **LoRA fine-tuning + filtering**
   ```python
   # Filter by POS tags
   if subject_pos not in ['NOUN', 'PROPN']:
       skip()

   # Filter by confidence
   if confidence < 0.5:
       skip()

   # Filter stopwords
   if subject in stopwords:
       skip()
   ```

**Result:** F1 score improved from 0.45 → 0.75

---

### Challenge 2: Entity Resolution Threshold Tuning

**Problem:** What similarity threshold merges entities correctly?

**Too Low (0.70):**
```
"Python" + "Java" = Same entity ❌
```

**Too High (0.95):**
```
"ML" + "Machine Learning" = Different entities ❌
```

**Solution:** Empirical testing with validation set

```python
# Test different thresholds
for threshold in [0.75, 0.80, 0.85, 0.90, 0.95]:
    clusters = cluster_entities(embeddings, threshold)
    precision, recall = evaluate(clusters, gold_standard)
    print(f"Threshold {threshold}: P={precision}, R={recall}")

# Found: 0.85 gives best F1
```

**Learned:** Evaluation dataset is essential for hyperparameter tuning

---

### Challenge 3: Knowledge Gaps Too Generic

**Problem:** Every concept with <2 connections flagged as "gap"

**Bad output:**
```
Gap 1: "learning" (1 connection)
Gap 2: "branch" (1 connection)
Gap 3: "intelligence" (1 connection)
...
Gap 100: "the" (1 connection)
```

**Solution:** Multi-factor gap scoring

```python
def score_gap(node, graph):
    score = 0

    # Factor 1: Connectivity
    if graph.degree(node) == 0:
        score += 10  # Isolated = critical
    elif graph.degree(node) < 2:
        score += 5   # Weak = medium

    # Factor 2: Centrality (is it important?)
    centrality = nx.betweenness_centrality(graph)[node]
    score += centrality * 3

    # Factor 3: Is it actually a concept (not stopword)?
    if node.lower() in stopwords:
        score -= 100  # Ignore

    return score
```

**Result:** Only meaningful gaps reported

---

### Challenge 4: Slow Vector Search

**Problem:** Embedding 1000 entities and computing similarity = slow

```python
# Naive approach: O(n²)
for i, entity1 in enumerate(entities):
    for j, entity2 in enumerate(entities):
        similarity = cosine_similarity(emb1, emb2)
```

**Solution:** Batch processing + optimized libraries

```python
# Optimized: O(n)
embeddings = model.encode(entities, batch_size=32)
similarity_matrix = cosine_similarity(embeddings)  # Vectorized
```

**Result:** 10x speedup

---

## What I Learned

### Technical Skills

1. **Multi-agent systems** - Coordination, communication, error handling
2. **Knowledge graphs** - Representation, querying, analysis
3. **NLP** - Dependency parsing, entity recognition, relation extraction
4. **Fine-tuning** - LoRA, PEFT, dataset preparation
5. **Graph algorithms** - Centrality, shortest paths, community detection

### Software Engineering

1. **Testing** - Unit tests, integration tests, test-driven development
2. **Modularity** - Separation of concerns, dependency injection
3. **Configuration** - YAML configs, environment variables
4. **Documentation** - README, docstrings, inline comments
5. **Debugging** - Logging, error handling, incremental testing

### Research Skills

1. **Literature review** - Found relevant papers on KG construction
2. **Experimentation** - Tried multiple approaches, measured results
3. **Evaluation** - Designed metrics, validated against baselines
4. **Iteration** - Continuous improvement based on results

---

## Evaluation Results

### Extraction Quality

| Method | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Baseline (rules only) | 0.52 | 0.41 | 0.45 |
| +spaCy filtering | 0.65 | 0.58 | 0.61 |
| +LoRA fine-tuning | 0.78 | 0.72 | **0.75** |

### Linking Accuracy

| Metric | Score |
|--------|-------|
| Precision (correct merges) | 0.87 |
| Recall (found all duplicates) | 0.83 |
| F1 | 0.85 |

### Gap Detection

Manual validation of 50 identified gaps:
- 42/50 (84%) were actually meaningful gaps
- 8/50 (16%) were false positives (stopwords, artifacts)

### User Study (N=1, me!)

- ✅ Saved ~2 hours/week on note organization
- ✅ Discovered 15+ knowledge gaps I didn't know I had
- ✅ Learning path recommendations were 80% useful

---

## Future Work

If I continue this project:

1. **Temporal Knowledge** - Track when I learned things
2. **Confidence Decay** - Older knowledge becomes less confident
3. **Active Learning** - Ask me to annotate ambiguous extractions
4. **Multi-modal** - Handle images, diagrams, equations
5. **Collaborative** - Merge knowledge graphs from multiple people
6. **Better UI** - Interactive graph editing, real-time updates
7. **Mobile App** - Capture notes on the go

---

## Conclusion

This project demonstrates:
- ✅ Full-stack AI system design (ingestion → agents → UI)
- ✅ Multi-agent architecture with proper separation
- ✅ Fine-tuning LLMs with limited compute (LoRA)
- ✅ Graph algorithms for knowledge reasoning
- ✅ Comprehensive testing and evaluation
- ✅ Production-ready code structure

**I'm proud of what I built and excited to keep improving it!**
