"""Test script to verify all agents are working correctly"""

import sys
from pathlib import Path
import io

# Fix Windows encoding issue
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logging
from src.agents.extractor import BaselineExtractor
from src.agents.linker import LinkerAgent
from src.agents.reasoner import ReasonerAgent
from src.agents.planner import PlannerAgent
from src.graph.graph_store import KnowledgeGraphStore
from loguru import logger


def test_agent_1_extractor():
    """Test Agent 1: Extractor"""
    print("\n" + "="*80)
    print("TESTING AGENT 1: EXTRACTOR")
    print("="*80)

    try:
        config = load_config('configs/config.yaml')
        extractor = BaselineExtractor(config)

        # Test with sample text
        test_text = """
        Machine learning is a branch of artificial intelligence.
        Neural networks are used for deep learning tasks.
        Python is a popular programming language for data science.
        """

        print(f"\n[INPUT] Input text:\n{test_text.strip()}\n")

        triples = extractor.extract_triples(test_text, source='test')

        print(f"[PASS] Extractor Agent is WORKING!")
        print(f"[INFO] Extracted {len(triples)} triples:")
        for i, triple in enumerate(triples, 1):
            print(f"   {i}. {triple}")

        return True, triples
    except Exception as e:
        print(f"[FAIL] Extractor Agent FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def test_agent_2_linker(triples):
    """Test Agent 2: Linker (Entity Deduplication)"""
    print("\n" + "="*80)
    print("[TEST] TESTING AGENT 2: LINKER")
    print("="*80)

    try:
        config = load_config('configs/config.yaml')
        linker = LinkerAgent(config)

        # Add entities from triples
        triple_dicts = [t.to_dict() for t in triples]
        linker.add_entities_from_triples(triple_dicts)

        print(f"\n[INFO] Added {len(linker.entities)} entities from triples")

        # Compute embeddings
        linker.compute_embeddings()
        print(f"[PASS] Computed embeddings for entities")

        # Find clusters
        clusters = linker.cluster_entities()
        print(f"[PASS] Linker Agent is WORKING!")
        print(f"[INFO] Found {len(clusters)} entity clusters")

        # Convert to list if it's a dict
        cluster_list = list(clusters.items()) if isinstance(clusters, dict) else list(clusters)
        for i, cluster in enumerate(cluster_list[:5], 1):
            print(f"   Cluster {i}: {cluster}")

        return True
    except Exception as e:
        print(f"[FAIL] Linker Agent FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_3_reasoner(kg_store):
    """Test Agent 3: Reasoner (Gap Detection)"""
    print("\n" + "="*80)
    print("[TEST] TESTING AGENT 3: REASONER")
    print("="*80)

    try:
        config = load_config('configs/config.yaml')
        reasoner = ReasonerAgent(config)

        # Analyze the graph
        analysis = reasoner.analyze_graph(kg_store)

        gaps = analysis.get('gaps', [])
        contradictions = analysis.get('contradictions', [])
        threads = analysis.get('threads', [])

        print(f"[PASS] Reasoner Agent is WORKING!")
        print(f"[INFO] Analysis Results:")
        print(f"   - Knowledge gaps found: {len(gaps)}")
        print(f"   - Contradictions found: {len(contradictions)}")
        print(f"   - Knowledge threads: {len(threads)}")

        if gaps:
            print(f"\n   Sample gaps:")
            for i, gap in enumerate(gaps[:3], 1):
                if isinstance(gap, dict):
                    print(f"      {i}. {gap.get('description', 'Gap detected')}")
                else:
                    print(f"      {i}. {gap}")

        return True
    except Exception as e:
        print(f"[FAIL] Reasoner Agent FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_4_planner(kg_store):
    """Test Agent 4: Planner (Learning Recommendations)"""
    print("\n" + "="*80)
    print("[TEST] TESTING AGENT 4: PLANNER")
    print("="*80)

    try:
        config = load_config('configs/config.yaml')
        planner = PlannerAgent(config)

        # Get recommendations
        recommendations = planner.recommend_next_steps(kg_store, current_topic="machine learning")

        print(f"[PASS] Planner Agent is WORKING!")
        print(f"[INFO] Generated {len(recommendations)} recommendations:")

        for i, rec in enumerate(recommendations[:5], 1):
            print(f"   {i}. {rec.concept} (Priority: {rec.priority:.2f})")
            print(f"      Reason: {rec.reasoning}")

        return True
    except Exception as e:
        print(f"[FAIL] Planner Agent FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_agent_collaboration():
    """Test all agents working together"""
    print("\n" + "="*80)
    print("TESTING MULTI-AGENT COLLABORATION")
    print("="*80)

    try:
        config = load_config('configs/config.yaml')

        # Step 1: Extract knowledge
        print("\n[Step 1] Extractor Agent extracting knowledge...")
        extractor = BaselineExtractor(config)
        text = """
        Artificial intelligence encompasses machine learning and deep learning.
        Neural networks are the foundation of deep learning.
        Transformers revolutionized natural language processing.
        GPT models use transformer architecture for text generation.
        """
        triples = extractor.extract_triples(text, source='collaboration_test')
        print(f"   [PASS] Extracted {len(triples)} triples")

        # Step 2: Build knowledge graph
        print("\n[Step 2] Building knowledge graph...")
        kg_store = KnowledgeGraphStore(config)
        triple_dicts = [t.to_dict() for t in triples]
        kg_store.add_triples(triple_dicts)
        stats = kg_store.get_statistics()
        print(f"   [PASS] Graph has {stats['graph']['num_nodes']} nodes and {stats['graph']['num_edges']} edges")

        # Step 3: Link entities
        print("\n[Step 3] Linker Agent deduplicating entities...")
        linker = LinkerAgent(config)
        linker.add_entities_from_triples(triple_dicts)
        linker.compute_embeddings()
        clusters = linker.cluster_entities()
        print(f"   [PASS] Found {len(clusters)} entity clusters")

        # Step 4: Reason about knowledge
        print("\n[Step 4] Reasoner Agent analyzing knowledge gaps...")
        reasoner = ReasonerAgent(config)
        analysis = reasoner.analyze_graph(kg_store)
        print(f"   [PASS] Found {len(analysis['gaps'])} gaps, {len(analysis['contradictions'])} contradictions")

        # Step 5: Plan learning path
        print("\n[Step 5] Planner Agent generating recommendations...")
        planner = PlannerAgent(config)
        recommendations = planner.recommend_next_steps(kg_store, current_topic="AI")
        print(f"   [PASS] Generated {len(recommendations)} learning recommendations")

        print("\n[PASS] MULTI-AGENT COLLABORATION IS WORKING!")
        return True

    except Exception as e:
        print(f"[FAIL] Multi-agent collaboration FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all agent tests"""
    print("\n" + "="*80)
    print("[TEST] KNOWLEDGE WEAVER - AGENT TESTING SUITE")
    print("="*80)

    setup_logging({'logging': {'level': 'WARNING'}})  # Reduce noise

    results = {}

    # Test individual agents
    success, triples = test_agent_1_extractor()
    results['Agent 1: Extractor'] = success

    if success and triples:
        # Build a test knowledge graph for other agents
        config = load_config('configs/config.yaml')
        kg_store = KnowledgeGraphStore(config)
        triple_dicts = [t.to_dict() for t in triples]
        kg_store.add_triples(triple_dicts)

        results['Agent 2: Linker'] = test_agent_2_linker(triples)
        results['Agent 3: Reasoner'] = test_agent_3_reasoner(kg_store)
        results['Agent 4: Planner'] = test_agent_4_planner(kg_store)
    else:
        print("\n[WARN] Skipping other agent tests due to Extractor failure")
        results['Agent 2: Linker'] = False
        results['Agent 3: Reasoner'] = False
        results['Agent 4: Planner'] = False

    # Test collaboration
    results['Multi-Agent Collaboration'] = test_multi_agent_collaboration()

    # Summary
    print("\n" + "="*80)
    print("[INFO] TEST SUMMARY")
    print("="*80)

    for agent, status in results.items():
        status_icon = "[PASS]" if status else "[FAIL]"
        status_text = "PASS" if status else "FAIL"
        print(f"{status_icon} {agent}: {status_text}")

    total = len(results)
    passed = sum(1 for s in results.values() if s)

    print(f"\n[INFO] Overall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n[SUCCESS] ALL AGENTS ARE WORKING CORRECTLY!")
    else:
        print(f"\n[WARN] {total - passed} agent(s) need attention")

    print("="*80 + "\n")


if __name__ == '__main__':
    main()
