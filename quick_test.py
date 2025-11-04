"""Quick test to verify core functionality"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if core modules can be imported"""
    print("=" * 60)
    print("TESTING CORE IMPORTS")
    print("=" * 60)

    tests = {
        "PyTorch": lambda: __import__('torch'),
        "Transformers": lambda: __import__('transformers'),
        "NetworkX": lambda: __import__('networkx'),
        "ChromaDB": lambda: __import__('chromadb'),
        "spaCy": lambda: __import__('spacy'),
        "Config Utils": lambda: __import__('src.utils'),
    }

    results = {}
    for name, test_fn in tests.items():
        try:
            test_fn()
            print(f"[PASS] {name}")
            results[name] = True
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            results[name] = False

    return results

def test_config():
    """Test if configuration loads"""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)

    try:
        from src.utils import load_config
        config = load_config('configs/config.yaml')
        print(f"[PASS] Configuration loaded successfully")
        print(f"[INFO] Base model: {config['models']['base_model']}")
        print(f"[INFO] Graph backend: {config['graph']['backend']}")
        return True
    except Exception as e:
        print(f"[FAIL] Configuration loading failed: {e}")
        return False

def test_basic_extraction():
    """Test basic knowledge extraction"""
    print("\n" + "=" * 60)
    print("TESTING BASIC EXTRACTION")
    print("=" * 60)

    try:
        from src.utils import load_config
        from src.agents.extractor import BaselineExtractor

        config = load_config('configs/config.yaml')
        extractor = BaselineExtractor(config)

        test_text = "Python is a programming language. Machine learning uses Python."
        triples = extractor.extract_triples(test_text, source='test')

        print(f"[PASS] Extraction working")
        print(f"[INFO] Extracted {len(triples)} triples from test text")
        for i, triple in enumerate(triples[:3], 1):
            print(f"  {i}. {triple}")

        return True
    except Exception as e:
        print(f"[FAIL] Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_graph_storage():
    """Test graph storage"""
    print("\n" + "=" * 60)
    print("TESTING GRAPH STORAGE")
    print("=" * 60)

    try:
        from src.utils import load_config
        from src.graph.graph_store import KnowledgeGraphStore

        config = load_config('configs/config.yaml')
        kg = KnowledgeGraphStore(config)

        # Add a simple triple
        kg.add_triples([{
            'subject': 'Python',
            'predicate': 'is_a',
            'object': 'programming_language',
            'source': 'test',
            'confidence': 1.0
        }])

        stats = kg.get_statistics()
        print(f"[PASS] Graph storage working")
        print(f"[INFO] Nodes: {stats['graph']['num_nodes']}, Edges: {stats['graph']['num_edges']}")

        return True
    except Exception as e:
        print(f"[FAIL] Graph storage failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 60)
    print("KNOWLEDGE WEAVER - QUICK TEST SUITE")
    print("=" * 60 + "\n")

    results = {}

    # Test imports
    import_results = test_imports()
    results.update(import_results)

    # Test configuration
    results['Configuration'] = test_config()

    # Test extraction
    results['Extraction'] = test_basic_extraction()

    # Test graph storage
    results['Graph Storage'] = test_graph_storage()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test, status in results.items():
        icon = "[PASS]" if status else "[FAIL]"
        print(f"{icon} {test}")

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n[SUCCESS] ALL CORE COMPONENTS WORKING!")
        print("Your project is ready for submission!")
    else:
        print(f"\n[WARN] {total - passed} component(s) need attention")
        print("Some optional dependencies may be missing.")

    print("=" * 60 + "\n")

    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
