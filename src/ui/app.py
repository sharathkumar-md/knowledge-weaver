"""
Knowledge Weaver - Streamlit UI
A clean, functional interface for the multi-agent knowledge graph system
"""

import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
import networkx as nx
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config
from src.agents.extractor import ExtractorAgent
from src.agents.linker import LinkerAgent
from src.agents.reasoner import ReasonerAgent
from src.agents.planner import PlannerAgent
from src.graph.graph_store import NetworkXStore, Neo4jStore
from src.ingestion.pipeline import IngestionPipeline
from loguru import logger


# Page configuration
st.set_page_config(
    page_title="Knowledge Weaver",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-success {
        color: #28a745;
        font-weight: 600;
    }
    .status-error {
        color: #dc3545;
        font-weight: 600;
    }
    .status-warning {
        color: #ffc107;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    """Load and cache the system configuration and agents"""
    try:
        config = load_config()

        # Initialize graph store
        backend = config.get('graph', {}).get('backend', 'networkx')
        if backend == 'neo4j':
            graph_store = Neo4jStore(config)
        else:
            graph_store = NetworkXStore(config)

        # Initialize agents
        extractor = ExtractorAgent(config)
        if extractor.use_lora and extractor.lora_extractor is not None:
            extractor_type = "LoRA Fine-tuned"
        else:
            extractor_type = "Baseline"

        linker = LinkerAgent(config)
        reasoner = ReasonerAgent(config)
        planner = PlannerAgent(config)

        return {
            'config': config,
            'graph_store': graph_store,
            'extractor': extractor,
            'extractor_type': extractor_type,
            'linker': linker,
            'reasoner': reasoner,
            'planner': planner
        }
    except Exception as e:
        st.error(f"Failed to load system: {e}")
        return None


def create_graph_visualization(graph_store):
    """Create an interactive graph visualization using Plotly"""
    G = graph_store.graph

    if len(G.nodes()) == 0:
        return None

    # Use spring layout for node positions
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Create edges
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Create nodes
    node_trace = go.Scatter(
        x=[], y=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=15,
            colorbar=dict(
                thickness=15,
                title='Connections',
                xanchor='left',
                titleside='right'
            )
        ),
        text=[],
        textposition="top center"
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_data = G.nodes[node]
        node_label = node_data.get('label', node)
        node_trace['text'] += tuple([node_label])

    # Color nodes by degree
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
    node_trace.marker.color = node_adjacencies

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0, l=0, r=0, t=0),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=600,
                       plot_bgcolor='white'
                   ))

    return fig


def dashboard_page(system):
    """Main dashboard showing system overview"""
    st.markdown('<div class="main-header">Knowledge Weaver Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Agent Knowledge Graph System</div>', unsafe_allow_html=True)

    # System status
    col1, col2, col3, col4 = st.columns(4)

    graph_store = system['graph_store']
    G = graph_store.graph

    with col1:
        st.metric("Total Concepts", len(G.nodes()))

    with col2:
        st.metric("Total Relations", len(G.edges()))

    with col3:
        backend = system['config'].get('graph', {}).get('backend', 'networkx')
        st.metric("Graph Backend", backend.upper())

    with col4:
        st.metric("Extractor Type", system['extractor_type'])

    st.divider()

    # Graph statistics
    if len(G.nodes()) > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Graph Statistics")
            stats = graph_store.get_statistics()

            stats_df = pd.DataFrame([
                {"Metric": "Nodes", "Value": stats.get('num_nodes', 0)},
                {"Metric": "Edges", "Value": stats.get('num_edges', 0)},
                {"Metric": "Density", "Value": f"{stats.get('density', 0):.4f}"},
                {"Metric": "Avg Degree", "Value": f"{stats.get('avg_degree', 0):.2f}"},
                {"Metric": "Connected Components", "Value": stats.get('num_components', 0)}
            ])

            st.dataframe(stats_df, hide_index=True, use_container_width=True)

        with col2:
            st.subheader("Top Concepts")

            # Get nodes sorted by degree
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]

            top_df = pd.DataFrame([
                {
                    "Concept": G.nodes[node]['label'] if 'label' in G.nodes[node] else node,
                    "Connections": degree
                }
                for node, degree in top_nodes
            ])

            st.dataframe(top_df, hide_index=True, use_container_width=True)

        st.divider()

        # Graph visualization
        st.subheader("Knowledge Graph Visualization")
        fig = create_graph_visualization(graph_store)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No knowledge graph data available. Extract knowledge from documents to get started.")


def extraction_page(system):
    """Page for extracting knowledge from text"""
    st.markdown('<div class="main-header">Knowledge Extraction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Extract concepts and relationships from text</div>', unsafe_allow_html=True)

    extractor = system['extractor']
    graph_store = system['graph_store']
    linker = system['linker']

    # Text input
    st.subheader("Input Text")
    text_input = st.text_area(
        "Enter text to extract knowledge from:",
        height=200,
        placeholder="Example: Machine learning is a subset of artificial intelligence. Deep learning uses neural networks for pattern recognition."
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        extract_button = st.button("Extract Knowledge", type="primary", use_container_width=True)

    if extract_button and text_input:
        with st.spinner("Extracting knowledge triples..."):
            try:
                # Extract triples
                triples = extractor.extract(text_input)

                if triples:
                    st.success(f"Extracted {len(triples)} knowledge triples")

                    # Display triples
                    st.subheader("Extracted Triples")
                    triples_data = []
                    for triple in triples:
                        triples_data.append({
                            "Subject": triple.subject,
                            "Relation": triple.relation,
                            "Object": triple.object,
                            "Confidence": f"{triple.confidence:.2f}"
                        })

                    df = pd.DataFrame(triples_data)
                    st.dataframe(df, hide_index=True, use_container_width=True)

                    # Add to graph option
                    if st.button("Add to Knowledge Graph"):
                        with st.spinner("Adding to graph and performing entity linking..."):
                            # Add triples to graph
                            for triple in triples:
                                # Create provenance
                                provenance = {
                                    'source': 'manual_input',
                                    'timestamp': datetime.now().isoformat(),
                                    'confidence': triple.confidence
                                }

                                # Add nodes and edges
                                from src.graph.schema_manager import NodeSchema, EdgeSchema, Provenance

                                subject_node = NodeSchema(
                                    id=triple.subject.lower().replace(' ', '_'),
                                    label=triple.subject,
                                    type='concept',
                                    provenance=[Provenance(**provenance)]
                                )

                                object_node = NodeSchema(
                                    id=triple.object.lower().replace(' ', '_'),
                                    label=triple.object,
                                    type='concept',
                                    provenance=[Provenance(**provenance)]
                                )

                                edge = EdgeSchema(
                                    source_id=subject_node.id,
                                    target_id=object_node.id,
                                    relation=triple.relation,
                                    type='semantic',
                                    confidence=triple.confidence,
                                    provenance=[Provenance(**provenance)]
                                )

                                graph_store.add_node(subject_node)
                                graph_store.add_node(object_node)
                                graph_store.add_edge(edge)

                            # Perform entity linking
                            entities = [triple.subject for triple in triples] + [triple.object for triple in triples]
                            clusters = linker.cluster_entities(entities)

                            st.success(f"Added {len(triples)} triples to knowledge graph")
                            st.info(f"Entity linking created {len(clusters)} clusters from {len(entities)} entities")

                            # Clear cache to reload graph
                            st.cache_resource.clear()
                            st.rerun()
                else:
                    st.warning("No triples extracted from the text")

            except Exception as e:
                st.error(f"Error during extraction: {e}")
                logger.error(f"Extraction error: {e}")


def analysis_page(system):
    """Page for analyzing the knowledge graph"""
    st.markdown('<div class="main-header">Knowledge Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Discover gaps and insights in your knowledge graph</div>', unsafe_allow_html=True)

    reasoner = system['reasoner']
    planner = system['planner']
    graph_store = system['graph_store']

    if len(graph_store.graph.nodes()) == 0:
        st.warning("No knowledge graph data available. Extract knowledge first.")
        return

    tab1, tab2, tab3 = st.tabs(["Knowledge Gaps", "Learning Recommendations", "Search"])

    with tab1:
        st.subheader("Detected Knowledge Gaps")

        if st.button("Analyze Knowledge Gaps", type="primary"):
            with st.spinner("Analyzing knowledge graph for gaps..."):
                try:
                    gaps = reasoner.find_knowledge_gaps()

                    if gaps:
                        st.success(f"Found {len(gaps)} knowledge gaps")

                        gaps_data = []
                        for gap in gaps[:20]:  # Show top 20
                            gaps_data.append({
                                "Concept 1": gap.get('source', 'N/A'),
                                "Concept 2": gap.get('target', 'N/A'),
                                "Gap Type": gap.get('type', 'missing_link'),
                                "Importance": f"{gap.get('importance', 0):.2f}"
                            })

                        df = pd.DataFrame(gaps_data)
                        st.dataframe(df, hide_index=True, use_container_width=True)
                    else:
                        st.info("No significant knowledge gaps detected")

                except Exception as e:
                    st.error(f"Error analyzing gaps: {e}")
                    logger.error(f"Gap analysis error: {e}")

    with tab2:
        st.subheader("Learning Path Recommendations")

        current_topic = st.text_input("Enter a topic you're currently learning:")

        if st.button("Get Recommendations", type="primary") and current_topic:
            with st.spinner("Generating learning recommendations..."):
                try:
                    recommendations = planner.recommend_learning_path(current_topic)

                    if recommendations:
                        st.success(f"Generated {len(recommendations)} recommendations")

                        for i, rec in enumerate(recommendations[:10], 1):
                            with st.expander(f"{i}. {rec.get('topic', 'Unknown')}"):
                                st.write(f"**Relevance:** {rec.get('relevance', 0):.2f}")
                                st.write(f"**Reason:** {rec.get('reason', 'No reason provided')}")

                                if 'prerequisites' in rec:
                                    st.write(f"**Prerequisites:** {', '.join(rec['prerequisites'])}")
                    else:
                        st.info("No recommendations available for this topic")

                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
                    logger.error(f"Recommendation error: {e}")

    with tab3:
        st.subheader("Search Knowledge Graph")

        search_query = st.text_input("Search for a concept:")

        if search_query:
            G = graph_store.graph
            matching_nodes = [
                node for node in G.nodes()
                if search_query.lower() in G.nodes[node].get('label', '').lower()
            ]

            if matching_nodes:
                st.success(f"Found {len(matching_nodes)} matching concepts")

                for node_id in matching_nodes[:10]:
                    node_data = G.nodes[node_id]
                    with st.expander(node_data.get('label', node_id)):
                        st.write(f"**ID:** {node_id}")
                        st.write(f"**Type:** {node_data.get('type', 'N/A')}")

                        # Show connections
                        neighbors = list(G.neighbors(node_id))
                        if neighbors:
                            st.write(f"**Connected to:** {len(neighbors)} concepts")
                            neighbor_labels = [G.nodes[n].get('label', n) for n in neighbors[:5]]
                            st.write(", ".join(neighbor_labels))
            else:
                st.info("No matching concepts found")


def settings_page(system):
    """Page for system settings"""
    st.markdown('<div class="main-header">System Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Configure the knowledge weaver system</div>', unsafe_allow_html=True)

    config = system['config']

    st.subheader("Current Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Model Settings**")
        st.text(f"Base Model: {config.get('models', {}).get('base_model', 'N/A')}")
        st.text(f"Embedding Model: {config.get('models', {}).get('embedding_model', 'N/A')}")

        st.write("")
        st.write("**Graph Settings**")
        st.text(f"Backend: {config.get('graph', {}).get('backend', 'N/A')}")

    with col2:
        st.write("**Agent Settings**")
        extractor_config = config.get('agents', {}).get('extractor', {})
        st.text(f"Confidence Threshold: {extractor_config.get('confidence_threshold', 0.5)}")
        st.text(f"Max Triples: {extractor_config.get('max_triples_per_chunk', 10)}")

        st.write("")
        st.write("**System Info**")
        st.text(f"Extractor: {system['extractor_type']}")

    st.divider()

    st.subheader("Graph Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Export Graph (JSON)", use_container_width=True):
            try:
                output_path = project_root / "output" / "graph_export.json"
                output_path.parent.mkdir(exist_ok=True)
                system['graph_store'].export(str(output_path))
                st.success(f"Graph exported to {output_path}")
            except Exception as e:
                st.error(f"Export failed: {e}")

    with col2:
        if st.button("Export Graph (GEXF)", use_container_width=True):
            try:
                output_path = project_root / "output" / "graph_export.gexf"
                output_path.parent.mkdir(exist_ok=True)
                nx.write_gexf(system['graph_store'].graph, str(output_path))
                st.success(f"Graph exported to {output_path}")
            except Exception as e:
                st.error(f"Export failed: {e}")

    with col3:
        if st.button("Clear Graph", type="secondary", use_container_width=True):
            if st.session_state.get('confirm_clear', False):
                system['graph_store'].graph.clear()
                st.cache_resource.clear()
                st.success("Graph cleared")
                st.session_state['confirm_clear'] = False
                st.rerun()
            else:
                st.session_state['confirm_clear'] = True
                st.warning("Click again to confirm")


def main():
    """Main application"""

    # Sidebar navigation
    st.sidebar.title("Knowledge Weaver")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Extract Knowledge", "Analysis", "Settings"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "Knowledge Weaver is a multi-agent system that automatically builds "
        "knowledge graphs from your documents and provides intelligent insights."
    )

    # Load system
    system = load_system()

    if system is None:
        st.error("Failed to initialize system. Please check configuration.")
        return

    # Route to pages
    if page == "Dashboard":
        dashboard_page(system)
    elif page == "Extract Knowledge":
        extraction_page(system)
    elif page == "Analysis":
        analysis_page(system)
    elif page == "Settings":
        settings_page(system)


if __name__ == "__main__":
    main()
