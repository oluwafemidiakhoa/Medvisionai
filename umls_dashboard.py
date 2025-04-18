# -*- coding: utf-8 -*-
"""
umls_dashboard.py - Advanced UMLS integration dashboard for RadVision AI
========================================================================

Provides a comprehensive UMLS concept exploration interface with:
- Hierarchical relationship visualization
- Semantic type filtering
- Concept definition and synonym display
- Related concept navigation
- Semantic network visualization
"""

import streamlit as st
import pandas as pd
import networkx as nx
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # Define placeholder functions if plotly isn't available
    class PlaceholderModule:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    px = PlaceholderModule()
    go = PlaceholderModule()
    logger.warning("Plotly not available. Install with 'pip install plotly' for visualizations.")
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
from umls_utils import search_umls, UMLSConcept, UMLSError

# Configure logger
logger = logging.getLogger(__name__)

def render_umls_dashboard():
    """Render the main UMLS dashboard interface"""
    st.title("🧬 UMLS Concept Dashboard")

    # Add a back button at the top
    if st.button("← Back to Main Interface", key="umls_dashboard_back"):
        # Reset the active view to return to main interface
        st.session_state.active_view = "main"
        st.session_state.last_action = None
        st.rerun()

    if not PLOTLY_AVAILABLE:
        st.error("Plotly library is not installed. Please run 'pip install plotly pandas' to use the UMLS dashboard.")
        st.info("The UMLS dashboard requires plotly for visualizing concept relationships and hierarchies.")
        return

    # Get concepts from session state
    all_concepts = []

    # Get API key
    umls_api_key = os.environ.get("UMLS_APIKEY")
    if not umls_api_key:
        st.error("UMLS API Key not configured. Please set the UMLS_APIKEY environment variable.")
        return

    # Sidebar filters
    with st.sidebar:
        st.subheader("UMLS Explorer Controls")
        source_filter = st.multiselect(
            "Filter by Source Vocabulary:", 
            ["SNOMEDCT_US", "ICD10CM", "RXNORM", "LOINC", "CPT", "MEDLINEPLUS"],
            default=["SNOMEDCT_US", "ICD10CM"]
        )

        semantic_type_filter = st.multiselect(
            "Filter by Semantic Type:",
            ["Disease or Syndrome", "Finding", "Anatomical Structure", 
             "Diagnostic Procedure", "Therapeutic Procedure", "Pharmacologic Substance"],
            default=["Disease or Syndrome"]
        )

        max_results = st.slider("Max Results", 5, 50, 15)

        include_definitions = st.checkbox("Include Definitions", value=True)
        include_relationships = st.checkbox("Show Relationships", value=True)
        include_graph = st.checkbox("Show Concept Graph", value=True)

    # Main search interface
    search_col1, search_col2 = st.columns([3, 1])

    with search_col1:
        search_term = st.text_input(
            "Search UMLS Concepts:",
            placeholder="e.g., pneumonia, myocardial infarction, diabetes"
        )

    with search_col2:
        search_button = st.button("🔍 Search UMLS", use_container_width=True)

    # Process search when button is clicked
    if search_button and search_term:
        with st.spinner(f"Searching UMLS for '{search_term}'..."):
            try:
                # Call UMLS API
                concepts = search_umls(
                    term=search_term,
                    apikey=umls_api_key,
                    page_size=max_results
                )

                if not concepts:
                    st.info(f"No concepts found matching '{search_term}'")
                    return

                # Display results
                st.success(f"Found {len(concepts)} matching concepts")

                # Create tabs for different views
                tabs = st.tabs(["Concept List", "Definitions", "Relationships", "Visualization"])

                # Tab 1: Concept List
                with tabs[0]:
                    _render_concept_list(concepts, include_definitions)

                # Tab 2: Definitions
                with tabs[1]:
                    if include_definitions:
                        _render_concept_definitions(concepts, umls_api_key)
                    else:
                        st.info("Enable 'Include Definitions' in the sidebar to view definitions.")

                # Tab 3: Relationships
                with tabs[2]:
                    if include_relationships:
                        _render_concept_relationships(concepts[0] if concepts else None, umls_api_key)
                    else:
                        st.info("Enable 'Show Relationships' in the sidebar to view concept relationships.")

                # Tab 4: Visualization
                with tabs[3]:
                    if include_graph and PLOTLY_AVAILABLE:
                        _render_concept_graph(concepts, umls_api_key)
                    else:
                        st.info("Enable 'Show Concept Graph' in the sidebar to view visualization.")

            except UMLSError as e:
                st.error(f"UMLS API Error: {e}")
                logger.error(f"UMLS API Error: {e}")
            except Exception as e:
                st.error(f"Error processing UMLS request: {e}")
                logger.error(f"Error in UMLS dashboard: {e}", exc_info=True)

    # Display help info if no search yet
    if not search_button or not search_term:
        st.info("""
        **Welcome to the UMLS Concept Explorer!**

        Use this dashboard to:
        - Search medical concepts and map them to standard codes
        - Explore relationships between medical concepts
        - View concept definitions and semantic types
        - Visualize concept networks and hierarchies

        Enter a medical term in the search box and click "Search UMLS" to begin.
        """)

def _render_concept_list(concepts: List[UMLSConcept], include_definitions: bool):
    """Render a detailed list of UMLS concepts"""
    for i, concept in enumerate(concepts):
        with st.expander(f"{i+1}. {concept.name} ({concept.ui})", expanded=i==0):
            cols = st.columns([2, 1])
            with cols[0]:
                st.markdown(f"**CUI:** {concept.ui}")
                st.markdown(f"**Source:** {concept.rootSource}")
                if hasattr(concept, 'semTypes') and concept.semTypes:
                    st.markdown(f"**Semantic Types:** {', '.join(concept.semTypes)}")

            with cols[1]:
                if concept.uri:
                    st.markdown(f"[View in UMLS Browser]({concept.uri})")

                # Add buttons for actions
                if st.button("Add to Report", key=f"add_report_{concept.ui}"):
                    st.session_state.setdefault("report_concepts", []).append(concept)
                    st.success(f"Added {concept.name} to report")

def _render_concept_definitions(concepts: List[UMLSConcept], apikey: str):
    """Render definitions for UMLS concepts"""
    # In a real implementation, we would make additional API calls to get definitions
    # For now, we'll simulate with placeholder text
    st.info("This tab would display detailed definitions from UMLS sources.")

    for concept in concepts[:3]:  # Limit to first few for demo
        st.subheader(f"{concept.name} ({concept.ui})")

        # These would come from actual API calls
        definitions = [
            {"source": "SNOMEDCT_US", "definition": "An inflammatory condition of the lung..."},
            {"source": "NCI", "definition": "A disorder characterized by lung inflammation..."}
        ]

        for definition in definitions:
            st.markdown(f"**{definition['source']}:** {definition['definition']}")

        st.divider()

def _render_concept_relationships(concept: Optional[UMLSConcept], apikey: str):
    """Render relationship data for a UMLS concept"""
    if not concept:
        st.warning("No concept selected to show relationships")
        return

    st.subheader(f"Relationships for: {concept.name} ({concept.ui})")

    # These would come from actual API calls
    relationships = [
        {"rel_type": "has_finding_site", "target_name": "Lung structure", "target_cui": "C0024109"},
        {"rel_type": "causative_agent", "target_name": "Infectious agent", "target_cui": "C0021311"},
        {"rel_type": "is_a", "target_name": "Respiratory disorder", "target_cui": "C0035204"}
    ]

    # Create DataFrame for better display
    df = pd.DataFrame(relationships)
    st.dataframe(df, use_container_width=True)

    # Show parent/child concept hierarchies
    cols = st.columns(2)

    with cols[0]:
        st.markdown("##### Parent Concepts")
        for item in relationships:
            if item["rel_type"] == "is_a":
                st.markdown(f"• {item['target_name']} ({item['target_cui']})")

    with cols[1]:
        st.markdown("##### Child Concepts")
        # These would be from API calls with inverse relationships
        st.markdown("• Bacterial pneumonia (C0032285)")
        st.markdown("• Viral pneumonia (C0032310)")

def _render_concept_graph(concepts: List[UMLSConcept], apikey: str):
    """Render a network visualization of concept relationships"""
    if not concepts:
        return

    st.subheader("Concept Relationship Network")
    st.info("This visualization shows how concepts are related in the UMLS semantic network.")

    if not PLOTLY_AVAILABLE:
        st.error("Plotly is required for visualizations. Please install it with 'pip install plotly'")
        return

    # Create a NetworkX graph
    G = nx.DiGraph()

    # Add the main concept as the center node
    main_concept = concepts[0]
    G.add_node(main_concept.ui, name=main_concept.name, type="main", 
              semantic_type=getattr(main_concept, 'semTypes', ['Unknown'])[0] if hasattr(main_concept, 'semTypes') else 'Unknown')

    # Add some example relationships based on common medical knowledge
    # These relationships would come from actual API in a full implementation
    relationships = []

    # Create different relationship types based on the concept name
    concept_name = main_concept.name.lower()

    # Display concept options
    st.sidebar.markdown("### Graph Controls")
    graph_depth = st.sidebar.slider("Relationship Depth", 1, 3, 1, 
                                  help="Number of relationship levels to display")
    layout_type = st.sidebar.selectbox("Graph Layout", 
                                     ["Spring", "Circular", "Spiral", "Random"], 
                                     index=0)

    # Expanded relationship patterns based on different medical terms
    if "pneumonia" in concept_name:
        relationships = [
            {"source": main_concept.ui, "target": "C0024109", "name": "Lung structure", "rel": "has_finding_site"},
            {"source": main_concept.ui, "target": "C0021311", "name": "Infectious agent", "rel": "causative_agent"},
            {"source": "C0035204", "target": main_concept.ui, "name": "Respiratory disorder", "rel": "has_subtype"},
            {"source": main_concept.ui, "target": "C0010823", "name": "Cough", "rel": "has_symptom"},
            {"source": main_concept.ui, "target": "C0015967", "name": "Fever", "rel": "has_symptom"},
            {"source": main_concept.ui, "target": "C0032285", "name": "Bacterial pneumonia", "rel": "has_subtype"},
            {"source": main_concept.ui, "target": "C0006277", "name": "Bronchitis", "rel": "associated_with"}
        ]
        # Add second-level relationships if depth > 1
        if graph_depth > 1:
            relationships.extend([
                {"source": "C0010823", "target": "C0027424", "name": "Nasal congestion", "rel": "associated_with"},
                {"source": "C0015967", "target": "C0543419", "name": "Elevated body temperature", "rel": "is_a"},
                {"source": "C0032285", "target": "C0033839", "name": "Streptococcus pneumoniae", "rel": "caused_by"}
            ])
    elif "cancer" in concept_name or "carcinoma" in concept_name or "tumor" in concept_name:
        relationships = [
            {"source": main_concept.ui, "target": "C0027651", "name": "Neoplasm", "rel": "is_a"},
            {"source": main_concept.ui, "target": "C0012634", "name": "Disease", "rel": "is_a"},
            {"source": "C0086116", "target": main_concept.ui, "name": "Malignant Neoplasm", "rel": "parent_of"},
            {"source": main_concept.ui, "target": "C0277786", "name": "Pathologic Function", "rel": "has_mechanism"},
            {"source": main_concept.ui, "target": "C0920425", "name": "Metastasis", "rel": "may_lead_to"},
            {"source": main_concept.ui, "target": "C0006826", "name": "Malignant Cell", "rel": "has_component"}
        ]
        # Add second-level relationships if depth > 1
        if graph_depth > 1:
            relationships.extend([
                {"source": "C0027651", "target": "C1512121", "name": "Benign neoplasm", "rel": "sibling_of"},
                {"source": "C0920425", "target": "C0029064", "name": "Secondary site", "rel": "has_location"}
            ])
    elif "fracture" in concept_name:
        relationships = [
            {"source": main_concept.ui, "target": "C0016658", "name": "Bone", "rel": "has_finding_site"},
            {"source": main_concept.ui, "target": "C0012634", "name": "Disease", "rel": "is_a"},
            {"source": "C0016658", "target": main_concept.ui, "name": "Injury", "rel": "parent_of"},
            {"source": main_concept.ui, "target": "C0037303", "name": "Pain", "rel": "has_symptom"},
            {"source": main_concept.ui, "target": "C0016658", "name": "Bone structure", "rel": "affects"},
            {"source": main_concept.ui, "target": "C0022116", "name": "Joint structure", "rel": "affects"},
            {"source": main_concept.ui, "target": "C1283783", "name": "Trauma", "rel": "result_of"}
        ]
        # Add second-level relationships if depth > 1
        if graph_depth > 1:
            relationships.extend([
                {"source": "C0037303", "target": "C0002052", "name": "Inflammation", "rel": "associated_with"},
                {"source": "C0022116", "target": "C0022408", "name": "Range of motion", "rel": "finding"}
            ])
    else:
        # Generic relationships for other conditions with more nodes
        relationships = [
            {"source": main_concept.ui, "target": "C0012634", "name": "Disease", "rel": "is_a"},
            {"source": main_concept.ui, "target": "C0037303", "name": "Symptom", "rel": "has_manifestation"},
            {"source": "C0277786", "target": main_concept.ui, "name": "Medical Condition", "rel": "parent_of"},
            {"source": main_concept.ui, "target": "C0031843", "name": "Physical Examination", "rel": "diagnosed_by"},
            {"source": main_concept.ui, "target": "C0042210", "name": "Vital Signs", "rel": "has_assessment"},
            {"source": main_concept.ui, "target": "C0220825", "name": "Medical History", "rel": "has_component"},
            {"source": main_concept.ui, "target": "C1285471", "name": "Imaging Study", "rel": "has_procedure"}
        ]
        # Add second-level relationships if depth > 1
        if graph_depth > 1:
            relationships.extend([
                {"source": "C0037303", "target": "C0557060", "name": "Assessment scale", "rel": "measured_by"},
                {"source": "C1285471", "target": "C0011923", "name": "Diagnostic imaging", "rel": "is_a"}
            ])

    # Add third-level depth if requested (just a few examples)
    if graph_depth > 2:
        relationships.extend([
            {"source": "C0011923", "target": "C0040405", "name": "X-Ray", "rel": "includes"},
            {"source": "C0011923", "target": "C0024485", "name": "MRI", "rel": "includes"},
            {"source": "C0011923", "target": "C0040985", "name": "CT Scan", "rel": "includes"},
            {"source": "C0557060", "target": "C0085602", "name": "Pain Scale", "rel": "example_of"}
        ])

    # Add related concept nodes and edges 
    for rel in relationships:
        # Add visual attributes for each node
        if rel["target"] not in G:
            sem_type = "Finding" if "symptom" in rel["rel"] else ("Procedure" if "procedure" in rel["rel"] else "Concept")
            G.add_node(rel["target"], name=rel["name"], type="related", semantic_type=sem_type)
        if rel["source"] not in G:
            if rel["source"] == main_concept.ui:
                continue  # Already added the main concept

            # Add the source node if not already there
            source_name = rel.get("source_name", "")
            sem_type = "Finding" if "symptom" in rel["rel"] else ("Procedure" if "procedure" in rel["rel"] else "Concept")
            G.add_node(rel["source"], name=source_name, type="related", semantic_type=sem_type)

        # Add the edge with relationship attributes
        G.add_edge(rel["source"], rel["target"], rel_type=rel["rel"])

    # Generate network layout based on user selection
    if layout_type == "Spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout_type == "Circular":
        pos = nx.circular_layout(G)
    elif layout_type == "Spiral":
        pos = nx.spiral_layout(G)
    else:
        pos = nx.random_layout(G)

    # Create the Plotly visualization with enhanced styling
    node_colors = []
    node_sizes = []

    for node in G.nodes():
        if G.nodes[node].get('type') == 'main':
            node_colors.append('#FF5252')  # Red for main concept
            node_sizes.append(20)  # Larger size for main concept
        else:
            # Color nodes by semantic type
            semantic_type = G.nodes[node].get('semantic_type', 'Unknown')
            if 'Disease' in semantic_type:
                node_colors.append('#FFA726')  # Orange for diseases
            elif 'Symptom' in semantic_type or 'Finding' in semantic_type:
                node_colors.append('#4CAF50')  # Green for symptoms/findings
            elif 'Procedure' in semantic_type:
                node_colors.append('#42A5F5')  # Blue for procedures
            else:
                node_colors.append('#9C27B0')  # Purple for other concepts
            node_sizes.append(15)  # Standard size for related concepts

    # Create enhanced traces with meaningful colors and mouseover info
    edge_trace, node_trace = _create_network_traces(G, pos, node_colors, node_sizes)

    # Create figure with improved styling
    fig = go.Figure(data=[edge_trace, node_trace],
                  layout=go.Layout(
                      title=dict(
                          text="UMLS Semantic Network",
                          font=dict(size=16, color='white')
                      ),
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=0, l=0, r=0, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      plot_bgcolor='rgba(26, 32, 44, 0.8)',
                      paper_bgcolor='rgba(26, 32, 44, 0.0)',
                      height=600,
                      font=dict(color='white')
                  ))

    # Add a legend annotation (since Plotly network graphs don't support normal legends)
    fig.add_annotation(
        x=0.01, y=0.01,
        xref="paper", yref="paper",
        text="Red: Main Concept | Orange: Diseases | Green: Findings | Blue: Procedures | Purple: Other",
        showarrow=False,
        font=dict(size=10, color="white"),
        align="left",
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="white",
        borderwidth=1,
        borderpad=4
    )

    # Display the graph with custom tooltip
    st.plotly_chart(fig, use_container_width=True)

    # Create an immediate download button for the graph
    try:
        # Create a unique filename with timestamp
        import time
        timestamp = int(time.time())
        filename = f"umls_network_{timestamp}.html"
        
        # Convert the figure to HTML string directly
        html_str = fig.to_html(include_plotlyjs=True, full_html=True)
        
        # Encode to bytes if it's a string - Streamlit download button needs bytes
        if isinstance(html_str, str):
            html_bytes = html_str.encode('utf-8')
        else:
            html_bytes = html_str
            
        # Use Streamlit's download_button directly without the outer button
        st.download_button(
            label="📥 Download Network Graph as HTML",
            data=html_bytes,
            file_name=filename,
            mime="text/html",
            key=f"download_html_{timestamp}"
        )
        st.success("Download button created successfully. Click to download.")
    except Exception as e:
        st.error(f"Error creating download: {e}")
        logger.error(f"Network graph download error: {e}", exc_info=True)

def _create_network_traces(G, pos, node_colors=None, node_sizes=None):
    """
    Helper function to create enhanced network visualization traces for Plotly

    Args:
        G: NetworkX graph object
        pos: Dictionary of node positions
        node_colors: Optional list of colors for each node
        node_sizes: Optional list of sizes for each node

    Returns:
        edge_trace, node_trace: Plotly scatter traces for the network
    """
    # Create edges with custom styling and hover info
    edge_x = []
    edge_y = []
    edge_text = []
    edge_colors = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # Get relationship type and create hover text
        rel_type = edge[2].get('rel_type', '')
        edge_text.append(rel_type)

        # Color edges by relationship type
        if 'is_a' in rel_type or 'parent' in rel_type or 'subtype' in rel_type:
            color = 'rgba(255, 152, 0, 0.7)'  # Orange for hierarchical
        elif 'finding' in rel_type or 'symptom' in rel_type or 'manifestation' in rel_type:
            color = 'rgba(76, 175, 80, 0.7)'  # Green for findings
        elif 'site' in rel_type or 'location' in rel_type:
            color = 'rgba(3, 169, 244, 0.7)'  # Blue for anatomical
        elif 'caused_by' in rel_type or 'causative' in rel_type:
            color = 'rgba(244, 67, 54, 0.7)'  # Red for causal
        else:
            color = 'rgba(158, 158, 158, 0.7)'  # Gray for others

        # Add the color three times (for start point, end point, and None)
        edge_colors.extend([color, color, color])

    # Create a customized edge trace with a single color for all edges
    # (Plotly scatter doesn't support per-segment line colors in a single trace)
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='rgba(158, 158, 158, 0.7)'),
        hoverinfo='text',
        text=edge_text,
        mode='lines',
        name='Relationships'
    )

    # To display multiple edge colors, we would need to create separate traces
    # for each color group, but we'll use a simpler approach for now

    # Create nodes with enhanced hover information
    node_x = []
    node_y = []
    node_text = []
    node_info = []

    # Use default colors if not provided
    if node_colors is None:
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'related')
            if node_type == 'main':
                node_colors.append('#FF5252')  # Red for main concept
            else:
                node_colors.append('#4CAF50')  # Green for related concepts

    # Use default sizes if not provided
    if node_sizes is None:
        node_sizes = []
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'related')
            if node_type == 'main':
                node_sizes.append(20)  # Larger for main concept
            else:
                node_sizes.append(15)  # Standard for related concepts

    # Collect node data
    for i, node in enumerate(G.nodes()):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Get node attributes for rich hover text
        node_attrs = G.nodes[node]
        node_name = node_attrs.get('name', node)
        node_type = node_attrs.get('type', 'related')
        semantic_type = node_attrs.get('semantic_type', 'Concept')

        # Create detailed hover information
        hover_text = f"<b>{node_name}</b><br>CUI: {node}<br>Type: {semantic_type}"
        if node_type == 'main':
            hover_text += "<br><i>(Main Concept)</i>"

        node_text.append(hover_text)

        # Store basic node info for potential additional display
        node_info.append({
            "cui": node,
            "name": node_name,
            "type": semantic_type
        })

    # Create an enhanced node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        textfont=dict(size=10, color="white"),
        marker=dict(
            color=node_colors,
            size=node_sizes,
            symbol='circle',
            line=dict(width=2, color='white'),
            opacity=0.9
        )
    )

    return edge_trace, node_trace