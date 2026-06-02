import json
import networkx as nx
from pyvis.network import Network
import os

def visualize_knowledge_graph(json_file="graph_data.json"):
    # 1. Load the Data
    if not os.path.exists(json_file):
        print(f"❌ Error: {json_file} not found. Run graph_extractor.py first!")
        return

    with open(json_file, "r") as f:
        triples = json.load(f)

    print(f"🔹 Loading {len(triples)} connections into the graph...")

    # 2. Create NetworkX Graph
    G = nx.DiGraph() # "DiGraph" means Directed Graph (arrows have direction)

    for item in triples:
        head = item["head"]
        tail = item["tail"]
        relation = item["relation"]

        # Add nodes (concepts)
        G.add_node(head, color="#97C2FC", title=head)
        G.add_node(tail, color="#FFFF00", title=tail) # Yellow for tail
        
        # Add edge (connection)
        G.add_edge(head, tail, title=relation, label=relation)

    # 3. Visualize with PyVis
    print("🎨 Generating interactive graph...")
    
    # Create the network visualizer
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
    
    # Import our NetworkX graph
    net.from_nx(G)
    
    # Add physics controls (makes it bounce and organize itself)
    net.show_buttons(filter_=['physics'])
    
    # Save output
    output_file = "knowledge_graph.html"
    net.write_html(output_file)
    
    print(f"✅ Graph saved to: {output_file}")
    print(f"👉 Open this file in your browser to see it!")

if __name__ == "__main__":
    visualize_knowledge_graph()
