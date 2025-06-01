import argparse
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.colors import LinearSegmentedColormap
import community as community_louvain  


def load_graph(graphml_file):
    if not os.path.exists(graphml_file):
        raise FileNotFoundError(f"GraphML file not found: {graphml_file}")
    
    graph = nx.read_graphml(graphml_file)
    return graph


def get_basic_stats(graph, graphml_file):
    stats = {
        'file': os.path.basename(graphml_file),
        'nodes': len(graph.nodes()),
        'edges': len(graph.edges()),
        'density': nx.density(graph),
        'is_connected': nx.is_connected(graph.to_undirected()),
        'num_connected_components': nx.number_connected_components(graph.to_undirected())
    }
    
    degrees = dict(graph.degree())
    if degrees:
        stats['avg_degree'] = np.mean(list(degrees.values()))
        stats['max_degree'] = max(degrees.values())
        stats['min_degree'] = min(degrees.values())
    
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    
    if in_degrees:
        stats['avg_in_degree'] = np.mean(list(in_degrees.values()))
        stats['max_in_degree'] = max(in_degrees.values())
        
    if out_degrees:
        stats['avg_out_degree'] = np.mean(list(out_degrees.values()))
        stats['max_out_degree'] = max(out_degrees.values())
    
    return stats

def get_centrality_measures(graph):
    centrality_data = {}
    
    degree_centrality = nx.degree_centrality(graph)
    centrality_data['degree'] = degree_centrality
    
    in_degree_centrality = nx.in_degree_centrality(graph)
    out_degree_centrality = nx.out_degree_centrality(graph)
    centrality_data['in_degree'] = in_degree_centrality
    centrality_data['out_degree'] = out_degree_centrality
    
    if len(graph.nodes()) < 500:
        betweenness_centrality = nx.betweenness_centrality(graph)
        centrality_data['betweenness'] = betweenness_centrality
    else:
        print("Skipping betweenness centrality (graph too large)")
    
    try:
        if nx.is_connected(graph.to_undirected()):
            closeness_centrality = nx.closeness_centrality(graph)
            centrality_data['closeness'] = closeness_centrality
        else:
            print("Skipping closeness centrality (graph not connected)")
    except:
        print("Skipping closeness centrality (graph not suitable)")
    
    try:
        eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)
        centrality_data['eigenvector'] = eigenvector_centrality
    except:
        print("Skipping eigenvector centrality (convergence issues)")
    
    return centrality_data


def get_top_users(centrality_data, top_n=10):
    top_users = {}
    
    for measure, values in centrality_data.items():
        if values:
            sorted_users = sorted(values.items(), key=lambda x: x[1], reverse=True)
            top_users[measure] = sorted_users[:top_n]
    
    return top_users


def create_visualizations(graph, centrality_data, output_dir, base_name):
    """Create multiple improved network visualizations"""
    plt.style.use('default')
    
    # 1. Overview: Filtered network showing only top nodes
    print("Creating filtered network overview...")
    create_filtered_network(graph, centrality_data, output_dir, base_name)
    
    # 2. Community detection visualization
    print("Creating community structure visualization...")
    create_community_visualization(graph, output_dir, base_name)
    
    # 3. Centrality-based node sizing
    print("Creating centrality-based visualizations...")
    create_centrality_visualizations(graph, centrality_data, output_dir, base_name)
    
    # 4. Largest connected component detailed view
    print("Creating largest component visualization...")
    create_largest_component_viz(graph, centrality_data, output_dir, base_name)
    
    # 5. Keep the original degree distribution and other plots
    create_degree_visualizations(graph, output_dir, base_name)
    create_centrality_bar_charts(centrality_data, output_dir, base_name)
    create_scatter_plots(graph, output_dir, base_name)


def create_filtered_network(graph, centrality_data, output_dir, base_name, top_n=50):
    """Create a filtered network showing only the most important nodes"""
    if 'degree' not in centrality_data or not centrality_data['degree']:
        print("Skipping filtered network: degree centrality not available")
        return
    
    try:
        # Get top nodes by degree centrality
        degree_cent = centrality_data['degree']
        top_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_node_ids = [node for node, _ in top_nodes]
        
        # Ensure all nodes exist in the graph
        valid_nodes = [node for node in top_node_ids if node in graph.nodes()]
        
        if len(valid_nodes) == 0:
            print("No valid nodes found for filtered network")
            return
        
        # Create subgraph with top nodes and their connections
        subgraph = graph.subgraph(valid_nodes).copy()
        
        if len(subgraph.nodes()) == 0:
            print("No nodes in filtered subgraph")
            return
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Use different layout based on size - with error handling
        try:
            if len(subgraph.nodes()) < 30:
                pos = nx.spring_layout(subgraph, k=3, iterations=100, seed=42)
            else:
                # Try kamada_kawai layout, fall back to spring layout if it fails
                try:
                    pos = nx.kamada_kawai_layout(subgraph)
                except:
                    print("Kamada-Kawai layout failed, using spring layout")
                    pos = nx.spring_layout(subgraph, k=1, iterations=50, seed=42)
        except Exception as e:
            print(f"Layout calculation failed: {e}, using random layout")
            pos = nx.random_layout(subgraph, seed=42)
        
        # Node sizes based on degree centrality
        node_sizes = []
        for node in subgraph.nodes():
            size = degree_cent.get(node, 0) * 3000 + 100
            node_sizes.append(size)
        
        # Node colors based on in-degree centrality if available
        if 'in_degree' in centrality_data and centrality_data['in_degree']:
            in_degree_cent = centrality_data['in_degree']
            node_colors = []
            for node in subgraph.nodes():
                color = in_degree_cent.get(node, 0)
                node_colors.append(color)
            colormap = 'Reds'
        else:
            node_colors = 'lightblue'
            colormap = None
        
        # Draw the network
        nodes = nx.draw_networkx_nodes(subgraph, pos, 
                              node_size=node_sizes,
                              node_color=node_colors,
                              cmap=colormap,
                              alpha=0.7,
                              linewidths=1,
                              edgecolors='black',
                              ax=ax)
        
        nx.draw_networkx_edges(subgraph, pos,
                              edge_color='gray',
                              alpha=0.5,
                              arrows=True,
                              arrowsize=15,
                              arrowstyle='->',
                              ax=ax)
        
        # Add labels for top 10 nodes
        top_10_nodes = [node for node in valid_nodes[:10] if node in pos]
        if top_10_nodes:
            top_10_pos = {node: pos[node] for node in top_10_nodes}
            # Truncate long node names for labels
            labels = {}
            for node in top_10_nodes:
                label = str(node)[:15] + '...' if len(str(node)) > 15 else str(node)
                labels[node] = label
            nx.draw_networkx_labels(subgraph, top_10_pos, labels=labels,
                                   font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title(f'Top {top_n} Most Connected Users - {base_name}\n'
                  f'Node size = Degree Centrality, Color = In-Degree Centrality', 
                  fontsize=16, fontweight='bold', pad=20)
        
        if colormap and isinstance(node_colors, list) and nodes is not None:
            plt.colorbar(nodes, ax=ax, label='In-Degree Centrality', shrink=0.8)
        
        ax.axis('off')
        plt.tight_layout()
        
        filtered_file = os.path.join(output_dir, f'{base_name}_filtered_network.png')
        plt.savefig(filtered_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Filtered network saved as {filtered_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error in create_filtered_network: {e}")
        import traceback
        traceback.print_exc()


def create_community_visualization(graph, output_dir, base_name):
    """Create community detection visualization"""
    try:
        # Convert to undirected for community detection
        undirected_graph = graph.to_undirected()
        
        # Remove isolated nodes for better community detection
        connected_nodes = [node for node in undirected_graph.nodes() 
                          if undirected_graph.degree(node) > 0]
        subgraph = undirected_graph.subgraph(connected_nodes).copy()
        
        if len(subgraph.nodes()) < 10:
            print("Skipping community visualization: not enough connected nodes")
            return
        
        # Detect communities using Louvain algorithm
        partition = community_louvain.best_partition(subgraph)
        
        # Filter to largest communities for visualization
        community_sizes = {}
        for node, comm in partition.items():
            community_sizes[comm] = community_sizes.get(comm, 0) + 1
        
        # Keep only communities with at least 5 members
        large_communities = {comm: size for comm, size in community_sizes.items() if size >= 5}
        
        if not large_communities:
            print("No large communities found")
            return
        
        # Filter nodes to only those in large communities
        filtered_nodes = [node for node, comm in partition.items() 
                         if comm in large_communities]
        viz_graph = subgraph.subgraph(filtered_nodes).copy()
        
        plt.figure(figsize=(16, 12))
        
        # Use force-directed layout
        pos = nx.spring_layout(viz_graph, k=1, iterations=50, seed=42)
        
        # Create color map for communities
        communities = list(large_communities.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        
        # Draw each community with different colors
        for i, comm in enumerate(communities):
            nodes_in_comm = [node for node in viz_graph.nodes() 
                           if partition.get(node) == comm]
            if nodes_in_comm:
                nx.draw_networkx_nodes(viz_graph, pos,
                                     nodelist=nodes_in_comm,
                                     node_color=[colors[i]],
                                     node_size=50,
                                     alpha=0.8,
                                     label=f'Community {comm} ({len(nodes_in_comm)} users)')
        
        # Draw edges
        nx.draw_networkx_edges(viz_graph, pos,
                              edge_color='gray',
                              alpha=0.3,
                              arrows=False)
        
        plt.title(f'Community Structure - {base_name}\n'
                  f'{len(large_communities)} communities with 5+ members shown', 
                  fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axis('off')
        plt.tight_layout()
        
        community_file = os.path.join(output_dir, f'{base_name}_communities.png')
        plt.savefig(community_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Community visualization saved as {community_file}")
        plt.close()
        
    except ImportError:
        print("Skipping community detection: python-louvain not installed")
        print("Install with: pip install python-louvain")
    except Exception as e:
        print(f"Error in community detection: {e}")


def create_centrality_visualizations(graph, centrality_data, output_dir, base_name):
    """Create visualizations with nodes sized by different centrality measures"""
    if not centrality_data:
        return
    
    try:
        # Select top 100 nodes for visualization
        if 'degree' in centrality_data:
            degree_cent = centrality_data['degree']
            top_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:100]
            top_node_ids = [node for node, _ in top_nodes]
            
            # Ensure all nodes exist in the graph
            valid_nodes = [node for node in top_node_ids if node in graph.nodes()]
            
            if len(valid_nodes) == 0:
                print("No valid nodes for centrality visualizations")
                return
                
            subgraph = graph.subgraph(valid_nodes).copy()
            
            if len(subgraph.nodes()) == 0:
                return
            
            # Create layout once
            try:
                pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
            except:
                pos = nx.random_layout(subgraph, seed=42)
            
            # Create visualization for each centrality measure
            centrality_measures = ['degree', 'in_degree', 'out_degree', 'eigenvector']
            
            for measure in centrality_measures:
                if measure not in centrality_data or not centrality_data[measure]:
                    continue
                    
                try:
                    fig, ax = plt.subplots(figsize=(14, 10))
                    
                    cent_data = centrality_data[measure]
                    
                    # Node sizes and colors based on centrality
                    node_sizes = []
                    node_colors = []
                    for node in subgraph.nodes():
                        if node in cent_data:
                            size = cent_data[node] * 2000 + 50
                            color = cent_data[node]
                        else:
                            size = 50
                            color = 0
                        node_sizes.append(size)
                        node_colors.append(color)
                    
                    # Draw network
                    nodes = nx.draw_networkx_nodes(subgraph, pos,
                                                 node_size=node_sizes,
                                                 node_color=node_colors,
                                                 cmap='plasma',
                                                 alpha=0.8,
                                                 linewidths=1,
                                                 edgecolors='black',
                                                 ax=ax)
                    
                    nx.draw_networkx_edges(subgraph, pos,
                                          edge_color='gray',
                                          alpha=0.4,
                                          arrows=True,
                                          arrowsize=10,
                                          ax=ax)
                    
                    # Add labels for top 5 nodes
                    top_5_nodes = []
                    for node, _ in sorted(cent_data.items(), key=lambda x: x[1], reverse=True)[:5]:
                        if node in subgraph.nodes():
                            top_5_nodes.append(node)
                    
                    if top_5_nodes:
                        top_5_pos = {node: pos[node] for node in top_5_nodes if node in pos}
                        labels = {}
                        for node in top_5_nodes:
                            label = str(node)[:12] + '...' if len(str(node)) > 12 else str(node)
                            labels[node] = label
                        nx.draw_networkx_labels(subgraph, top_5_pos, labels=labels,
                                               font_size=8, font_weight='bold', ax=ax)
                    
                    ax.set_title(f'Network by {measure.replace("_", " ").title()} Centrality - {base_name}\n'
                              f'Top 100 users, node size and color indicate centrality', 
                              fontsize=14, fontweight='bold')
                    
                    # Add colorbar
                    if nodes is not None:
                        plt.colorbar(nodes, ax=ax, label=f'{measure.replace("_", " ").title()} Centrality')
                    ax.axis('off')
                    plt.tight_layout()
                    
                    cent_file = os.path.join(output_dir, f'{base_name}_network_{measure}_centrality.png')
                    plt.savefig(cent_file, dpi=300, bbox_inches='tight', facecolor='white')
                    print(f"{measure.title()} centrality network saved as {cent_file}")
                    plt.close()
                    
                except Exception as e:
                    print(f"Error creating {measure} centrality visualization: {e}")
                    
    except Exception as e:
        print(f"Error in create_centrality_visualizations: {e}")


def create_largest_component_viz(graph, centrality_data, output_dir, base_name):
    """Create detailed visualization of the largest connected component"""
    try:
        # Get largest connected component
        if graph.is_directed():
            undirected = graph.to_undirected()
            components = nx.connected_components(undirected)
        else:
            components = nx.connected_components(graph)
        
        largest_cc = max(components, key=len)
        
        if len(largest_cc) < 10:
            print("Largest component too small for visualization")
            return
        
        # Create subgraph of largest component
        cc_graph = graph.subgraph(largest_cc).copy()
        
        # Limit to top nodes if too large
        if len(cc_graph.nodes()) > 150:
            if 'degree' in centrality_data:
                degree_cent = centrality_data['degree']
                cc_nodes_ranked = []
                for node in cc_graph.nodes():
                    if node in degree_cent:
                        cc_nodes_ranked.append((node, degree_cent[node]))
                    else:
                        cc_nodes_ranked.append((node, 0))
                cc_nodes_ranked.sort(key=lambda x: x[1], reverse=True)
                top_cc_nodes = [node for node, _ in cc_nodes_ranked[:150]]
                cc_graph = cc_graph.subgraph(top_cc_nodes).copy()
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Use hierarchical layout for large components
        try:
            if len(cc_graph.nodes()) > 50:
                pos = nx.spring_layout(cc_graph, k=1.5, iterations=30, seed=42)
            else:
                pos = nx.spring_layout(cc_graph, k=3, iterations=100, seed=42)
        except:
            pos = nx.random_layout(cc_graph, seed=42)
        
        # Node sizes based on degree
        degrees = dict(cc_graph.degree())
        node_sizes = []
        for node in cc_graph.nodes():
            size = degrees.get(node, 1) * 20 + 30
            node_sizes.append(size)
        
        # Node colors based on eigenvector centrality if available
        if 'eigenvector' in centrality_data and centrality_data['eigenvector']:
            eig_cent = centrality_data['eigenvector']
            node_colors = []
            for node in cc_graph.nodes():
                color = eig_cent.get(node, 0)
                node_colors.append(color)
            colormap = 'viridis'
        else:
            node_colors = 'lightblue'
            colormap = None
        
        # Draw network
        nodes = nx.draw_networkx_nodes(cc_graph, pos,
                                      node_size=node_sizes,
                                      node_color=node_colors,
                                      cmap=colormap,
                                      alpha=0.7,
                                      linewidths=0.5,
                                      edgecolors='black',
                                      ax=ax)
        
        nx.draw_networkx_edges(cc_graph, pos,
                              edge_color='gray',
                              alpha=0.4,
                              arrows=True,
                              arrowsize=8,
                              ax=ax)
        
        ax.set_title(f'Largest Connected Component - {base_name}\n'
                  f'{len(cc_graph.nodes())} users, {len(cc_graph.edges())} connections',
                  fontsize=16, fontweight='bold')
        
        if colormap and nodes is not None:
            plt.colorbar(nodes, ax=ax, label='Eigenvector Centrality')
        
        ax.axis('off')
        plt.tight_layout()
        
        cc_file = os.path.join(output_dir, f'{base_name}_largest_component.png')
        plt.savefig(cc_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Largest component visualization saved as {cc_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error in create_largest_component_viz: {e}")
        import traceback
        traceback.print_exc()


def create_degree_visualizations(graph, output_dir, base_name):
    """Create degree distribution and related visualizations"""
    plt.figure(figsize=(8, 6))
    degrees = [d for n, d in graph.degree()]
    plt.hist(degrees, bins=min(30, len(set(degrees))), alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Degree Distribution: {base_name}', fontsize=14)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    degree_file = os.path.join(output_dir, f'{base_name}_degree_distribution.png')
    plt.savefig(degree_file, dpi=300, bbox_inches='tight')
    print(f"Degree distribution saved as {degree_file}")
    plt.close()


def create_centrality_bar_charts(centrality_data, output_dir, base_name):
    """Create bar charts for centrality measures"""
    centrality_colors = {
        'degree': 'skyblue',
        'in_degree': 'lightgreen',
        'out_degree': 'lightcoral',
        'betweenness': 'gold',
        'closeness': 'plum',
        'eigenvector': 'orange'
    }
    
    for measure, values in centrality_data.items():
        if values:
            plt.figure(figsize=(12, 8))
            
            top_users = sorted(values.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if top_users:
                users, scores = zip(*top_users)
                
                bars = plt.barh(range(len(users)), scores, 
                               color=centrality_colors.get(measure, 'lightblue'),
                               alpha=0.8, edgecolor='black', linewidth=0.5)
                
                plt.yticks(range(len(users)), 
                          [str(u)[:20] + '...' if len(str(u)) > 20 else str(u) for u in users])
                plt.title(f'Top 10 Most Influential Users by {measure.replace("_", " ").title()} Centrality: {base_name}', 
                         fontsize=14, fontweight='bold')
                plt.xlabel(f'{measure.replace("_", " ").title()} Centrality Score', fontsize=12)
                plt.ylabel('Users', fontsize=12)
                
                for i, (bar, score) in enumerate(zip(bars, scores)):
                    plt.text(score + max(scores) * 0.01, i, f'{score:.4f}', 
                            va='center', ha='left', fontsize=10)
                
                plt.gca().invert_yaxis()
                
                plt.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                
                safe_measure_name = measure.replace('_', '_')
                centrality_file = os.path.join(output_dir, f'{base_name}_top_users_{safe_measure_name}_centrality.png')
                plt.savefig(centrality_file, dpi=300, bbox_inches='tight')
                print(f"Top users by {measure} centrality saved as {centrality_file}")
                plt.close()


def create_scatter_plots(graph, output_dir, base_name):
    """Create scatter plots for degree analysis"""
    plt.figure(figsize=(8, 6))
    in_degrees = [d for n, d in graph.in_degree()]
    out_degrees = [d for n, d in graph.out_degree()]
    plt.scatter(in_degrees, out_degrees, alpha=0.6, color='coral')
    plt.xlabel('In-degree')
    plt.ylabel('Out-degree')
    plt.title(f'In-degree vs Out-degree: {base_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    scatter_file = os.path.join(output_dir, f'{base_name}_in_out_degree_scatter.png')
    plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
    print(f"In-degree vs Out-degree scatter saved as {scatter_file}")
    plt.close()


def save_analysis_results(graphml_file, basic_stats, centrality_data, top_users, output_dir):
    base_name = os.path.splitext(os.path.basename(graphml_file))[0]
    
    results = {
        'source_file': graphml_file,
        'basic_statistics': basic_stats,
        'top_users_by_centrality': top_users
    }
    
    results_file = os.path.join(output_dir, f'{base_name}_graph_analysis.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Analysis results saved to {results_file}")
    
    centrality_file = os.path.join(output_dir, f'{base_name}_centrality_data.json')
    with open(centrality_file, 'w', encoding='utf-8') as f:
        json.dump(centrality_data, f, indent=2, ensure_ascii=False)
    print(f"Centrality data saved to {centrality_file}")
    
    return results_file, centrality_file


def main():
    parser = argparse.ArgumentParser(description='Network Graph Analysis')
    parser.add_argument('graphml_file', help='Path to GraphML file to analyze')
    parser.add_argument('--no-plots', action='store_true', help='Skip creating visualization plots')
    parser.add_argument('--output-dir', default='.', help='Output directory for results (default: current directory)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        graph = load_graph(args.graphml_file)
        
        basic_stats = get_basic_stats(graph, args.graphml_file)
        
        centrality_data = get_centrality_measures(graph)
        
        top_users = get_top_users(centrality_data)
        
        if not args.no_plots:
            base_name = os.path.splitext(os.path.basename(args.graphml_file))[0]
            create_visualizations(graph, centrality_data, args.output_dir, base_name)
        
        # Save results
        results_file, centrality_file = save_analysis_results(args.graphml_file, basic_stats, centrality_data, top_users, args.output_dir)
        
        print(f"Results saved to: {results_file}")
        print(f"Centrality data: {centrality_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()