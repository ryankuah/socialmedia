import argparse
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.colors import LinearSegmentedColormap
import community as community_louvain  
import random
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Network Graph Analysis')
    parser.add_argument('graphml_file', help='Path to GraphML file to analyze')
    
    args = parser.parse_args()
    
    base_name = os.path.splitext(os.path.basename(args.graphml_file))[0]
    output_dir = base_name
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load graph
        if not os.path.exists(args.graphml_file):
            raise FileNotFoundError(f"GraphML file not found: {args.graphml_file}")
        
        graph = nx.read_graphml(args.graphml_file)
        
        # Get basic stats
        stats = {
            'file': os.path.basename(args.graphml_file),
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
        
        basic_stats = stats
        
        # Get centrality measures
        centrality_data = {}
        num_nodes = len(graph.nodes())
        
        degree_centrality = nx.degree_centrality(graph)
        centrality_data['degree'] = degree_centrality
        
        in_degree_centrality = nx.in_degree_centrality(graph)
        out_degree_centrality = nx.out_degree_centrality(graph)
        centrality_data['in_degree'] = in_degree_centrality
        centrality_data['out_degree'] = out_degree_centrality
        
        if num_nodes < 1000:
            betweenness_centrality = nx.betweenness_centrality(graph)
            centrality_data['betweenness'] = betweenness_centrality
        else:
            sample_size = min(500, num_nodes // 2)
            try:
                betweenness_centrality = nx.betweenness_centrality(graph, k=sample_size, seed=42)
                centrality_data['betweenness'] = betweenness_centrality
            except Exception as e:
                pass
        
        try:
            if nx.is_connected(graph.to_undirected()):
                closeness_centrality = nx.closeness_centrality(graph)
                centrality_data['closeness'] = closeness_centrality
            else:
                components = list(nx.connected_components(graph.to_undirected()))
                largest_cc = max(components, key=len)
                if len(largest_cc) >= 10:
                    cc_graph = graph.subgraph(largest_cc)
                    cc_closeness = nx.closeness_centrality(cc_graph)
                    closeness_centrality = {node: 0.0 for node in graph.nodes()}
                    closeness_centrality.update(cc_closeness)
                    centrality_data['closeness'] = closeness_centrality
        except Exception as e:
            pass
        
        try:
            if num_nodes < 5000:
                eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=2000, tol=1e-6)
                centrality_data['eigenvector'] = eigenvector_centrality
            else:
                pagerank_centrality = nx.pagerank(graph, max_iter=200, tol=1e-6)
                centrality_data['pagerank'] = pagerank_centrality
        except Exception as e:
            try:
                pagerank_centrality = nx.pagerank(graph, max_iter=200, tol=1e-6)
                centrality_data['pagerank'] = pagerank_centrality
            except Exception as e2:
                pass
        
        # Get top users
        top_users = {}
        
        for measure, values in centrality_data.items():
            if values:
                sorted_users = sorted(values.items(), key=lambda x: x[1], reverse=True)
                top_users[measure] = sorted_users[:10]
        
        # Create visualizations
        plt.style.use('default')
        
        # Sample graph if too large
        viz_graph = graph
        if len(graph.nodes()) > 2000:
            target_size = 1500
            if centrality_data and 'degree' in centrality_data:
                degree_cent = centrality_data['degree']
                top_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)
                n_top = int(target_size * 0.7)
                sampled_nodes = [node for node, _ in top_nodes[:n_top]]
                
                remaining_nodes = [node for node in graph.nodes() if node not in sampled_nodes]
                n_random = target_size - len(sampled_nodes)
                if n_random > 0 and remaining_nodes:
                    random.seed(42)
                    sampled_nodes.extend(random.sample(remaining_nodes, min(n_random, len(remaining_nodes))))
            else:
                degrees = dict(graph.degree())
                sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
                n_top = int(target_size * 0.5)
                sampled_nodes = [node for node, _ in sorted_nodes[:n_top]]
                
                remaining_nodes = [node for node, _ in sorted_nodes[n_top:]]
                n_random = target_size - len(sampled_nodes)
                if n_random > 0 and remaining_nodes:
                    random.seed(42)
                    sampled_nodes.extend(random.sample(remaining_nodes, min(n_random, len(remaining_nodes))))
            
            viz_graph = graph.subgraph(sampled_nodes).copy()
        
        # Create filtered network visualization
        if len(graph.nodes()) > 1000:
            top_n = min(200, len(graph.nodes()) // 5)
        else:
            top_n = 100
        
        if 'degree' in centrality_data and centrality_data['degree']:
            try:
                degree_cent = centrality_data['degree']
                top_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
                top_node_ids = [node for node, _ in top_nodes]
                
                valid_nodes = [node for node in top_node_ids if node in graph.nodes()]
                
                if len(valid_nodes) > 0:
                    subgraph = graph.subgraph(valid_nodes).copy()
                    
                    if len(subgraph.nodes()) > 0:
                        fig, ax = plt.subplots(figsize=(20, 16))
                        
                        if len(subgraph.nodes()) < 50:
                            pos = nx.spring_layout(subgraph, k=3, iterations=100, seed=42)
                        elif len(subgraph.nodes()) < 150:
                            pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
                        else:
                            try:
                                pos = nx.nx_agraph.graphviz_layout(subgraph, prog='neato')
                            except:
                                try:
                                    pos = nx.kamada_kawai_layout(subgraph)
                                except:
                                    pos = nx.spring_layout(subgraph, k=1, iterations=30, seed=42)
                        
                        node_sizes = []
                        for node in subgraph.nodes():
                            size = degree_cent.get(node, 0) * 2000 + 50
                            node_sizes.append(size)
                        
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
                        
                        nodes = nx.draw_networkx_nodes(subgraph, pos, 
                                              node_size=node_sizes,
                                              node_color=node_colors,
                                              cmap=colormap,
                                              alpha=0.7,
                                              linewidths=1,
                                              edgecolors='black',
                                              ax=ax)
                        
                        edge_alpha = 0.6 if len(subgraph.edges()) < 500 else 0.3
                        arrow_size = 20 if len(subgraph.edges()) < 200 else 10
                        
                        nx.draw_networkx_edges(subgraph, pos,
                                              edge_color='gray',
                                              alpha=edge_alpha,
                                              arrows=True,
                                              arrowsize=arrow_size,
                                              arrowstyle='->',
                                              ax=ax)
                        
                        n_labels = min(15, len(valid_nodes) // 10, 25)
                        top_label_nodes = [node for node in valid_nodes[:n_labels] if node in pos]
                        if top_label_nodes:
                            top_pos = {node: pos[node] for node in top_label_nodes}
                            labels = {}
                            for node in top_label_nodes:
                                label = str(node)[:12] + '...' if len(str(node)) > 12 else str(node)
                                labels[node] = label
                            nx.draw_networkx_labels(subgraph, top_pos, labels=labels,
                                                   font_size=8, font_weight='bold', ax=ax)
                        
                        ax.set_title(f'Top {len(valid_nodes)} Most Connected Users\n'
                                  f'Node size = Degree Centrality, Color = In-Degree Centrality\n'
                                  f'Showing {len(subgraph.edges())} connections', 
                                  fontsize=16, fontweight='bold', pad=20)
                        
                        if colormap and isinstance(node_colors, list) and nodes is not None:
                            plt.colorbar(nodes, ax=ax, label='In-Degree Centrality', shrink=0.8)
                        
                        ax.axis('off')
                        plt.tight_layout()
                        
                        filtered_file = os.path.join(output_dir, 'filtered_network.png')
                        plt.savefig(filtered_file, dpi=300, bbox_inches='tight', facecolor='white')
                        plt.close()
                        
            except Exception as e:
                import traceback
                traceback.print_exc()
        
        # Create community visualization
        try:
            undirected_graph = graph.to_undirected()
            
            connected_nodes = [node for node in undirected_graph.nodes() 
                              if undirected_graph.degree(node) > 0]
            subgraph = undirected_graph.subgraph(connected_nodes).copy()
            
            if len(subgraph.nodes()) >= 10:
                if len(subgraph.nodes()) > 1000:
                    degrees = dict(subgraph.degree())
                    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
                    n_sample = min(800, len(subgraph.nodes()))
                    sampled_nodes = [node for node, _ in sorted_nodes[:n_sample]]
                    subgraph = subgraph.subgraph(sampled_nodes).copy()
                
                partition = community_louvain.best_partition(subgraph)
                
                community_sizes = {}
                for node, comm in partition.items():
                    community_sizes[comm] = community_sizes.get(comm, 0) + 1
                
                min_size = max(3, len(subgraph.nodes()) // 100)
                large_communities = {comm: size for comm, size in community_sizes.items() if size >= min_size}
                
                if large_communities:
                    filtered_nodes = [node for node, comm in partition.items() 
                                     if comm in large_communities]
                    viz_graph_comm = subgraph.subgraph(filtered_nodes).copy()
                    
                    plt.figure(figsize=(20, 16))
                    
                    if len(viz_graph_comm.nodes()) < 200:
                        pos = nx.spring_layout(viz_graph_comm, k=2, iterations=50, seed=42)
                    else:
                        try:
                            pos = nx.nx_agraph.graphviz_layout(viz_graph_comm, prog='sfdp')
                        except:
                            pos = nx.spring_layout(viz_graph_comm, k=1, iterations=30, seed=42)
                    
                    communities = list(large_communities.keys())
                    if len(communities) <= 10:
                        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
                    else:
                        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(communities))))
                    
                    for i, comm in enumerate(communities):
                        nodes_in_comm = [node for node in viz_graph_comm.nodes() 
                                       if partition.get(node) == comm]
                        if nodes_in_comm:
                            color_idx = i % len(colors)
                            nx.draw_networkx_nodes(viz_graph_comm, pos,
                                                 nodelist=nodes_in_comm,
                                                 node_color=[colors[color_idx]],
                                                 node_size=30,
                                                 alpha=0.8,
                                                 label=f'Community {comm} ({len(nodes_in_comm)} users)')
                    
                    edge_alpha = 0.4 if len(viz_graph_comm.edges()) < 1000 else 0.2
                    nx.draw_networkx_edges(viz_graph_comm, pos,
                                          edge_color='gray',
                                          alpha=edge_alpha,
                                          arrows=False)
                    
                    plt.title(f'Community Structure\n'
                              f'{len(large_communities)} communities with {min_size}+ members shown\n'
                              f'{len(viz_graph_comm.nodes())} nodes, {len(viz_graph_comm.edges())} edges', 
                              fontsize=16, fontweight='bold')
                    
                    if len(communities) <= 15:
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                    else:
                        sorted_comms = sorted(large_communities.items(), key=lambda x: x[1], reverse=True)[:10]
                        handles = []
                        labels = []
                        for i, (comm, size) in enumerate(sorted_comms):
                            color_idx = communities.index(comm) % len(colors)
                            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor=colors[color_idx], markersize=8))
                            labels.append(f'Community {comm} ({size} users)')
                        plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                        
                    plt.axis('off')
                    plt.tight_layout()
                    
                    community_file = os.path.join(output_dir, 'communities.png')
                    plt.savefig(community_file, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()
                    
        except ImportError:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        
        # Create degree distribution
        plt.figure(figsize=(8, 6))
        degrees = [d for n, d in graph.degree()]
        plt.hist(degrees, bins=min(30, len(set(degrees))), alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Degree Distribution', fontsize=14)
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        degree_file = os.path.join(output_dir, 'degree_distribution.png')
        plt.savefig(degree_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create centrality bar charts
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
                
                top_users_measure = sorted(values.items(), key=lambda x: x[1], reverse=True)[:10]
                
                if top_users_measure:
                    users, scores = zip(*top_users_measure)
                    
                    bars = plt.barh(range(len(users)), scores, 
                                   color=centrality_colors.get(measure, 'lightblue'),
                                   alpha=0.8, edgecolor='black', linewidth=0.5)
                    
                    plt.yticks(range(len(users)), 
                              [str(u)[:20] + '...' if len(str(u)) > 20 else str(u) for u in users])
                    plt.title(f'Top 10 Most Influential Users by {measure.replace("_", " ").title()} Centrality', 
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
                    centrality_file = os.path.join(output_dir, f'top_users_{safe_measure_name}_centrality.png')
                    plt.savefig(centrality_file, dpi=300, bbox_inches='tight')
                    plt.close()
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        in_degrees = [d for n, d in graph.in_degree()]
        out_degrees = [d for n, d in graph.out_degree()]
        plt.scatter(in_degrees, out_degrees, alpha=0.6, color='coral')
        plt.xlabel('In-degree')
        plt.ylabel('Out-degree')
        plt.title('In-degree vs Out-degree', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        scatter_file = os.path.join(output_dir, 'in_out_degree_scatter.png')
        plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save analysis results
        results = {
            'source_file': args.graphml_file,
            'basic_statistics': basic_stats,
            'top_users_by_centrality': top_users
        }
        
        results_file = os.path.join(output_dir, 'graph_analysis.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        centrality_file = os.path.join(output_dir, 'centrality_data.json')
        with open(centrality_file, 'w', encoding='utf-8') as f:
            json.dump(centrality_data, f, indent=2, ensure_ascii=False)
        
    except FileNotFoundError as e:
        sys.exit(1)
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()