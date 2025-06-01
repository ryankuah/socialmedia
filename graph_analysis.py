import argparse
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


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
    plt.style.use('default')
    
    plt.figure(figsize=(10, 8))
    
    if len(graph.nodes()) < 100:
        pos = nx.spring_layout(graph, k=1, iterations=50)
    else:
        pos = nx.spring_layout(graph, k=0.5, iterations=20)
    
    nx.draw(graph, pos, node_size=20, node_color='lightblue',
            edge_color='gray', alpha=0.6, arrows=True, arrowsize=10)
    plt.title(f'Network Graph: {base_name}', fontsize=14)
    plt.axis('off')
    
    network_file = os.path.join(output_dir, f'{base_name}_network_graph.png')
    plt.savefig(network_file, dpi=300, bbox_inches='tight')
    print(f"Network graph saved as {network_file}")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    degrees = [d for n, d in graph.degree()]
    plt.hist(degrees, bins=min(30, len(set(degrees))), alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Degree Distribution: {base_name}', fontsize=14)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    
    degree_file = os.path.join(output_dir, f'{base_name}_degree_distribution.png')
    plt.savefig(degree_file, dpi=300, bbox_inches='tight')
    print(f"Degree distribution saved as {degree_file}")
    plt.close()
    
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
    
    plt.figure(figsize=(8, 6))
    in_degrees = [d for n, d in graph.in_degree()]
    out_degrees = [d for n, d in graph.out_degree()]
    plt.scatter(in_degrees, out_degrees, alpha=0.6, color='coral')
    plt.xlabel('In-degree')
    plt.ylabel('Out-degree')
    plt.title(f'In-degree vs Out-degree: {base_name}', fontsize=14)
    
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