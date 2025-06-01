import subprocess
import json
import networkx as nx
import datetime
import time
import os
from collections import defaultdict
import re


def run_truthbrush_command(command_args):
    try:
        result = subprocess.run(['truthbrush'] + command_args, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            if any(indicator in result.stderr.lower() for indicator in ['rate limit', 'too many requests', 'limit exceeded', '429', 'quota']):
                return "RATE_LIMITED"
            return None
        
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            fixed_stdout = result.stdout.replace("'", '"').replace(': None,', ': null,').replace(': None}', ': null,').replace(': True,', ': true,').replace(': True}', ': true}').replace(': False,', ': false,').replace(': False}', ': false}')
            
            try:
                return json.loads(fixed_stdout)
            except json.JSONDecodeError:
                import ast
                try:
                    if result.stdout.strip().startswith(('{', '[')):
                        return ast.literal_eval(result.stdout)
                except:
                    pass
                return None
            
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def main():
    sFileName = "truthsocial.graphml"
    user_posts_file = "truthsocial.json"
    checkpoint_file = "checkpoint.json"
    
    try:
        if subprocess.run(['truthbrush', '--help'], capture_output=True, timeout=10).returncode != 0:
            print("ERROR: truthbrush command failed. Please ensure truthbrush is installed and working.")
            return
    except FileNotFoundError:
        print("ERROR: truthbrush command not found. Please install truthbrush first.")
        print("You can install it with: npm install -g truthbrush")
        return
    except Exception as e:
        print(f"ERROR: Failed to check truthbrush availability: {e}")
        return
    
    graph = nx.DiGraph()
    user_data = defaultdict(lambda: {'posts': [], 'total_posts': 0, 'total_interactions': 0, 'replies_made': 0, 'replies_received': 0})
    all_posts_with_replies = []
    processed_post_ids = set()
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            all_posts_with_replies = checkpoint_data.get('collected_posts', [])
            processed_post_ids = set(checkpoint_data.get('processed_post_ids', []))
        except:
            pass
    
    posts = run_truthbrush_command(['search', '--searchtype', 'statuses', 'ukraine'])
    
    if posts == "RATE_LIMITED":
        print("RATE LIMITED: Initial search was rate limited. Exiting.")
        return
    
    print(f"DEBUG: Initial search returned {len(posts) if posts else 0} posts")
    
    if posts and isinstance(posts, list):
        new_posts = [post for post in posts[:1000] if post.get('id') and post.get('id') not in processed_post_ids]
        print(f"DEBUG: After filtering, {len(new_posts)} posts to process")
        
        for i, post in enumerate(new_posts):
            post_id = post.get('id')
            all_posts_with_replies.append(post)
            print(f"DEBUG: Processing post {i+1}/{len(new_posts)} (ID: {post_id})")
            
            comments_data = run_truthbrush_command(['comments', str(post_id), '--includeall', '100'])
            
            if comments_data == "RATE_LIMITED":
                print(f"RATE LIMITED: Hit rate limit while getting comments for post {i+1}/{len(new_posts)}. Saving checkpoint and exiting.")
                checkpoint_data = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'processed_post_ids': list(processed_post_ids),
                    'collected_posts': all_posts_with_replies,
                    'search_term': 'ukraine'
                }
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                print(f"Checkpoint saved. Run script again to resume from post {i+2}.")
                return
            
            if comments_data and isinstance(comments_data, list):
                print(f"DEBUG: Found {len(comments_data)} comments for post {post_id}")
                all_posts_with_replies.extend(comments_data)
            else:
                print(f"DEBUG: No comments found for post {post_id}")
            
            processed_post_ids.add(post_id)
            time.sleep(2)
            
            if (i + 1) % 10 == 0:
                checkpoint_data = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'processed_post_ids': list(processed_post_ids),
                    'collected_posts': all_posts_with_replies,
                    'search_term': 'ukraine'
                }
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                print(f"DEBUG: Checkpoint saved after processing {i+1} posts")
    else:
        print("DEBUG: No posts found or posts is not a list")
    
    for post in all_posts_with_replies:
        author = (post.get('account', {}).get('username') or 
                  post.get('username') or 
                  f"user_{post.get('account_id', 'unknown')}")
        
        replied_to_user = f"user_{post['in_reply_to_account_id']}" if post.get('in_reply_to_account_id') else None
        
        content = post.get('content', '') or post.get('text', '')
        mentions = re.findall(r'@([a-zA-Z0-9_]+)', content) if content else []
        
        if not graph.has_node(author):
            graph.add_node(author, post_count=0, replies_made=0, replies_received=0)
        
        graph.nodes[author]['post_count'] += 1
        
        interactions = post.get('favourites_count', 0) + post.get('reblogs_count', 0) + post.get('replies_count', 0)
        
        user_data[author]['posts'].append({
            'content': post.get('content', ''),
            'created_at': post.get('created_at', ''),
            'likes_count': post.get('favourites_count', 0),
            'reposts_count': post.get('reblogs_count', 0),
            'replies_count': post.get('replies_count', 0),
            'is_reply': replied_to_user is not None,
            'replied_to': replied_to_user,
            'post_id': post.get('id')
        })
        user_data[author]['total_posts'] += 1
        user_data[author]['total_interactions'] += interactions
        
        if replied_to_user and replied_to_user != author:
            if not graph.has_node(replied_to_user):
                graph.add_node(replied_to_user, post_count=0, replies_made=0, replies_received=0)
            
            if graph.has_edge(author, replied_to_user):
                graph[author][replied_to_user]['reply_count'] += 1
            else:
                graph.add_edge(author, replied_to_user, reply_count=1, interaction_type='reply')
            
            graph.nodes[author]['replies_made'] += 1
            graph.nodes[replied_to_user]['replies_received'] += 1
            user_data[author]['replies_made'] += 1
            user_data[replied_to_user]['replies_received'] += 1
        
        for mention in mentions:
            if mention not in (author, replied_to_user):
                if not graph.has_node(mention):
                    graph.add_node(mention, post_count=0, replies_made=0, replies_received=0)
                
                if graph.has_edge(author, mention):
                    graph[author][mention]['mention_count'] = graph[author][mention].get('mention_count', 0) + 1
                else:
                    graph.add_edge(author, mention, mention_count=1, interaction_type='mention')
    
    nx.write_graphml(graph, sFileName)
    
    user_data_dict = dict(user_data)
    
    reply_edges = sum(1 for _, _, data in graph.edges(data=True) if data.get('interaction_type') == 'reply')
    mention_edges = sum(1 for _, _, data in graph.edges(data=True) if data.get('interaction_type') == 'mention')
    
    summary_stats = {
        'search_term': 'ukraine',
        'total_users': len(user_data_dict),
        'total_posts': sum(user['total_posts'] for user in user_data_dict.values()),
        'total_interactions': sum(user['total_interactions'] for user in user_data_dict.values()),
        'total_replies_made': sum(user['replies_made'] for user in user_data_dict.values()),
        'total_replies_received': sum(user['replies_received'] for user in user_data_dict.values()),
        'network_nodes': len(graph.nodes()),
        'network_edges': len(graph.edges()),
        'reply_edges': reply_edges,
        'mention_edges': mention_edges,
        'collection_date': datetime.datetime.now().isoformat(),
        'data_collection_method': 'focused_conversation_threads'
    }
    
    final_data = {
        'summary': summary_stats,
        'users': user_data_dict
    }
    
    with open(user_posts_file, "w", encoding="utf-8") as json_file:
        json.dump(final_data, json_file, indent=2, ensure_ascii=False)
    
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    # Debugging information
    print(f"\n=== DATA COLLECTION SUMMARY ===")
    print(f"Total posts collected: {len(all_posts_with_replies)}")
    print(f"Unique users found: {len(user_data_dict)}")
    print(f"Network nodes: {len(graph.nodes())}")
    print(f"Network edges: {len(graph.edges())}")
    print(f"Reply relationships: {reply_edges}")
    print(f"Mention relationships: {mention_edges}")
    print(f"Files saved: {sFileName}, {user_posts_file}")
    print(f"================================\n")

if __name__ == "__main__":
    main()