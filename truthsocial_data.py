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


def load_existing_data(filename):
    """Load existing data from truthsocial.json if it exists"""
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract all posts from the existing data structure
            all_posts = []
            processed_ids = set()
            
            if 'users' in data:
                for username, user_data in data['users'].items():
                    for post in user_data.get('posts', []):
                        if post.get('post_id'):
                            processed_ids.add(post['post_id'])
                            # Convert back to API format for consistency - PRESERVE ORIGINAL AUTHOR
                            api_post = {
                                'id': post['post_id'],
                                'content': post.get('content', ''),
                                'created_at': post.get('created_at', ''),
                                'favourites_count': post.get('likes_count', 0),
                                'reblogs_count': post.get('reposts_count', 0),
                                'replies_count': post.get('replies_count', 0),
                                'in_reply_to_account_id': post.get('replied_to', '').replace('user_', '') if post.get('replied_to') and post.get('replied_to').startswith('user_') else None,
                                'account': {'username': username}  # FIXED: Use actual username instead of 'placeholder'
                            }
                            all_posts.append(api_post)
            
            return all_posts, processed_ids, data.get('summary', {})
        except Exception as e:
            print(f"Warning: Could not load existing data from {filename}: {e}")
    
    return [], set(), {}


def get_oldest_post_id(posts):
    """Get the oldest post ID for backward pagination"""
    if not posts:
        return None
    
    # Sort by ID (lower IDs are older) and return the smallest
    post_ids = [post.get('id') for post in posts if post.get('id')]
    return min(post_ids) if post_ids else None


def search_with_pagination(search_term, max_id=None, limit_pages=5):
    """Search for posts with backward pagination"""
    all_posts = []
    current_max_id = max_id
    
    for page in range(limit_pages):
        print(f"DEBUG: Searching '{search_term}' - Page {page + 1}/{limit_pages}" + (f" (max_id: {current_max_id})" if current_max_id else ""))
        
        # Build search command
        search_args = ['search', '--searchtype', 'statuses', search_term]
        if current_max_id:
            search_args.extend(['--max-id', str(current_max_id)])
        
        posts = run_truthbrush_command(search_args)
        
        if posts == "RATE_LIMITED":
            print(f"RATE LIMITED: Hit rate limit on page {page + 1} for '{search_term}'")
            break
        
        if not posts or not isinstance(posts, list) or len(posts) == 0:
            print(f"DEBUG: No more posts found for '{search_term}' on page {page + 1}")
            break
        
        print(f"DEBUG: Found {len(posts)} posts on page {page + 1}")
        all_posts.extend(posts)
        
        # Update max_id for next page (use the oldest post ID from current batch)
        post_ids = [post.get('id') for post in posts if post.get('id')]
        if post_ids:
            current_max_id = min(post_ids)
        else:
            break
        
        # Rate limiting between requests
        time.sleep(3)
    
    return all_posts


def main():
    sFileName = "truthsocial.graphml"
    user_posts_file = "truthsocial.json"
    checkpoint_file = "checkpoint.json"
    
    # Ukraine conflict hashtags discovered from truthbrush search
    search_terms = [
        "UkraineConflict",
        "RussiaUkraineConflict", 
        "ukraine",  # Keep the plain word as well
        "russia",   # Add plain word searches too
        "putin"     # Add other relevant terms
    ]
    
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
    
    # Load existing data
    print("Loading existing data...")
    existing_posts, processed_post_ids, existing_summary = load_existing_data(user_posts_file)
    print(f"Loaded {len(existing_posts)} existing posts with {len(processed_post_ids)} processed IDs")
    
    # Initialize data structures
    graph = nx.DiGraph()
    user_data = defaultdict(lambda: {'posts': [], 'total_posts': 0, 'total_interactions': 0, 'replies_made': 0, 'replies_received': 0})
    all_posts_with_replies = existing_posts.copy()
    
    # Load checkpoint if exists
    checkpoint_data = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            print(f"Loaded checkpoint: {checkpoint_data.get('current_phase', 'unknown phase')}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    # Determine what phase we're in
    current_phase = checkpoint_data.get('current_phase', 'search_recent')
    current_term_index = checkpoint_data.get('current_term_index', 0)
    
    # Phase 1: Search recent posts for each term
    if current_phase == 'search_recent':
        print("=== PHASE 1: Searching recent posts for Ukraine conflict hashtags ===")
        
        for i, search_term in enumerate(search_terms[current_term_index:], start=current_term_index):
            print(f"\nSearching recent posts for: '{search_term}' ({i+1}/{len(search_terms)})")
            
            posts = search_with_pagination(search_term, max_id=None, limit_pages=3)
            
            if posts == "RATE_LIMITED":
                print(f"RATE LIMITED: Saving checkpoint at term {i}")
                checkpoint_data = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'current_phase': 'search_recent',
                    'current_term_index': i,
                    'processed_post_ids': list(processed_post_ids),
                    'collected_posts': len(all_posts_with_replies),
                    'message': f'Rate limited while searching recent posts for "{search_term}"'
                }
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                print(f"Checkpoint saved. Run script again to resume.")
                return
            
            # Filter out already processed posts
            new_posts = [post for post in posts if post.get('id') and post.get('id') not in processed_post_ids]
            print(f"Found {len(new_posts)} new posts for '{search_term}'")
            
            all_posts_with_replies.extend(new_posts)
            for post in new_posts:
                processed_post_ids.add(post.get('id'))
        
        # Move to next phase
        current_phase = 'search_historical'
        current_term_index = 0
        
        # Save intermediate progress after Phase 1
        print(f"\n=== SAVING INTERMEDIATE PROGRESS AFTER PHASE 1 ===")
        print(f"Collected {len(all_posts_with_replies)} posts so far")
        
        # Process all collected posts to build graph and user data for intermediate save
        temp_graph = nx.DiGraph()
        temp_user_data = defaultdict(lambda: {'posts': [], 'total_posts': 0, 'total_interactions': 0, 'replies_made': 0, 'replies_received': 0})
        
        for post in all_posts_with_replies:
            author = (post.get('account', {}).get('username') or 
                      post.get('username') or 
                      f"user_{post.get('account_id', 'unknown')}")
            
            replied_to_user = f"user_{post['in_reply_to_account_id']}" if post.get('in_reply_to_account_id') else None
            content = post.get('content', '') or post.get('text', '')
            interactions = post.get('favourites_count', 0) + post.get('reblogs_count', 0) + post.get('replies_count', 0)
            
            temp_user_data[author]['posts'].append({
                'content': post.get('content', ''),
                'created_at': post.get('created_at', ''),
                'likes_count': post.get('favourites_count', 0),
                'reposts_count': post.get('reblogs_count', 0),
                'replies_count': post.get('replies_count', 0),
                'is_reply': replied_to_user is not None,
                'replied_to': replied_to_user,
                'post_id': post.get('id')
            })
            temp_user_data[author]['total_posts'] += 1
            temp_user_data[author]['total_interactions'] += interactions
        
        # Save intermediate results
        temp_data_dict = dict(temp_user_data)
        intermediate_summary = {
            'search_terms': search_terms,
            'total_users': len(temp_data_dict),
            'total_posts': sum(user['total_posts'] for user in temp_data_dict.values()),
            'total_interactions': sum(user['total_interactions'] for user in temp_data_dict.values()),
            'collection_date': datetime.datetime.now().isoformat(),
            'data_collection_method': 'comprehensive_multi_hashtag_search',
            'total_posts_collected': len(all_posts_with_replies),
            'unique_post_ids': len(processed_post_ids),
            'phase_completed': 'recent_posts_search'
        }
        
        intermediate_data = {
            'summary': intermediate_summary,
            'users': temp_data_dict
        }
        
        # Save intermediate file
        intermediate_file = "truthsocial_intermediate.json"
        with open(intermediate_file, "w", encoding="utf-8") as json_file:
            json.dump(intermediate_data, json_file, indent=2, ensure_ascii=False)
        
        print(f"Intermediate progress saved to {intermediate_file}")

    # Phase 2: Search historical posts (backward pagination)
    if current_phase == 'search_historical':
        print("\n=== PHASE 2: Searching historical posts for Ukraine conflict hashtags ===")
        
        # Get the oldest post ID from our current data for backward pagination
        oldest_id = get_oldest_post_id(all_posts_with_replies)
        print(f"Using oldest post ID for historical search: {oldest_id}")
        
        for i, search_term in enumerate(search_terms[current_term_index:], start=current_term_index):
            print(f"\nSearching historical posts for: '{search_term}' ({i+1}/{len(search_terms)})")
            
            posts = search_with_pagination(search_term, max_id=oldest_id, limit_pages=5)
            
            if posts == "RATE_LIMITED":
                print(f"RATE LIMITED: Saving checkpoint at term {i}")
                checkpoint_data = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'current_phase': 'search_historical',
                    'current_term_index': i,
                    'processed_post_ids': list(processed_post_ids),
                    'collected_posts': len(all_posts_with_replies),
                    'message': f'Rate limited while searching historical posts for "{search_term}"'
                }
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                print(f"Checkpoint saved. Run script again to resume.")
                return
            
            # Filter out already processed posts
            new_posts = [post for post in posts if post.get('id') and post.get('id') not in processed_post_ids]
            print(f"Found {len(new_posts)} new historical posts for '{search_term}'")
            
            all_posts_with_replies.extend(new_posts)
            for post in new_posts:
                processed_post_ids.add(post.get('id'))
        
        # Move to next phase
        current_phase = 'process_final_data'  # Skip comments for now
    
    # Phase 3: Collect comments for posts that don't have them yet (OPTIONAL)
    if current_phase == 'collect_comments':
        print("\n=== PHASE 3: Collecting comments (OPTIONAL) ===")
        print("Skipping comment collection to avoid rate limits. Run separately if needed.")
        current_phase = 'process_final_data'
    
    # Final Phase: Process all data
    if current_phase == 'process_final_data':
        print(f"\n=== FINAL PHASE: Processing {len(all_posts_with_replies)} total posts ===")
    
    print(f"\n=== PROCESSING {len(all_posts_with_replies)} TOTAL POSTS ===")
    
    # Process all collected posts to build graph and user data
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
    
    # Save network graph
    nx.write_graphml(graph, sFileName)
    
    # Prepare final data
    user_data_dict = dict(user_data)
    
    reply_edges = sum(1 for _, _, data in graph.edges(data=True) if data.get('interaction_type') == 'reply')
    mention_edges = sum(1 for _, _, data in graph.edges(data=True) if data.get('interaction_type') == 'mention')
    
    summary_stats = {
        'search_terms': search_terms,
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
        'data_collection_method': 'comprehensive_multi_term_search_with_pagination',
        'total_posts_collected': len(all_posts_with_replies),
        'unique_post_ids': len(processed_post_ids)
    }
    
    final_data = {
        'summary': summary_stats,
        'users': user_data_dict
    }
    
    # Save the comprehensive dataset
    with open(user_posts_file, "w", encoding="utf-8") as json_file:
        json.dump(final_data, json_file, indent=2, ensure_ascii=False)
    
    # Clean up checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    # Show final summary
    print(f"\n=== FINAL DATA COLLECTION SUMMARY ===")
    print(f"Search terms used: {', '.join(search_terms)}")
    print(f"Total posts collected: {len(all_posts_with_replies)}")
    print(f"Unique post IDs: {len(processed_post_ids)}")
    print(f"Unique users found: {len(user_data_dict)}")
    print(f"Network nodes: {len(graph.nodes())}")
    print(f"Network edges: {len(graph.edges())}")
    print(f"Reply relationships: {reply_edges}")
    print(f"Mention relationships: {mention_edges}")
    print(f"Files saved: {sFileName}, {user_posts_file}")
    print(f"====================================\n")

if __name__ == "__main__":
    main()