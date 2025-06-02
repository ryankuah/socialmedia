import requests
import json
import networkx as nx
import datetime
import time
import re
import os
from collections import defaultdict


def call_apify_hashtag_api(api_token, hashtag, max_posts=500, timeout=300):
    """Call the Apify Truth Social Hashtag Scraper API"""
    
    # Prepare the input data
    input_data = {
        "hashtag": hashtag,
        "maxPosts": max_posts,
        "cleanContent": True,
        "proxyConfiguration": {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"]
        }
    }
    
    try:
        # Start the actor run
        response = requests.post(
            f"https://api.apify.com/v2/acts/muhammetakkurtt~truthsocial-hashtag-scraper/runs?token={api_token}",
            json=input_data,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        if response.status_code != 201:
            return None
        
        run_info = response.json()
        run_id = run_info['data']['id']
        
        # Wait for the run to complete
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check run status
            status_response = requests.get(
                f"https://api.apify.com/v2/actor-runs/{run_id}?token={api_token}",
                timeout=30
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data['data']['status']
                
                if status == 'SUCCEEDED':
                    break
                elif status in ['FAILED', 'ABORTED', 'TIMED-OUT']:
                    return None
                
                # Wait before checking again
                time.sleep(10)
            else:
                time.sleep(10)
        
        else:
            return None
        
        # Get the results
        results_response = requests.get(
            f"https://api.apify.com/v2/datasets/{run_info['data']['defaultDatasetId']}/items?token={api_token}",
            timeout=60
        )
        
        if results_response.status_code == 200:
            results = results_response.json()
            return results
        else:
            return None
            
    except requests.exceptions.Timeout:
        return None
    except Exception as e:
        return None


def call_apify_comment_api(api_token, post_id, max_posts=50, sort_by="trending", timeout=300):
    """Call the Apify Truth Social Comment Scraper API"""
    
    # Prepare the input data
    input_data = {
        "postId": post_id,
        "maxPosts": max_posts,
        "sortBy": sort_by,
        "cleanContent": True,
        "proxyConfiguration": {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"]
        }
    }
    
    try:
        # Start the actor run
        response = requests.post(
            f"https://api.apify.com/v2/acts/muhammetakkurtt~truthsocial-comment-scraper/runs?token={api_token}",
            json=input_data,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        if response.status_code != 201:
            return []
        
        run_info = response.json()
        run_id = run_info['data']['id']
        
        # Wait for the run to complete
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check run status
            status_response = requests.get(
                f"https://api.apify.com/v2/actor-runs/{run_id}?token={api_token}",
                timeout=30
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data['data']['status']
                
                if status == 'SUCCEEDED':
                    break
                elif status in ['FAILED', 'ABORTED', 'TIMED-OUT']:
                    return []
                
                # Wait before checking again
                time.sleep(5)
            else:
                time.sleep(5)
        
        else:
            return []
        
        # Get the results
        results_response = requests.get(
            f"https://api.apify.com/v2/datasets/{run_info['data']['defaultDatasetId']}/items?token={api_token}",
            timeout=60
        )
        
        if results_response.status_code == 200:
            results = results_response.json()
            return results
        else:
            return []
            
    except requests.exceptions.Timeout:
        return []
    except Exception as e:
        return []


def extract_author_name(post):
    """Extract consistent author name from post"""
    # Try multiple ways to get the username
    username = (post.get('account', {}).get('username') or 
                post.get('username') or 
                post.get('author', {}).get('username'))
    
    if username:
        return username
    
    # Fallback to account ID
    account_id = (post.get('account', {}).get('id') or 
                  post.get('account_id') or 
                  post.get('id'))
    
    if account_id:
        return f"user_{account_id}"
    
    return "unknown_user"


def extract_replied_to_user(post):
    """Extract the user being replied to"""
    replied_to_account_id = post.get('in_reply_to_account_id')
    if replied_to_account_id:
        # Try to find the actual username if available in post data
        if post.get('in_reply_to'):
            replied_username = extract_author_name(post['in_reply_to'])
            return replied_username
        else:
            # Fallback to user_id format
            return f"user_{replied_to_account_id}"
    return None


def filter_ukraine_posts(posts):
    """Filter posts that mention Ukraine or related conflicts"""
    ukraine_keywords = [
        'ukraine', 'ukrainian', 'kyiv', 'kiev', 'zelensky', 'putin', 'russia', 'russian',
        'conflict', 'war', 'invasion', 'nato', 'crimea', 'donbas', 'donetsk', 'luhansk',
        'sanctions', 'military aid', 'defense', 'battlefield', 'ukraineconflict'
    ]
    
    filtered_posts = []
    for post in posts:
        content = (post.get('content', '') or post.get('text', '') or '').lower()
        
        # Check hashtags as well
        tags = post.get('tags', [])
        tag_names = [tag.get('name', '').lower() for tag in tags if isinstance(tag, dict)]
        all_tags = ' '.join(tag_names)
        
        # Include if content or tags contain keywords, or if it's a substantial post
        if (any(keyword in content for keyword in ukraine_keywords) or 
            any(keyword in all_tags for keyword in ukraine_keywords) or 
            'ukraineconflict' in all_tags or
            len(content) > 10):  # Include most posts since hashtag search is targeted
            filtered_posts.append(post)
    
    return filtered_posts


def process_single_post(post, user_data, graph):
    """Process a single post and update user_data and graph structures"""
    author = extract_author_name(post)
    replied_to_user = extract_replied_to_user(post)
    content = post.get('content', '') or post.get('text', '')
    
    # Extract mentions from content
    mentions = []
    if content:
        # Find @username mentions
        mentions = re.findall(r'@([a-zA-Z0-9_]+)', content)
        # Also check mentions array if available
        if post.get('mentions'):
            for mention in post.get('mentions', []):
                if isinstance(mention, dict) and mention.get('username'):
                    mentions.append(mention['username'])
    
    # Extract interaction counts (handle different API formats)
    likes = post.get('favourites_count', 0) or post.get('likes_count', 0) or 0
    reblogs = post.get('reblogs_count', 0) or post.get('reposts_count', 0) or 0
    replies = post.get('replies_count', 0) or 0
    interactions = likes + reblogs + replies
    
    # Ensure all values are integers
    likes = int(likes) if likes else 0
    reblogs = int(reblogs) if reblogs else 0
    replies = int(replies) if replies else 0
    interactions = int(interactions) if interactions else 0
    
    # Update user data
    user_data[author]['posts'].append({
        'content': content[:500] if content else "",  # Truncate very long content
        'created_at': post.get('created_at', ''),
        'likes_count': likes,
        'reposts_count': reblogs,
        'replies_count': replies,
        'is_reply': replied_to_user is not None,
        'replied_to': replied_to_user,
        'post_id': str(post.get('id', '')) or str(post.get('post_id', '')),
        'url': post.get('url', ''),
        'tags': [tag.get('name', '') for tag in post.get('tags', []) if isinstance(tag, dict)]
    })
    user_data[author]['total_posts'] += 1
    user_data[author]['total_interactions'] += interactions
    
    # Update graph - ensure all node attributes are proper types
    if not graph.has_node(author):
        graph.add_node(author, 
                      post_count=0, 
                      replies_made=0, 
                      replies_received=0, 
                      comments_made=0,
                      total_interactions=0,
                      node_type='user')
    
    # Update node attributes
    graph.nodes[author]['post_count'] = graph.nodes[author].get('post_count', 0) + 1
    graph.nodes[author]['total_interactions'] = graph.nodes[author].get('total_interactions', 0) + interactions
    
    # Handle reply relationships
    if replied_to_user and replied_to_user != author:
        if not graph.has_node(replied_to_user):
            graph.add_node(replied_to_user, 
                          post_count=0, 
                          replies_made=0, 
                          replies_received=0, 
                          comments_made=0,
                          total_interactions=0,
                          node_type='user')
        
        # Create or update reply edge
        if graph.has_edge(author, replied_to_user):
            graph[author][replied_to_user]['reply_count'] = graph[author][replied_to_user].get('reply_count', 0) + 1
        else:
            graph.add_edge(author, replied_to_user, 
                          reply_count=1, 
                          mention_count=0,
                          comment_count=0,
                          interaction_type='reply')
        
        # Update node counts
        graph.nodes[author]['replies_made'] = graph.nodes[author].get('replies_made', 0) + 1
        graph.nodes[replied_to_user]['replies_received'] = graph.nodes[replied_to_user].get('replies_received', 0) + 1
        user_data[author]['replies_made'] += 1
        if replied_to_user not in user_data:
            user_data[replied_to_user] = {'posts': [], 'total_posts': 0, 'total_interactions': 0, 'replies_made': 0, 'replies_received': 0, 'comments_made': 0}
        user_data[replied_to_user]['replies_received'] += 1
    
    # Handle mention relationships
    for mention in set(mentions):  # Use set to avoid duplicates
        if mention and mention not in (author, replied_to_user):
            if not graph.has_node(mention):
                graph.add_node(mention, 
                              post_count=0, 
                              replies_made=0, 
                              replies_received=0, 
                              comments_made=0,
                              total_interactions=0,
                              node_type='user')
            
            # Create or update mention edge
            if graph.has_edge(author, mention):
                graph[author][mention]['mention_count'] = graph[author][mention].get('mention_count', 0) + 1
            else:
                graph.add_edge(author, mention, 
                              reply_count=0,
                              mention_count=1, 
                              comment_count=0,
                              interaction_type='mention')


def process_comments(comments, user_data, graph):
    """Process comments and add to network"""
    comment_count = 0
    
    for comment in comments:
        author = extract_author_name(comment)
        replied_to_user = extract_replied_to_user(comment)
        content = comment.get('content', '') or comment.get('text', '')
        
        # Extract mentions from content
        mentions = []
        if content:
            mentions = re.findall(r'@([a-zA-Z0-9_]+)', content)
            # Also check mentions array if available
            if comment.get('mentions'):
                for mention in comment.get('mentions', []):
                    if isinstance(mention, dict) and mention.get('username'):
                        mentions.append(mention['username'])
        
        # Extract interaction counts
        likes = int(comment.get('favourites_count', 0) or 0)
        reblogs = int(comment.get('reblogs_count', 0) or 0)
        replies = int(comment.get('replies_count', 0) or 0)
        interactions = likes + reblogs + replies
        
        # Initialize user if not exists
        if author not in user_data:
            user_data[author] = {'posts': [], 'total_posts': 0, 'total_interactions': 0, 'replies_made': 0, 'replies_received': 0, 'comments_made': 0}
        
        # Add comment to user data
        user_data[author]['comments_made'] += 1
        user_data[author]['total_interactions'] += interactions
        
        # Update graph
        if not graph.has_node(author):
            graph.add_node(author, 
                          post_count=0, 
                          replies_made=0, 
                          replies_received=0, 
                          comments_made=0,
                          total_interactions=0,
                          node_type='user')
        
        graph.nodes[author]['comments_made'] = graph.nodes[author].get('comments_made', 0) + 1
        graph.nodes[author]['total_interactions'] = graph.nodes[author].get('total_interactions', 0) + interactions
        
        # Handle comment reply relationships
        if replied_to_user and replied_to_user != author:
            if not graph.has_node(replied_to_user):
                graph.add_node(replied_to_user, 
                              post_count=0, 
                              replies_made=0, 
                              replies_received=0, 
                              comments_made=0,
                              total_interactions=0,
                              node_type='user')
            
            # Create or update comment edge
            if graph.has_edge(author, replied_to_user):
                graph[author][replied_to_user]['comment_count'] = graph[author][replied_to_user].get('comment_count', 0) + 1
            else:
                graph.add_edge(author, replied_to_user, 
                              reply_count=0,
                              mention_count=0,
                              comment_count=1, 
                              interaction_type='comment')
            
            # Update user data
            user_data[author]['replies_made'] += 1
            if replied_to_user not in user_data:
                user_data[replied_to_user] = {'posts': [], 'total_posts': 0, 'total_interactions': 0, 'replies_made': 0, 'replies_received': 0, 'comments_made': 0}
            user_data[replied_to_user]['replies_received'] += 1
        
        # Handle mentions in comments
        for mention in set(mentions):  # Use set to avoid duplicates
            if mention and mention not in (author, replied_to_user):
                if not graph.has_node(mention):
                    graph.add_node(mention, 
                                  post_count=0, 
                                  replies_made=0, 
                                  replies_received=0, 
                                  comments_made=0,
                                  total_interactions=0,
                                  node_type='user')
                
                # Create or update mention edge
                if graph.has_edge(author, mention):
                    graph[author][mention]['mention_count'] = graph[author][mention].get('mention_count', 0) + 1
                else:
                    graph.add_edge(author, mention, 
                                  reply_count=0,
                                  mention_count=1,
                                  comment_count=0,
                                  interaction_type='mention')
        
        comment_count += 1
    
    print(f"✅ Processed {comment_count} comments into network")


def generate_summary_stats(user_data_dict, graph, all_posts, all_comments):
    """Generate summary statistics"""
    reply_edges = sum(1 for _, _, data in graph.edges(data=True) if data.get('interaction_type') == 'reply')
    mention_edges = sum(1 for _, _, data in graph.edges(data=True) if data.get('interaction_type') == 'mention')
    comment_edges = sum(1 for _, _, data in graph.edges(data=True) if data.get('interaction_type') == 'comment')
    
    return {
        'search_method': 'Apify Truth Social Hashtag Scraper - UkraineConflict',
        'total_users': len(user_data_dict),
        'total_posts': sum(user['total_posts'] for user in user_data_dict.values()),
        'total_comments': len(all_comments),
        'total_interactions': sum(user['total_interactions'] for user in user_data_dict.values()),
        'total_replies_made': sum(user['replies_made'] for user in user_data_dict.values()),
        'total_replies_received': sum(user['replies_received'] for user in user_data_dict.values()),
        'total_comments_made': sum(user.get('comments_made', 0) for user in user_data_dict.values()),
        'network_nodes': len(graph.nodes()),
        'network_edges': len(graph.edges()),
        'reply_edges': reply_edges,
        'mention_edges': mention_edges,
        'comment_edges': comment_edges,
        'collection_date': datetime.datetime.now().isoformat(),
        'data_collection_method': 'apify_truthsocial_hashtag_and_comment_scraper',
        'total_posts_collected': len(all_posts),
        'total_comments_collected': len(all_comments)
    }


def main():
    api_token = os.getenv('APIFY_API_TOKEN')
    if not api_token:
        return
    
    graph = nx.DiGraph()
    user_data = defaultdict(lambda: {
        'posts': [],
        'total_posts': 0,
        'total_score': 0,
        'first_post_date': None,
        'last_post_date': None
    })
    
    hashtag = "UkraineConflict"
    
    # Call Apify hashtag API
    input_data = {
        "hashtag": hashtag,
        "maxPosts": 300,
        "cleanContent": True,
        "proxyConfiguration": {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"]
        }
    }
    
    try:
        # Start the actor run
        response = requests.post(
            f"https://api.apify.com/v2/acts/muhammetakkurtt~truthsocial-hashtag-scraper/runs?token={api_token}",
            json=input_data,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        if response.status_code != 201:
            return
        
        run_info = response.json()
        run_id = run_info['data']['id']
        
        # Wait for the run to complete
        timeout = 600
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check run status
            status_response = requests.get(
                f"https://api.apify.com/v2/actor-runs/{run_id}?token={api_token}",
                timeout=30
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data['data']['status']
                
                if status == 'SUCCEEDED':
                    break
                elif status in ['FAILED', 'ABORTED', 'TIMED-OUT']:
                    return
                
                time.sleep(10)
            else:
                time.sleep(10)
        else:
            return
        
        # Get the results
        results_response = requests.get(
            f"https://api.apify.com/v2/datasets/{run_info['data']['defaultDatasetId']}/items?token={api_token}",
            timeout=60
        )
        
        if results_response.status_code == 200:
            results = results_response.json()
        else:
            return
            
    except Exception as e:
        return
    
    if not results:
        return
    
    for post in results:
        # Extract author name
        username = (post.get('account', {}).get('username') or 
                   post.get('username') or 
                   post.get('author', {}).get('username'))
        
        if not username:
            account_id = (post.get('account', {}).get('id') or 
                         post.get('account_id') or 
                         post.get('id'))
            if account_id:
                username = f"user_{account_id}"
            else:
                username = "unknown_user"
        
        # Check if valid author
        if (username and 
            username != 'None' and 
            username != '[deleted]' and 
            username != 'AutoModerator'):
            
            # Add or update graph node
            if username in graph:
                graph.nodes[username]['subNum'] += 1
            else:
                graph.add_node(username, subNum=1)
            
            content = post.get('content', '') or post.get('text', '')
            created_at = post.get('created_at', '')
            likes = int(post.get('favourites_count', 0) or post.get('likes_count', 0) or 0)
            
            # Parse date if available
            post_date = None
            if created_at:
                try:
                    post_date = datetime.datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except:
                    pass
            
            # Only keep essential post data for analysis
            post_data_entry = {
                "content": content,
                "score": likes,
                "created": post_date.timestamp() if post_date else 0
            }
            
            user_data[username]['posts'].append(post_data_entry)
            user_data[username]['total_posts'] += 1
            user_data[username]['total_score'] += likes
            
            # Update user dates
            if post_date:
                if user_data[username]['first_post_date'] is None:
                    user_data[username]['first_post_date'] = post_date.isoformat()
                    user_data[username]['last_post_date'] = post_date.isoformat()
                else:
                    if post_date < datetime.datetime.fromisoformat(user_data[username]['first_post_date']):
                        user_data[username]['first_post_date'] = post_date.isoformat()
                    if post_date > datetime.datetime.fromisoformat(user_data[username]['last_post_date']):
                        user_data[username]['last_post_date'] = post_date.isoformat()
            
            # Handle replies
            replied_to_account_id = post.get('in_reply_to_account_id')
            if replied_to_account_id:
                if post.get('in_reply_to'):
                    replied_username = (post['in_reply_to'].get('account', {}).get('username') or 
                                      post['in_reply_to'].get('username') or 
                                      post['in_reply_to'].get('author', {}).get('username'))
                    if not replied_username:
                        replied_username = f"user_{replied_to_account_id}"
                else:
                    replied_username = f"user_{replied_to_account_id}"
                
                if (replied_username and 
                    replied_username != username and
                    replied_username != 'None' and 
                    replied_username != '[deleted]'):
                    
                    if replied_username not in graph:
                        graph.add_node(replied_username, subNum=0)
                    
                    if graph.has_edge(username, replied_username):
                        graph[username][replied_username]['replyNum'] += 1
                    else:
                        graph.add_edge(username, replied_username, replyNum=1)
        
        # Get comments for this post if it has replies
        replies_count = post.get('replies_count', 0) or 0
        if replies_count > 0:
            post_id = post.get('id')
            if post_id:
                # Call comment API
                comment_input_data = {
                    "postId": post_id,
                    "maxPosts": 50,
                    "sortBy": "trending",
                    "cleanContent": True,
                    "proxyConfiguration": {
                        "useApifyProxy": True,
                        "apifyProxyGroups": ["RESIDENTIAL"]
                    }
                }
                
                try:
                    comment_response = requests.post(
                        f"https://api.apify.com/v2/acts/muhammetakkurtt~truthsocial-comment-scraper/runs?token={api_token}",
                        json=comment_input_data,
                        headers={'Content-Type': 'application/json'},
                        timeout=60
                    )
                    
                    if comment_response.status_code == 201:
                        comment_run_info = comment_response.json()
                        comment_run_id = comment_run_info['data']['id']
                        
                        # Wait for comment run to complete
                        comment_start_time = time.time()
                        comment_timeout = 300
                        
                        while time.time() - comment_start_time < comment_timeout:
                            comment_status_response = requests.get(
                                f"https://api.apify.com/v2/actor-runs/{comment_run_id}?token={api_token}",
                                timeout=30
                            )
                            
                            if comment_status_response.status_code == 200:
                                comment_status_data = comment_status_response.json()
                                comment_status = comment_status_data['data']['status']
                                
                                if comment_status == 'SUCCEEDED':
                                    break
                                elif comment_status in ['FAILED', 'ABORTED', 'TIMED-OUT']:
                                    break
                                
                        time.sleep(5)
                else:
                                time.sleep(5)
                        
                        # Get comment results
                        if comment_status == 'SUCCEEDED':
                            comment_results_response = requests.get(
                                f"https://api.apify.com/v2/datasets/{comment_run_info['data']['defaultDatasetId']}/items?token={api_token}",
                                timeout=60
                            )
                            
                            if comment_results_response.status_code == 200:
                                comments = comment_results_response.json()
                                
                                for comment in comments:
                                    # Extract comment author name
                                    comment_username = (comment.get('account', {}).get('username') or 
                                                      comment.get('username') or 
                                                      comment.get('author', {}).get('username'))
                                    
                                    if not comment_username:
                                        comment_account_id = (comment.get('account', {}).get('id') or 
                                                           comment.get('account_id') or 
                                                           comment.get('id'))
                                        if comment_account_id:
                                            comment_username = f"user_{comment_account_id}"
                                        else:
                                            comment_username = "unknown_user"
                                    
                                    # Check if valid comment author
                                    if (comment_username and 
                                        comment_username != 'None' and 
                                        comment_username != '[deleted]'):
                                        
                                        comment_content = comment.get('content', '') or comment.get('text', '')
                                        comment_created_at = comment.get('created_at', '')
                                        comment_likes = int(comment.get('favourites_count', 0) or 0)
                                        
                                        # Parse comment date
                                        comment_post_date = None
                                        if comment_created_at:
                                            try:
                                                comment_post_date = datetime.datetime.fromisoformat(comment_created_at.replace('Z', '+00:00'))
                                            except:
                                                pass
                                        
                                        # Only keep essential comment data for analysis
                                        comment_data_entry = {
                                            "content": comment_content,
                                            "score": comment_likes,
                                            "created": comment_post_date.timestamp() if comment_post_date else 0
                                        }
                                        
                                        user_data[comment_username]['posts'].append(comment_data_entry)
                                        user_data[comment_username]['total_posts'] += 1
                                        user_data[comment_username]['total_score'] += comment_likes
                                        
                                        # Update user dates for comment author
                                        if comment_post_date:
                                            if user_data[comment_username]['first_post_date'] is None:
                                                user_data[comment_username]['first_post_date'] = comment_post_date.isoformat()
                                                user_data[comment_username]['last_post_date'] = comment_post_date.isoformat()
                                            else:
                                                if comment_post_date < datetime.datetime.fromisoformat(user_data[comment_username]['first_post_date']):
                                                    user_data[comment_username]['first_post_date'] = comment_post_date.isoformat()
                                                if comment_post_date > datetime.datetime.fromisoformat(user_data[comment_username]['last_post_date']):
                                                    user_data[comment_username]['last_post_date'] = comment_post_date.isoformat()
                                        
                                        # Add or update graph node for comment author
                                        if comment_username not in graph:
                                            graph.add_node(comment_username, subNum=0)
                                        
                                        # Handle comment reply relationships
                                        comment_replied_to_account_id = comment.get('in_reply_to_account_id')
                                        if comment_replied_to_account_id:
                                            if comment.get('in_reply_to'):
                                                comment_replied_username = (comment['in_reply_to'].get('account', {}).get('username') or 
                                                                          comment['in_reply_to'].get('username') or 
                                                                          comment['in_reply_to'].get('author', {}).get('username'))
                                                if not comment_replied_username:
                                                    comment_replied_username = f"user_{comment_replied_to_account_id}"
                                            else:
                                                comment_replied_username = f"user_{comment_replied_to_account_id}"
                                            
                                            if (comment_replied_username and 
                                                comment_replied_username != comment_username and
                                                comment_replied_username != 'None' and 
                                                comment_replied_username != '[deleted]'):
                                                
                                                if comment_replied_username not in graph:
                                                    graph.add_node(comment_replied_username, subNum=0)
                                                
                                                if graph.has_edge(comment_username, comment_replied_username):
                                                    graph[comment_username][comment_replied_username]['replyNum'] += 1
                                                else:
                                                    graph.add_edge(comment_username, comment_replied_username, replyNum=1)
                        
                        time.sleep(5)  # Rate limiting
                
                except Exception as e:
                    pass
    
    # Save graph
    try:
        graph_file = "truthsocial.graphml"
        nx.write_graphml(graph, graph_file)
    except Exception as e:
            pass
    
    # Save user data
    user_data_dict = dict(user_data)
    
    with open("truthsocial.json", "w", encoding="utf-8") as f:
        json.dump(user_data_dict, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main() 