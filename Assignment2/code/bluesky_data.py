import json
import networkx as nx
import datetime
import time
import os
import re
from collections import defaultdict
import argparse

try:
    from atproto import Client
    from atproto.exceptions import RequestException, BadRequestError
except ImportError:
    exit(1)

def main():
    handle = os.getenv('BLUESKY_HANDLE')
    password = os.getenv('BLUESKY_PASSWORD')
    if not handle or not password:
        return
    
    # Authenticate client
    try:
        client = Client()
        client.login(handle, password)
    except Exception:
        return
    
    query = "ukraineconflict"
    max_posts = 1000
    
    reply_graph = nx.DiGraph()
    user_data = defaultdict(lambda: {
        'posts': [],
        'total_posts': 0,
        'total_score': 0,
        'first_post_date': None,
        'last_post_date': None
    })
    
    # Search for posts
    search_posts_data = []
    cursor = None
    try:
        while len(search_posts_data) < max_posts:
            search_params = {
                'q': query,
                'limit': min(25, max_posts - len(search_posts_data))
            }
            if cursor:
                search_params['cursor'] = cursor
            response = client.app.bsky.feed.search_posts(search_params)
            if not response.posts:
                break
            search_posts_data.extend(response.posts)
            cursor = getattr(response, 'cursor', None)
            if not cursor:
                break
            time.sleep(1)
    except Exception:
        pass
    
    if not search_posts_data:
        return
    
    # Process posts and get replies
    all_posts = []
    processed_uris = set()
    did_to_handle = {}
    
    for i, post in enumerate(search_posts_data):
        all_posts.append(post)
        processed_uris.add(post.uri)
        
        # Get replies for this post
        try:
            response = client.app.bsky.feed.get_post_thread({'uri': post.uri, 'depth': 2})
            replies = []
            if hasattr(response, 'thread') and hasattr(response.thread, 'replies'):
                for reply in response.thread.replies or []:
                    if hasattr(reply, 'post'):
                        replies.append(reply.post)
                        if hasattr(reply, 'replies'):
                            for nested_reply in reply.replies or []:
                                if hasattr(nested_reply, 'post'):
                                    replies.append(nested_reply.post)
            
            for reply in replies[:50]:  # Limit replies
                if reply.uri not in processed_uris:
                    all_posts.append(reply)
                    processed_uris.add(reply.uri)
        except Exception:
            pass
        
        if i % 50 == 0:
            time.sleep(2)
        else:
            time.sleep(0.5)
    
    # Build DID to handle mapping
    for post in all_posts:
        author = post.author
        if author and author.handle and author.did:
            did_to_handle[author.did] = author.handle
    
    # Process all posts
    for post in all_posts:
        author = post.author
        author_handle = author.handle
        if not author_handle:
            continue
        
        # Check if valid author
        if (author_handle and 
            author_handle != 'None' and 
            author_handle != '[deleted]' and 
            author_handle != 'AutoModerator'):
            
            post_record = post
            post_data = post_record.record
            content = getattr(post_data, 'text', '')
            created_at = getattr(post_data, 'created_at', '')
            reply_info = getattr(post_data, 'reply', None)
            is_reply = reply_info is not None
            replied_to_user = None
            
            # Handle reply relationships
            if is_reply and reply_info.parent:
                parent_uri = reply_info.parent.uri
                if parent_uri:
                    try:
                        parent_parts = parent_uri.split('/')
                        if len(parent_parts) >= 3:
                            parent_did = parent_parts[2]
                            replied_to_user = did_to_handle.get(parent_did)
                    except:
                        pass
            
            likes_count = getattr(post_record, 'like_count', 0)
            reposts_count = getattr(post_record, 'repost_count', 0)
            replies_count = getattr(post_record, 'reply_count', 0)
            interactions = likes_count + reposts_count + replies_count
            
            # Parse date
            post_date = None
            if created_at:
                try:
                    post_date = datetime.datetime.fromisoformat(str(created_at).replace('Z', '+00:00'))
                except:
                    pass
            
            # Add or update graph node
            if author_handle in reply_graph:
                reply_graph.nodes[author_handle]['subNum'] += 1
            else:
                reply_graph.add_node(author_handle, subNum=1)
            
            # Only keep essential post data for analysis
            post_data_entry = {
                "content": content,
                "score": interactions,
                "created": post_date.timestamp() if post_date else 0
            }
            
            user_data[author_handle]['posts'].append(post_data_entry)
            user_data[author_handle]['total_posts'] += 1
            user_data[author_handle]['total_score'] += interactions
            
            # Update user dates
            if post_date:
                if user_data[author_handle]['first_post_date'] is None:
                    user_data[author_handle]['first_post_date'] = post_date.isoformat()
                    user_data[author_handle]['last_post_date'] = post_date.isoformat()
                else:
                    if post_date < datetime.datetime.fromisoformat(user_data[author_handle]['first_post_date']):
                        user_data[author_handle]['first_post_date'] = post_date.isoformat()
                    if post_date > datetime.datetime.fromisoformat(user_data[author_handle]['last_post_date']):
                        user_data[author_handle]['last_post_date'] = post_date.isoformat()
            
            # Handle reply edges
            if is_reply and replied_to_user and replied_to_user != author_handle:
                if replied_to_user not in reply_graph:
                    reply_graph.add_node(replied_to_user, subNum=0)
                
                if reply_graph.has_edge(author_handle, replied_to_user):
                    reply_graph[author_handle][replied_to_user]['replyNum'] += 1
                else:
                    reply_graph.add_edge(author_handle, replied_to_user, replyNum=1)
            
            # Handle mention edges
            mentions = re.findall(r'@([a-zA-Z0-9._-]+)', content)
            for mention_user in mentions:
                mention_handle = f"{mention_user}.bsky.social"
                if mention_handle != author_handle and mention_handle != replied_to_user:
                    if mention_handle not in reply_graph:
                        reply_graph.add_node(mention_handle, subNum=0)
                    
                    if reply_graph.has_edge(author_handle, mention_handle):
                        reply_graph[author_handle][mention_handle]['replyNum'] += 1
                    else:
                        reply_graph.add_edge(author_handle, mention_handle, replyNum=1)
    
    # Save graph
    nx.write_graphml(reply_graph, "bluesky.graphml")
    
    # Save user data
    user_data_dict = dict(user_data)
    
    with open("bluesky.json", "w", encoding="utf-8") as f:
        json.dump(user_data_dict, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main() 