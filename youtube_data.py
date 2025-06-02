import os
import json
import datetime
import time
from collections import defaultdict
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import networkx as nx

def main():
    api_key = ""
    
    graph_file = "youtube.graphml"
    json_file = "youtube.json"
    
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    reply_graph = nx.DiGraph()
    
    user_data = defaultdict(lambda: {
        'videos': [],
        'comments': [],
        'total_posts': 0,
        'total_score': 0,
        'first_post_date': None,
        'last_post_date': None
    })
    
    query = "Ukraine Conflict"
    max_results = 250
    
    # Get all videos
    all_videos = []
    next_page_token = None
    
    while len(all_videos) < max_results:
        try:
            remaining = max_results - len(all_videos)
            current_max = min(50, remaining)
            
            search_response = youtube.search().list(
                q=query,
                part="id,snippet",
                maxResults=current_max,
                type="video",
                pageToken=next_page_token
            ).execute()
            
            all_videos.extend(search_response["items"])
            
            next_page_token = search_response.get("nextPageToken")
            if not next_page_token:
                break
                
        except Exception as e:
            break
    
    for item in all_videos:
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        channel_title = item["snippet"]["channelTitle"]
        published_at = item["snippet"]["publishedAt"]
        
        video_author_name = channel_title
        
        # Check if valid author
        if (video_author_name and 
            video_author_name != 'None'):
            
            # Add or update graph node for video author
            if video_author_name in reply_graph:
                reply_graph.nodes[video_author_name]['subNum'] += 1
            else:
                reply_graph.add_node(video_author_name, subNum=1)
            
            # Parse date
            post_date = datetime.datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            
            # Only keep essential video data for analysis
            video_data_entry = {
                "title": title,
                "score": 1,  # Videos get score of 1
                "created": post_date.timestamp()
            }
            
            user_data[video_author_name]['videos'].append(video_data_entry)
            user_data[video_author_name]['total_posts'] += 1
            user_data[video_author_name]['total_score'] += 1
            
            # Update user dates
            if user_data[video_author_name]['first_post_date'] is None:
                user_data[video_author_name]['first_post_date'] = post_date.isoformat()
                user_data[video_author_name]['last_post_date'] = post_date.isoformat()
            else:
                if post_date < datetime.datetime.fromisoformat(user_data[video_author_name]['first_post_date']):
                    user_data[video_author_name]['first_post_date'] = post_date.isoformat()
                if post_date > datetime.datetime.fromisoformat(user_data[video_author_name]['last_post_date']):
                    user_data[video_author_name]['last_post_date'] = post_date.isoformat()
        
        try:
            # Get all comments for video
            all_comments = []
            next_page_token = None
            comments_fetched = 0
            max_comments = 150
            
            while comments_fetched < max_comments:
                try:
                    remaining = max_comments - comments_fetched
                    current_max = min(100, remaining)
                    
                    comment_response = youtube.commentThreads().list(
                        videoId=video_id,
                        part="snippet,replies",
                        maxResults=current_max,
                        textFormat="plainText",
                        pageToken=next_page_token,
                        order="relevance"
                    ).execute()
                    
                    batch_comments = comment_response["items"]
                    all_comments.extend(batch_comments)
                    comments_fetched += len(batch_comments)
                    
                    next_page_token = comment_response.get("nextPageToken")
                    if not next_page_token or comments_fetched >= max_comments:
                        break
                        
                except Exception as e:
                    break
            
            if all_comments:
                for comment_item in all_comments:
                    # Process top-level comment
                    top_comment = comment_item["snippet"]["topLevelComment"]["snippet"]
                    comment_author = top_comment["authorDisplayName"]
                    comment_text = top_comment["textDisplay"]
                    comment_likes = top_comment.get("likeCount", 0)
                    comment_published = top_comment["publishedAt"]
                    
                    # Check if valid comment author
                    if (comment_author and 
                        comment_author != 'None' and 
                        comment_author != '[deleted]'):
                        
                        # Parse comment date
                        comment_post_date = datetime.datetime.fromisoformat(comment_published.replace('Z', '+00:00'))
                        
                        # Only keep essential comment data for analysis
                        comment_data_entry = {
                            "body": comment_text,
                            "score": comment_likes,
                            "created": comment_post_date.timestamp()
                        }
                        
                        user_data[comment_author]['comments'].append(comment_data_entry)
                        user_data[comment_author]['total_posts'] += 1
                        user_data[comment_author]['total_score'] += comment_likes
                        
                        # Update user dates for comment author
                        if user_data[comment_author]['first_post_date'] is None:
                            user_data[comment_author]['first_post_date'] = comment_post_date.isoformat()
                            user_data[comment_author]['last_post_date'] = comment_post_date.isoformat()
                        else:
                            if comment_post_date < datetime.datetime.fromisoformat(user_data[comment_author]['first_post_date']):
                                user_data[comment_author]['first_post_date'] = comment_post_date.isoformat()
                            if comment_post_date > datetime.datetime.fromisoformat(user_data[comment_author]['last_post_date']):
                                user_data[comment_author]['last_post_date'] = comment_post_date.isoformat()
                        
                        # Add or update graph node for comment author
                        if comment_author not in reply_graph:
                            reply_graph.add_node(comment_author, subNum=0)
                        
                        # Add edge from video author to comment author (if valid video author)
                        if (video_author_name and 
                            video_author_name != 'None' and 
                            video_author_name != comment_author):
                            
                            if video_author_name not in reply_graph:
                                reply_graph.add_node(video_author_name, subNum=0)
                            
                            if reply_graph.has_edge(video_author_name, comment_author):
                                reply_graph[video_author_name][comment_author]['replyNum'] += 1
                            else:
                                reply_graph.add_edge(video_author_name, comment_author, replyNum=1)
                    
                    # Process replies
                    if "replies" in comment_item:
                        replies = comment_item["replies"]["comments"]
                        for reply in replies:
                            reply_snippet = reply["snippet"]
                            reply_author = reply_snippet["authorDisplayName"]
                            reply_text = reply_snippet["textDisplay"]
                            reply_likes = reply_snippet.get("likeCount", 0)
                            reply_published = reply_snippet["publishedAt"]
                            
                            # Check if valid reply author
                            if (reply_author and 
                                reply_author != 'None' and 
                                reply_author != '[deleted]'):
                                
                                # Parse reply date
                                reply_post_date = datetime.datetime.fromisoformat(reply_published.replace('Z', '+00:00'))
                                
                                # Only keep essential reply data for analysis
                                reply_data_entry = {
                                    "body": reply_text,
                                    "score": reply_likes,
                                    "created": reply_post_date.timestamp()
                                }
                                
                                user_data[reply_author]['comments'].append(reply_data_entry)
                                user_data[reply_author]['total_posts'] += 1
                                user_data[reply_author]['total_score'] += reply_likes
                                
                                # Update user dates for reply author
                                if user_data[reply_author]['first_post_date'] is None:
                                    user_data[reply_author]['first_post_date'] = reply_post_date.isoformat()
                                    user_data[reply_author]['last_post_date'] = reply_post_date.isoformat()
                                else:
                                    if reply_post_date < datetime.datetime.fromisoformat(user_data[reply_author]['first_post_date']):
                                        user_data[reply_author]['first_post_date'] = reply_post_date.isoformat()
                                    if reply_post_date > datetime.datetime.fromisoformat(user_data[reply_author]['last_post_date']):
                                        user_data[reply_author]['last_post_date'] = reply_post_date.isoformat()
                                
                                # Add or update graph node for reply author
                                if reply_author not in reply_graph:
                                    reply_graph.add_node(reply_author, subNum=0)
                                
                                # Add edge from comment author to reply author (if valid comment author)
                                if (comment_author and 
                                    comment_author != 'None' and 
                                    comment_author != '[deleted]' and
                                    comment_author != reply_author):
                                    
                                    if comment_author not in reply_graph:
                                        reply_graph.add_node(comment_author, subNum=0)
                                    
                                    if reply_graph.has_edge(comment_author, reply_author):
                                        reply_graph[comment_author][reply_author]['replyNum'] += 1
                                    else:
                                        reply_graph.add_edge(comment_author, reply_author, replyNum=1)
        
        except Exception as e:
            pass
    
    nx.readwrite.write_graphml(reply_graph, graph_file)
    
    user_data_dict = dict(user_data)
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(user_data_dict, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main() 
