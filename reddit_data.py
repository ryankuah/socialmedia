import networkx as nx
import matplotlib.pyplot as plt
import json
import datetime
from collections import defaultdict
from redditClient import redditClient


def main():
    sFileName = "reddit.graphml"
    user_posts_file = "reddit.json"
    
    client = redditClient()
    
    replyGraph = nx.DiGraph()
    
    dSubCommentId = dict()
    
    user_data = defaultdict(lambda: {
        'submissions': [],
        'comments': [],
        'total_posts': 0,
        'total_score': 0,
        'first_post_date': None,
        'last_post_date': None
    })
    
    subreddit = client.subreddit('UkrainianConflict')
    
    submissions_processed = 0
    
    for submission in subreddit.hot(limit=1000):
        submissions_processed += 1
        
        submission_author_name = str(submission.author) if submission.author else None
        if (submission_author_name and 
            submission_author_name != 'None' and 
            submission_author_name != '[deleted]' and 
            submission_author_name != 'AutoModerator'):
            
            if submission.author.name in replyGraph:
                replyGraph.nodes[submission.author.name]['subNum'] += 1
            else:
                replyGraph.add_node(submission.author.name, subNum=1)
            
            submissionId = submission.name
            dSubCommentId[submissionId] = {submissionId : submission.author.name}

        
        submission_author = str(submission.author)
        if (submission_author != 'None' and 
            submission_author != '[deleted]' and 
            submission_author != 'AutoModerator'):
            submission_data = {
                "title": submission.title,
                "selftext": submission.selftext,
                "score": submission.score,
                "created": submission.created_utc,
                "created_readable": datetime.datetime.fromtimestamp(submission.created_utc).isoformat(),
                "url": submission.url,
                "num_comments": submission.num_comments,
                "upvote_ratio": submission.upvote_ratio if hasattr(submission, 'upvote_ratio') else None
            }
            
            user_data[submission_author]['submissions'].append(submission_data)
            user_data[submission_author]['total_posts'] += 1
            user_data[submission_author]['total_score'] += submission.score
            
            post_date = datetime.datetime.fromtimestamp(submission.created_utc)
            if user_data[submission_author]['first_post_date'] is None:
                user_data[submission_author]['first_post_date'] = post_date.isoformat()
                user_data[submission_author]['last_post_date'] = post_date.isoformat()
            else:
                if post_date < datetime.datetime.fromisoformat(user_data[submission_author]['first_post_date']):
                    user_data[submission_author]['first_post_date'] = post_date.isoformat()
                if post_date > datetime.datetime.fromisoformat(user_data[submission_author]['last_post_date']):
                    user_data[submission_author]['last_post_date'] = post_date.isoformat()
        
        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            
            comment_author_name = str(comment.author) if comment.author else None
            if (comment.author is not None and 
                comment_author_name != 'AutoModerator'):
                
                dSubCommentId[submissionId].update({comment.name : comment.author.name})
                
                comment_author = str(comment.author)
                if (comment_author != 'None' and 
                    comment_author != '[deleted]' and 
                    comment_author != 'AutoModerator'):
                    comment_data = {
                        "body": comment.body,
                        "score": comment.score,
                        "created": comment.created_utc,
                        "created_readable": datetime.datetime.fromtimestamp(comment.created_utc).isoformat(),
                        "parent_submission_title": submission.title,
                        "parent_submission_author": str(submission.author),
                        "is_submitter": comment.is_submitter if hasattr(comment, 'is_submitter') else False
                    }
                    
                    user_data[comment_author]['comments'].append(comment_data)
                    user_data[comment_author]['total_posts'] += 1
                    user_data[comment_author]['total_score'] += comment.score
                    
                    post_date = datetime.datetime.fromtimestamp(comment.created_utc)
                    if user_data[comment_author]['first_post_date'] is None:
                        user_data[comment_author]['first_post_date'] = post_date.isoformat()
                        user_data[comment_author]['last_post_date'] = post_date.isoformat()
                    else:
                        if post_date < datetime.datetime.fromisoformat(user_data[comment_author]['first_post_date']):
                            user_data[comment_author]['first_post_date'] = post_date.isoformat()
                        if post_date > datetime.datetime.fromisoformat(user_data[comment_author]['last_post_date']):
                            user_data[comment_author]['last_post_date'] = post_date.isoformat()
                
                if comment.parent_id in dSubCommentId[submissionId]:
                    if replyGraph.has_edge(comment.author.name, dSubCommentId[submissionId][comment.parent_id]):
                        replyGraph[comment.author.name][dSubCommentId[submissionId][comment.parent_id]]['replyNum'] += 1
                    else:
                        if not comment.author.name in replyGraph:
                            replyGraph.add_node(comment.author.name, subNum=0)
                        
                        if not dSubCommentId[submissionId][comment.parent_id] in replyGraph:
                            replyGraph.add_node(dSubCommentId[submissionId][comment.parent_id], subNum=0)
                        
                        replyGraph.add_edge(comment.author.name, dSubCommentId[submissionId][comment.parent_id], replyNum=1)
        
        if submissions_processed % 10 == 0:
            pass
    
    nx.readwrite.write_graphml(replyGraph, sFileName)
    
    user_data_dict = dict(user_data)
    
    summary_stats = {
        'total_users': len(user_data_dict),
        'total_submissions': sum(len(user['submissions']) for user in user_data_dict.values()),
        'total_comments': sum(len(user['comments']) for user in user_data_dict.values()),
        'collection_date': datetime.datetime.now().isoformat(),
        'subreddit': subreddit.display_name
    }
    
    final_data = {
        'summary': summary_stats,
        'users': user_data_dict
    }
    
    with open(user_posts_file, "w", encoding="utf-8") as json_file:
        json.dump(final_data, json_file, indent=2, ensure_ascii=False)
    