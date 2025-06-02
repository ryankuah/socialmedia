import networkx as nx
import json
import datetime
from collections import defaultdict
from redditClient import redditClient

def main():
    graph_file = "reddit.graphml"
    json_file = "reddit.json"
    client = redditClient()
    reply_graph = nx.DiGraph()
    submission_comment_mapping = dict()
    user_data = defaultdict(lambda: {
        'submissions': [],
        'comments': [],
        'total_posts': 0,
        'total_score': 0,
        'first_post_date': None,
        'last_post_date': None
    })
    subreddit = client.subreddit('UkrainianConflict')
    for submission in subreddit.hot(limit=1000):
        submission_author_name = str(submission.author) if submission.author else None
        if (submission_author_name and 
            submission_author_name != 'None' and 
            submission_author_name != '[deleted]' and 
            submission_author_name != 'AutoModerator'):
            if submission.author.name in reply_graph:
                reply_graph.nodes[submission.author.name]['subNum'] += 1
            else:
                reply_graph.add_node(submission.author.name, subNum=1)
            submission_id = submission.name
            submission_comment_mapping[submission_id] = {submission_id : submission.author.name}
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
                submission_comment_mapping[submission_id].update({comment.name : comment.author.name})
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
                if comment.parent_id in submission_comment_mapping[submission_id]:
                    if reply_graph.has_edge(comment.author.name, submission_comment_mapping[submission_id][comment.parent_id]):
                        reply_graph[comment.author.name][submission_comment_mapping[submission_id][comment.parent_id]]['replyNum'] += 1
                    else:
                        if not comment.author.name in reply_graph:
                            reply_graph.add_node(comment.author.name, subNum=0)
                        if not submission_comment_mapping[submission_id][comment.parent_id] in reply_graph:
                            reply_graph.add_node(submission_comment_mapping[submission_id][comment.parent_id], subNum=0)
                        reply_graph.add_edge(comment.author.name, submission_comment_mapping[submission_id][comment.parent_id], replyNum=1)
    nx.readwrite.write_graphml(reply_graph, graph_file)
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
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    