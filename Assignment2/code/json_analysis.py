import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime
from collections import Counter
import re
import nltk
import gensim
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Social Media Data Analysis')
    parser.add_argument('json_file', help='Path to JSON file to analyze')
    
    args = parser.parse_args()
    
    base_name = os.path.splitext(os.path.basename(args.json_file))[0]
    output_dir = base_name
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load user data
        if not os.path.exists(args.json_file):
            raise FileNotFoundError(f"JSON file not found: {args.json_file}")
        
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        users = data.get('users', {})
        summary = data.get('summary', {})
        
        # If data is structured differently (direct user dict), handle it
        if not users and isinstance(data, dict):
            users = data
        
        # Extract all text content
        all_texts = []
        
        for username, data in users.items():
            if 'submissions' in data or 'comments' in data:
                for submission in data.get('submissions', []):
                    if submission.get('title'):
                        all_texts.append(submission['title'])
                    if submission.get('selftext') and submission['selftext'].strip():
                        all_texts.append(submission['selftext'])
                
                for comment in data.get('comments', []):
                    if comment.get('body') and comment['body'].strip():
                        all_texts.append(comment['body'])
            
            elif 'posts' in data:
                for post in data.get('posts', []):
                    if post.get('content') and post['content'].strip():
                        import html
                        
                        clean_content = html.unescape(post['content'])
                        clean_content = re.sub(r'<[^>]+>', '', clean_content)
                        clean_content = re.sub(r'https?://\S+', '', clean_content)
                        clean_content = re.sub(r'&\w+;', '', clean_content)
                        clean_content = re.sub(r'\s+', ' ', clean_content)
                        
                        if clean_content.strip():
                            all_texts.append(clean_content.strip())
                    
                    for field in ['text', 'caption', 'title']:
                        if post.get(field) and post[field].strip():
                            import html
                            clean_text = html.unescape(post[field])
                            clean_text = re.sub(r'<[^>]+>', '', clean_text)
                            clean_text = re.sub(r'https?://\S+', '', clean_text)
                            clean_text = re.sub(r'&\w+;', '', clean_text)
                            clean_text = re.sub(r'\s+', ' ', clean_text)
                            if clean_text.strip():
                                all_texts.append(clean_text.strip())
        
        filtered_texts = []
        for text in all_texts:
            if text and isinstance(text, str) and len(text.strip()) >= 3:
                filtered_texts.append(text.strip())
        
        if not filtered_texts:
            results = {
                'total_texts': 0, 'processed_docs': 0, 'sentiment_analysis': {},
                'topic_analysis': {}, 'word_frequency': {}, 'user_activity': {},
                'temporal_analysis': {}
            }
        else:
            # Preprocess texts
            processed_docs = []
            for text in filtered_texts:
                if not text or len(text.strip()) == 0:
                    processed_docs.append([])
                    continue
                
                text = text.lower()
                
                # Text replacements
                replacements = {
                    r'\bgovt\b': "government", r'\bppl\b': "people", r'\bpls\b': "please", 
                    r'\bu\b': "you", r'\bur\b': "your", r'\bthru\b': "through", 
                    r'\bw/\b': "with", r'\bb4\b': "before",
                    r"\bdon't\b": "do not", r"\bwon't\b": "will not", r"\bcan't\b": "cannot", 
                    r"\bshouldn't\b": "should not", r"\bwouldn't\b": "would not", r"\bcouldn't\b": "could not"
                }
                
                for pattern, replacement in replacements.items():
                    text = re.sub(pattern, replacement, text)
                
                try:
                    tokens = word_tokenize(text)
                except Exception as e:
                    processed_docs.append([])
                    continue
                
                lemmatizer = WordNetLemmatizer()
                processed_tokens = []
                
                for tok in tokens:
                    tok = tok.strip()
                    if (tok.isalnum() and len(tok) >= 3 and not tok.isdigit() and 
                        not tok.startswith('http')):
                        try:
                            pos_tags = nltk.pos_tag([tok])
                            pos = wordnet.NOUN
                            if pos_tags[0][1].startswith('J'):
                                pos = wordnet.ADJ
                            elif pos_tags[0][1].startswith('V'):
                                pos = wordnet.VERB
                            elif pos_tags[0][1].startswith('R'):
                                pos = wordnet.ADV
                            
                            lemma = lemmatizer.lemmatize(tok, pos=pos)
                            processed_tokens.append(lemma)
                        except Exception as e:
                            processed_tokens.append(tok)
                
                processed_docs.append(processed_tokens)
            
            # Remove stopwords
            custom_stopwords = [
                "post", "comment", "thread", "edit", "upvote", "downvote", "karma", "mod", "moderator",
                "like", "share", "follow", "followers", "delete", "remove", "update", "thanks", "thank",
                "day", "year", "world", "country", "peace", "support", "help", "get", "one", "people",
                "know", "say", "make", "well", "want", "may", "use", "hit", "please", "reddit", "love",
                "thing", "look", "much", "even", "time", "see", "think", "also", "back", "need", "come",
                "go", "take", "way", "new", "good", "right", "work", "state", "still", "find", "give",
                "part", "place", "case", "point", "group", "number", "fact", "hand", "high", "large",
                "public", "important", "different", "possible", "bad", "great", "little", "own", "old",
                "long", "man", "woman", "child", "life", "never", "home", "side", "eye", "head", "house",
                "service", "friend", "father", "power", "hour", "game", "line", "end", "member", "law",
                "car", "city", "community", "name", "president", "team", "minute", "idea", "kid", "lot",
                "turn", "put", "mean", "keep", "let", "begin", "seem", "talk", "start", "show", "hear",
                "play", "run", "move", "live", "believe", "hold", "bring", "happen", "write", "provide",
                "sit", "stand", "lose", "pay", "meet", "include", "continue", "set", "learn", "change",
                "lead", "understand", "try", "call", "tell", "ask", "feel", "leave", "think", "come",
                "go", "do", "make", "get", "see", "hear", "open", "close", "cut", "build", "send", "read",
                "stop", "win", "buy", "sell", "break", "fix", "kill", "die", "born", "grow", "fall", "rise",
                "fly", "drive", "walk", "stay", "would", "could", "should", "might", "must", "shall", "will",
                "anything", "something", "nothing", "everything", "somebody", "nobody", "anybody", "everybody",
                "someone", "anyone", "everyone", "somewhere", "anywhere", "everywhere", "nowhere", "somehow",
                "anyhow", "anyway", "always", "sometimes", "never", "often", "usually", "rarely", "seldom",
                "hardly", "barely", "nearly", "almost", "quite", "rather", "pretty", "fairly", "very",
                "extremely", "incredibly", "absolutely", "completely", "totally", "many", "actually",
                "sure", "probably", "deal", "maybe", "reason", "big", "far", "yes", "really", "enough", 
                "around", "another", "without", "between", "through", "during", "before", "after", "above", 
                "below", "down", "under", "again", "further", "then", "once", "here", "there", "when", 
                "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", 
                "some", "such", "no", "nor", "not", "only", "same", "than", "too", "can", "will", "just", 
                "should", "now", "lot", "dont", "doesnt", "didnt", "cant", "wont", "wouldnt", "couldnt", 
                "shouldnt", "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "hadnt", "going", "said", 
                "got", "first", "last", "next", "every", "made", "came", "went", "getting", "doing", "making", 
                "looking", "coming", "saying", "trying", "working", "giving", "taking", "real", "true", "false",
                "able", "better", "best", "worse", "worst", "less", "least", "full", "half", "whole",
                "small", "large", "short", "long", "early", "late", "young", "old", "strong", "weak"
            ]
            
            try:
                stop_words = set(stopwords.words("english") + custom_stopwords)
            except Exception as e:
                stop_words = set(custom_stopwords)
            
            processed_docs = [[word for word in doc if (word not in stop_words and len(word) >= 3 and 
                             len(word) <= 20 and word.isalpha())] for doc in processed_docs]
            
            non_empty_docs = [doc for doc in processed_docs if doc]
            
            all_words = [word for doc in non_empty_docs for word in doc]
            word_freq = Counter(all_words)
            
            # Sentiment analysis
            try:
                analyzer = SentimentIntensityAnalyzer()
            except Exception as e:
                analyzer = None
            
            sentiment_scores = []
            positive_count = negative_count = neutral_count = 0
            
            if analyzer:
                for text in filtered_texts:
                    if text and len(text.strip()) > 0:
                        try:
                            compound_score = analyzer.polarity_scores(text)['compound']
                            sentiment_scores.append(compound_score)
                            
                            if compound_score >= 0.05:
                                positive_count += 1
                            elif compound_score <= -0.05:
                                negative_count += 1
                            else:
                                neutral_count += 1
                        except Exception as e:
                            sentiment_scores.append(0)
                            neutral_count += 1
            
            total = len(sentiment_scores)
            
            sentiment_analysis = {
                'sentiment_scores': sentiment_scores,
                'avg_sentiment': np.mean(sentiment_scores) if sentiment_scores else 0,
                'positive_ratio': positive_count / total if total > 0 else 0,
                'negative_ratio': negative_count / total if total > 0 else 0,
                'neutral_ratio': neutral_count / total if total > 0 else 0,
                'total_analyzed': total
            }
            
            # Topic modeling
            topic_analysis = {'topics': [], 'coherence_score': 0}
            if non_empty_docs and len(non_empty_docs) >= 2:
                try:
                    dictionary = corpora.Dictionary(non_empty_docs)
                    dictionary.filter_extremes(no_below=5, no_above=0.60)
                    
                    if len(dictionary) > 0:
                        corpus = [dictionary.doc2bow(doc) for doc in non_empty_docs if doc]
                        if corpus:
                            max_topics = min(5, len(corpus), len(dictionary) // 4, 10)
                            if max_topics >= 2:
                                lda_model = gensim.models.LdaModel(
                                    corpus=corpus, num_topics=max_topics, id2word=dictionary,
                                    passes=20, iterations=400, random_state=42, alpha=0.1, eta=0.01
                                )
                                
                                try:
                                    coherence_model = CoherenceModel(
                                        model=lda_model, texts=non_empty_docs, dictionary=dictionary, coherence='c_v'
                                    )
                                    coherence_score = coherence_model.get_coherence()
                                except Exception as e:
                                    coherence_score = 0
                                
                                topics = []
                                for topic_id in range(lda_model.num_topics):
                                    topic_words = lda_model.show_topic(topic_id, topn=10)
                                    topics.append({
                                        'topic_id': topic_id,
                                        'words': [word for word, prob in topic_words],
                                        'probabilities': [prob for word, prob in topic_words]
                                    })
                                
                                topic_analysis = {'topics': topics, 'coherence_score': coherence_score}
                except Exception as e:
                    pass
            
            # User activity analysis
            post_times = []
            post_scores = []
            total_posts = 0
            user_stats = []
            
            for username, data in users.items():
                user_data = {'username': username, 'total_posts': 0, 'submissions': 0, 'comments': 0, 'posts': 0}
                
                if 'submissions' in data or 'comments' in data:
                    submissions = data.get('submissions', [])
                    comments = data.get('comments', [])
                    
                    user_data['submissions'] = len(submissions)
                    user_data['comments'] = len(comments)
                    user_data['total_posts'] = len(submissions) + len(comments)
                    
                    for submission in submissions:
                        if 'score' in submission:
                            post_scores.append(submission['score'])
                        if 'created' in submission:
                            post_times.append(submission['created'])
                    
                    for comment in comments:
                        if 'score' in comment:
                            post_scores.append(comment['score'])
                        if 'created' in comment:
                            post_times.append(comment['created'])
                
                elif 'posts' in data:
                    posts = data.get('posts', [])
                    user_data['posts'] = len(posts)
                    user_data['total_posts'] = len(posts)
                    
                    for post in posts:
                        if 'likes_count' in post:
                            post_scores.append(post['likes_count'])
                        if 'created_at' in post:
                            try:
                                dt = datetime.fromisoformat(post['created_at'].replace('Z', '+00:00'))
                                post_times.append(dt.timestamp())
                            except Exception as e:
                                pass
                
                if 'total_posts' in data and user_data['total_posts'] == 0:
                    user_data['total_posts'] = data['total_posts']
                
                total_posts += user_data['total_posts']
                user_stats.append(user_data)
            
            user_stats.sort(key=lambda x: x['total_posts'], reverse=True)
            
            # Temporal analysis
            temporal_stats = {'total_posts_with_timestamps': len(post_times)}
            if post_times:
                temporal_stats.update({
                    'earliest_post': min(post_times),
                    'latest_post': max(post_times)
                })
                
                datetimes = [datetime.fromtimestamp(ts) for ts in post_times]
                hour_counts = Counter(dt.hour for dt in datetimes)
                weekday_counts = Counter(dt.weekday() for dt in datetimes)
                weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                temporal_stats.update({
                    'most_active_hour': hour_counts.most_common(1)[0][0] if hour_counts else None,
                    'hourly_distribution': dict(hour_counts),
                    'most_active_weekday': weekday_names[weekday_counts.most_common(1)[0][0]] if weekday_counts else None,
                    'weekday_distribution': {weekday_names[k]: v for k, v in weekday_counts.items()}
                })
            
            results = {
                'total_texts': len(filtered_texts),
                'processed_docs': len(non_empty_docs),
                'sentiment_analysis': sentiment_analysis,
                'topic_analysis': topic_analysis,
                'word_frequency': {
                    'top_words': word_freq.most_common(30),
                    'total_unique_words': len(word_freq),
                    'total_words': sum(word_freq.values())
                },
                'user_activity': {
                    'total_users': len(users),
                    'user_activity': user_stats,
                    'summary': {
                        'total_posts_all_users': total_posts,
                        'avg_posts_per_user': total_posts / len(users) if users else 0,
                        'most_active_user': user_stats[0]['username'] if user_stats else None,
                        'avg_post_score': np.mean(post_scores) if post_scores else 0
                    }
                },
                'temporal_analysis': temporal_stats
            }
        
        # Save results
        results_file = os.path.join(output_dir, 'analysis.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Create visualizations
        plt.style.use('default')
        
        # Top users visualization
        user_activity_data = results.get('user_activity', {})
        user_activity = user_activity_data.get('user_activity', [])[:10]
        if user_activity:
            plt.figure(figsize=(10, 6))
            usernames = [user['username'][:15] + '...' if len(user['username']) > 15 else user['username'] for user in user_activity]
            post_counts = [user['total_posts'] for user in user_activity]
            
            plt.barh(range(len(usernames)), post_counts, color='lightblue')
            plt.yticks(range(len(usernames)), usernames)
            plt.xlabel('Total Posts')
            plt.title('Top 10 Most Active Users')
            plt.tight_layout()
            
            viz_file = os.path.join(output_dir, "top_users.png")
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Post distribution
        all_post_counts = [user['total_posts'] for user in user_activity_data.get('user_activity', [])]
        if all_post_counts:
            plt.figure(figsize=(10, 6))
            plt.hist(all_post_counts, bins=min(30, len(set(all_post_counts))), alpha=0.7, color='lightgreen')
            plt.xlabel('Posts per User')
            plt.ylabel('Number of Users')
            plt.title('Distribution of Posts per User')
            plt.tight_layout()
            
            viz_file = os.path.join(output_dir, "post_distribution.png")
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Word frequency
        if results.get('word_frequency', {}).get('top_words'):
            plt.figure(figsize=(10, 8))
            words, counts = zip(*results['word_frequency']['top_words'][:15])
            plt.barh(range(len(words)), counts, color='coral')
            plt.yticks(range(len(words)), words)
            plt.xlabel('Frequency')
            plt.title('Top 15 Most Common Words')
            plt.tight_layout()
            
            viz_file = os.path.join(output_dir, "word_frequency.png")
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Hourly activity
        if results.get('temporal_analysis', {}).get('hourly_distribution'):
            plt.figure(figsize=(12, 6))
            hours = list(range(24))
            counts = [results['temporal_analysis']['hourly_distribution'].get(h, 0) for h in hours]
            plt.bar(hours, counts, alpha=0.7, color='gold')
            plt.xlabel('Hour of Day')
            plt.ylabel('Number of Posts')
            plt.title('Posting Activity by Hour')
            plt.xticks(range(0, 24, 2))
            plt.tight_layout()
            
            viz_file = os.path.join(output_dir, "hourly_activity.png")
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Sentiment distribution
        sentiment_data = results.get('sentiment_analysis', {})
        if sentiment_data.get('sentiment_scores'):
            plt.figure(figsize=(10, 6))
            sentiment_scores = sentiment_data['sentiment_scores']
            plt.hist(sentiment_scores, bins=30, alpha=0.7, color='purple', edgecolor='black')
            plt.xlabel('Sentiment Score')
            plt.ylabel('Number of Texts')
            plt.title('Distribution of Sentiment Scores')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
            plt.axvline(x=sentiment_data.get('avg_sentiment', 0), color='green', linestyle='-', alpha=0.8, label=f'Average: {sentiment_data.get("avg_sentiment", 0):.3f}')
            plt.legend()
            plt.tight_layout()
            
            viz_file = os.path.join(output_dir, "sentiment_distribution.png")
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Sentiment categories pie chart
        if sentiment_data.get('total_analyzed', 0) > 0:
            plt.figure(figsize=(8, 8))
            categories = ['Positive', 'Neutral', 'Negative']
            ratios = [
                sentiment_data.get('positive_ratio', 0),
                sentiment_data.get('neutral_ratio', 0),
                sentiment_data.get('negative_ratio', 0)
            ]
            colors = ['lightgreen', 'lightgray', 'lightcoral']
            
            plt.pie(ratios, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Sentiment Categories Distribution')
            plt.axis('equal')
            plt.tight_layout()
            
            viz_file = os.path.join(output_dir, "sentiment_categories.png")
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Topics visualization
        if results.get('topic_analysis', {}).get('topics'):
            topics = results['topic_analysis']['topics']
            fig, axes = plt.subplots(len(topics), 1, figsize=(12, 4 * len(topics)))
            if len(topics) == 1:
                axes = [axes]
            
            for i, topic in enumerate(topics):
                words = topic['words'][:8]
                probs = topic['probabilities'][:8]
                
                axes[i].barh(range(len(words)), probs, color=plt.cm.Set3(i))
                axes[i].set_yticks(range(len(words)))
                axes[i].set_yticklabels(words)
                axes[i].set_xlabel('Probability')
                axes[i].set_title(f'Topic {topic["topic_id"]}: {", ".join(words[:3])}...')
                axes[i].invert_yaxis()
            
            plt.tight_layout()
            viz_file = os.path.join(output_dir, "topics.png")
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
        
    except FileNotFoundError as e:
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 