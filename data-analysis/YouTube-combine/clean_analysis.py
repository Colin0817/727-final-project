import pandas as pd
import re
import numpy
import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from bertopic import BERTopic

# Suppose the dataset is in CSV format and contains two columns: 'comment' and 'label'.
data = pd.read_csv("youtube_related.csv")

# Assuming the first 49 comments as training data and the remaining as test data
train_data = data.iloc[:41]
test_data = data.iloc[41:49]


# Functions to clean up comment text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    return text


# Cleaning Reviews
data["cleaned_comment"] = data["comment"].apply(clean_text)


def identify_objects(comment, objects):
    mentioned_objects = [obj for obj in objects if obj.lower() in comment]
    return mentioned_objects


objects_list = [
    "ChatGPT",
    "Claude",
    "gpt",
    "sunnet",
    "open ai",
    "openai",
    "claudeai",
    "both",
    "none",
    "chatgtp",
]
data["type"] = data["cleaned_comment"].apply(
    lambda x: identify_objects(x, objects_list)
)


# Define the division logic first
def classify_type(objects_in_comment):
    chatgpt_keywords = ["ChatGPT", "gpt", "open ai", "openai", "chatgtp"]
    claude_keywords = ["Claude", "sonnet", "claudeai"]

    # Determining if a comment contains both chatgpt and claude keywords
    if any(obj in chatgpt_keywords for obj in objects_in_comment) and any(
        obj in claude_keywords for obj in objects_in_comment
    ):
        return "both"
    elif any(obj in ["both", "none"] for obj in objects_in_comment):
        return "both"
    elif any(obj in chatgpt_keywords for obj in objects_in_comment):
        return "chatgpt"
    elif any(obj in claude_keywords for obj in objects_in_comment):
        return "claude"
    else:
        return "none"


# Applying classification functions
data["classification"] = data["type"].apply(classify_type)

# Three types of data were extracted separately
chatgpt_data = data[data["classification"] == "chatgpt"]
claude_data = data[data["classification"] == "claude"]
both_data = data[data["classification"] == "both"]
print(chatgpt_data)

# Loading Sentiment Analysis Models
sentiment_analyzer = pipeline("sentiment-analysis")


def analyze_sentiment(comment):
    result = sentiment_analyzer(comment)
    label = result[0]["label"]
    score = result[0]["score"]
    return label, score

# Loading a Sentiment Analysis Model with Triple Categorization Support
sentiment_analyzers = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

# Analyzing the Sentiment of a Single Comment
def analyze_sentiments(comment):
    result = sentiment_analyzers(comment)  # [{"label": "NEUTRAL", "score": 0.85}]
    label = result[0]["label"]
    score = result[0]["score"]
    return label, score

id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

def map_label(label):
    label_id = int(label.split("_")[-1])  # Extract the numbered part, e.g. “LABEL_1” -> 1
    return id2label[label_id]  # Mapping to Specific Emotions

# Processing Sentiment Analysis Results from DataFrame
def process_sentiment(df):
    df["result"] = df["cleaned_comment"].apply(lambda x: analyze_sentiments(x))
    df["raw_label"] = df["result"].apply(lambda x: x[0])
    df["trend"] = df["raw_label"].apply(map_label)
    df["score"] = df["result"].apply(lambda x: x[1])
    return df[["comment","cleaned_comment", "trend", "score"]]

# Processing sentiment analysis on chatgpt_data and claude_data separately
chatgpt_data = process_sentiment(chatgpt_data)
claude_data = process_sentiment(claude_data)

# Loading the English Language Model
nlp = spacy.load("en_core_web_sm")

# Common Cause Category Words
reason_keywords = {
    "price_keywords": [
        "price",
        "costs",
        "expensive",
        "cheap",
        "value",
        "affordable",
        "worth",
        "free",
        "subscription",
        "pay",
        "costly",
    ],
    "performance_keywords": [
        "performance",
        "speed",
        "efficient",
        "quality",
        "accuracy",
        "reliable",
        "lag",
        "fast",
        "smooth",
        "slow",
        "output",
        "number",
        "load",
        "uplaod"
    ],
    "support_keywords": [
        "support",
        "customer service",
        "help",
        "responsive",
        "team",
        "issues",
        "bug",
        "contact",
        "solve",
        "response time",
        "feedback",
        "documentation",
        "wait",
        "secured",
        "conversation",
    ],
    "usability_keywords": [
        "easy",
        "speak",
        "talk",
        "iterations",
        "user friendly",
        "information",
        "code",
        "ppt",
        "math",
        "scriptwriting",
        "coding",
        "create",
        "programming",
        "text",
        "remember",
        "voice",
        "talking",
        "writing",
    ],
}


def extract_reasons(comment):
    doc = nlp(comment)
    reasons = []
    for token in doc:
        for category, keywords in reason_keywords.items():
            if token.text.lower() in keywords:
                reasons.append(category)
    return list(set(reasons))


# Add Cause Analysis to DataFrame
chatgpt_data["reasons"] = chatgpt_data["comment"].apply(lambda x: extract_reasons(x))
claude_data["reasons"] = claude_data["comment"].apply(lambda x: extract_reasons(x))
chatgpt_data.to_csv("chatgpt_data.csv", index=False)
claude_data.to_csv("claude_data.csv", index=False)

chatgpt_keywords = ["ChatGPT", "gpt", "open ai", "openai", "chatgtp"]
claude_keywords = ["Claude", "sonnet", "claudeai"]

# Define function to extract the part of the comment about ChatGPT and Claude
def split_comment_by_object(comment, chatgpt_keywords, claude_keywords):
    chatgpt_mentions = [
        word for word in chatgpt_keywords if word.lower() in comment.lower()
    ]
    claude_mentions = [
        word for word in claude_keywords if word.lower() in comment.lower()
    ]

    chatgpt_part = ""
    claude_part = ""

    # If the comment mentions ChatGPT-related keywords, consider that part of the comment to be ChatGPT's remarks
    if chatgpt_mentions:
        chatgpt_part = " ".join(
            [
                word
                for word in comment.split()
                if any(k.lower() in word.lower() for k in chatgpt_keywords)
            ]
        )

    # If the comment mentions Claude-related keywords, consider this part to be Claude's remarks
    if claude_mentions:
        claude_part = " ".join(
            [
                word
                for word in comment.split()
                if any(k.lower() in word.lower() for k in claude_keywords)
            ]
        )

    return chatgpt_part, claude_part


both_data["chatgpt_part"], both_data["claude_part"] = zip(
    *both_data["comment"].apply(
        lambda x: split_comment_by_object(x, chatgpt_keywords, claude_keywords)
    )
)


def analyze_parts_sentiment(chatgpt_part, claude_part):
    # Sentiment analysis of the section
    chatgpt_sentiment = analyze_sentiment(chatgpt_part) if chatgpt_part else ("neutral", 0.0)
    claude_sentiment = analyze_sentiment(claude_part) if claude_part else ("neutral", 0.0)

    # Extracting sentiment labels and scores
    chatgpt_label, chatgpt_score = chatgpt_sentiment
    claude_label, claude_score = claude_sentiment

    # Determining the bias of comments
    if chatgpt_label == "POSITIVE" and claude_label == "NEGATIVE":
        overall_bias = "chatgpt"
    elif claude_label == "POSITIVE" and chatgpt_label == "NEGATIVE":
        overall_bias = "claude"
    elif chatgpt_label == "POSITIVE" and claude_label == "POSITIVE":
        overall_bias = "chatgpt" if chatgpt_score > claude_score else "claude"
    elif chatgpt_label == "NEGATIVE" and claude_label == "NEGATIVE":
        overall_bias = "chatgpt" if chatgpt_score > claude_score else "claude"
    else:
        overall_bias = "neutral"

    # Return to Sentiment Score and Bias
    return chatgpt_score, claude_score, overall_bias

# Apply sentiment analysis and get scores and biases
both_data["chatgpt_score"], both_data["claude_score"], both_data["overall_bias"] = zip(
    *both_data.apply(
        lambda row: analyze_parts_sentiment(row["chatgpt_part"], row["claude_part"]),
        axis=1,
    )
)


# TF-IDF vectorization of chatgpt_part
chatgpt_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
X_chatgpt = chatgpt_vectorizer.fit_transform(both_data["chatgpt_part"].fillna(''))

# TF-IDF vectorization of claude_part
claude_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
X_claude = claude_vectorizer.fit_transform(both_data["claude_part"].fillna(''))

# LDA models were constructed
lda_chatgpt_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_chatgpt_model.fit(X_chatgpt)

lda_claude_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_claude_model.fit(X_claude)

# Print the subject keywords for each section
def print_top_words(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

# Assign the ChatGPT portion and Claude portion of each comment to the most likely topic
chatgpt_topic_assignments = lda_chatgpt_model.transform(X_chatgpt)
both_data['chatgpt_lda_topic'] = chatgpt_topic_assignments.argmax(axis=1)

claude_topic_assignments = lda_claude_model.transform(X_claude)
both_data['claude_lda_topic'] = claude_topic_assignments.argmax(axis=1)

# Save
both_data.to_csv("both_data_with_lda.csv", index=False)

