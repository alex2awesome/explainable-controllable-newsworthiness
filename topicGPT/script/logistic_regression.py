import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

# Helper method to get concepts
# Also removes any rows that couldn't be assinged a concept for some reason
def get_concepts(df):
    drop_rows = []
    for index, row in df.iterrows():
        text = row['responses']
        concept = re.search("\[1\] (.*?):", text)
        if concept:
            concept = concept.group(1)
        else:
            drop_rows.append(index)
        df.at[index, 'concepts'] = concept
    return df.copy().drop(drop_rows)

def get_sample_assignment():
    # Get topics with texts and labels
    assignment_sample_df = pd.read_json('./data/output/sf_policies/sf_level1_500_assignment.jsonl', lines=True)

    # Copy responses column to new column called concepts
    assignment_sample_df['concepts'] = assignment_sample_df['responses']
    concepts_sample_df = get_concepts(assignment_sample_df)
    return concepts_sample_df


def calculate_logistic_regression(concepts_sample_df):
    vectorizer = TfidfVectorizer()
    y_pred = model.predict(X_test)
    X = vectorizer.fit_transform(concepts_sample_df['concepts'])
    y = concepts_sample_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores = pd.Series(cross_val_score(model, X_train, y_train, cv=150))
    print(scores.mean())
    print("Accuracy:", score)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    concepts_sample_df = get_sample_assignment()
    calculate_logistic_regression(concepts_sample_df)