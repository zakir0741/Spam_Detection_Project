import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

if __name__ == "__main__":
    df = load_data()
    df.to_csv("cleaned_spam.csv", index=False)
    print("Dataset cleaned and saved.")
