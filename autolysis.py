import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

# Load API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY is not set.")
    sys.exit(1)

genai.configure(api_key=api_key)

# Function to load dataset
def load_data(file_path):
    """Loads a CSV file and returns a Pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset '{file_path}' loaded successfully!")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Dataset Shape: {df.shape}\n")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

# Function for basic analysis
def basic_analysis(df):
    """Performs basic analysis on the dataset."""
    print("Basic Analysis:")
    print("===============")
    print(df.head(), "\n")
    print("Summary Statistics:\n", df.describe(include="all"), "\n")
    print("Missing Values Count:\n", df.isnull().sum(), "\n")

# Function for generating visualizations
def generate_visualizations(df, output_dir="goodreads"):
    """Generates and saves visualizations for the dataset."""
    print("Generating Visualizations...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if "average_rating" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["average_rating"].dropna(), bins=30, kde=True)
        plt.title("Distribution of Book Ratings")
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        plt.savefig(f"{output_dir}/rating_distribution.png")
        print("Saved: rating_distribution.png")

# Function to generate story using Gemini
def generate_story_gemini(df, output_dir="goodreads"):
    """Uses Gemini AI to generate a story based on data analysis."""
    print("Generating Story...")

    prompt = f"""
    You are an AI data analyst. Summarize the following book dataset analysis as a compelling story:
    - The dataset contains {df.shape[0]} books with {df.shape[1]} attributes.
    - The average book rating is {df['average_rating'].mean():.2f}.
    - The dataset contains books from {df['original_publication_year'].nunique()} different years.
    - Missing values: {df.isnull().sum().sum()} across multiple fields.
    - Top authors: {df['authors'].value_counts().head(3).to_dict()}.

    Provide insights on trends, interesting patterns, and potential book recommendations.
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        story = response.text

        readme_path = f"{output_dir}/README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# Goodreads Dataset Analysis\n\n")
            f.write(story)
        
        print(f"Story saved to {readme_path}")

    except Exception as e:
        print(f"Error generating story: {e}")

# Main Execution
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py dataset.csv")
        sys.exit(1)

    dataset_path = sys.argv[1]
    df = load_data(dataset_path)

    basic_analysis(df)
    generate_visualizations(df, "goodreads")
    generate_story_gemini(df, "goodreads")