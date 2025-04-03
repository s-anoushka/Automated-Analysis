import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

# Load API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY is not set. Story generation will be skipped.")
    api_key = None
else:
    genai.configure(api_key=api_key)

# Function to load dataset
def load_data(file_path):
    """Loads a CSV file and returns a Pandas DataFrame."""
    try:
        df = pd.read_csv(file_path, encoding="latin1", on_bad_lines='skip')  # Handle encoding and malformed data
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
def generate_visualizations(df, output_dir="media"):
    """Generates and saves visualizations for the dataset."""
    print("Generating Visualizations...")
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

    if "overall" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["overall"].dropna(), bins=30, kde=True)
        plt.title("Distribution of Overall Ratings")
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        plt.savefig(f"{output_dir}/rating_distribution.png")
        print("Saved: rating_distribution.png")

# Function to generate story using Gemini
def generate_story_gemini(df, output_dir="media"):
    """Uses Gemini AI to generate a story based on data analysis."""
    if not api_key:
        print("Skipping story generation due to missing API key.")
        return
    
    print("Generating Story...")
    os.makedirs(output_dir, exist_ok=True)
    
    avg_rating = df["overall"].mean() if "overall" in df.columns else "N/A"
    unique_types = df["type"].nunique() if "type" in df.columns else "Unknown"
    top_languages = df["language"].value_counts().head(3).to_dict() if "language" in df.columns else {}
    missing_values = df.isnull().sum().sum()

    prompt = f"""
    You are an AI data analyst. Summarize the following media dataset analysis as a compelling story:
    - The dataset contains {df.shape[0]} media items with {df.shape[1]} attributes.
    - The average overall rating is {avg_rating}.
    - The dataset includes {unique_types} different media types.
    - Missing values: {missing_values} across multiple fields.
    - Most common languages: {top_languages}.

    Provide insights on trends, interesting patterns, and potential recommendations.
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        story = response.text if hasattr(response, "text") else "No response generated."
        
        readme_path = f"{output_dir}/README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# Media Dataset Analysis\n\n")
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
    generate_visualizations(df, "media")
    generate_story_gemini(df, "media")