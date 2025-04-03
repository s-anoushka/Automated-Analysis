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
        df = pd.read_csv(file_path, encoding="latin1")
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
def generate_visualizations(df, output_dir="happiness"):
    """Generates and saves visualizations for the dataset."""
    print("Generating Visualizations...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Visualization 1: Distribution of Life Ladder scores
    if "Life Ladder" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["Life Ladder"].dropna(), bins=30, kde=True)
        plt.title("Distribution of Life Ladder Scores")
        plt.xlabel("Life Ladder Score")
        plt.ylabel("Frequency")
        plt.savefig(f"{output_dir}/life_ladder_distribution.png")
        plt.close()
        print("Saved: life_ladder_distribution.png")
    
    # Visualization 2: Life Ladder over the years for each country (if applicable)
    if "year" in df.columns and "Country name" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="year", y="Life Ladder", hue="Country name", marker="o")
        plt.title("Life Ladder Scores Over Years by Country")
        plt.xlabel("Year")
        plt.ylabel("Life Ladder Score")
        plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/life_ladder_trends.png")
        plt.close()
        print("Saved: life_ladder_trends.png")

# Function to generate story using Gemini
def generate_story_gemini(df, output_dir="happiness"):
    """Uses Gemini AI to generate a story based on data analysis."""
    print("Generating Story...")
    
    # Gather key statistics for the prompt
    num_countries = df["Country name"].nunique() if "Country name" in df.columns else "N/A"
    years = df["year"].unique() if "year" in df.columns else []
    year_range = f"{min(years)} - {max(years)}" if years.size > 0 else "N/A"
    avg_life_ladder = df["Life Ladder"].mean() if "Life Ladder" in df.columns else "N/A"
    total_missing = df.isnull().sum().sum()
    
    prompt = f"""
    You are an AI data analyst. Summarize the following happiness dataset analysis as a compelling story:
    - The dataset covers {num_countries} countries over the years {year_range}.
    - The average Life Ladder score is {avg_life_ladder:.2f} (if available).
    - The dataset includes metrics such as Log GDP per capita, Social support, Healthy life expectancy, Freedom to make life choices, Generosity, Perceptions of corruption, Positive affect, and Negative affect.
    - Total missing values across the dataset: {total_missing}.
    
    Provide insights on trends, interesting patterns, and possible interpretations of what drives happiness based on the data.
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        story = response.text
        
        # Save the story to a README file in the output directory
        readme_path = f"{output_dir}/README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# Happiness Dataset Analysis\n\n")
            f.write(story)
        
        print(f"Story saved to {readme_path}")
    except Exception as e:
        print(f"Error generating story: {e}")

# Main Execution
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python autolysis_happiness.py happiness.csv")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    df = load_data(dataset_path)
    
    basic_analysis(df)
    generate_visualizations(df, output_dir="happiness")
    generate_story_gemini(df, output_dir="happiness")