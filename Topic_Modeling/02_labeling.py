import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add root directory to sys.path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS

# --- CONFIGURATION ---
# CSV files mapping documents to dominant topics
INPUT_DOC_TOPICS_EN = os.path.join(DATA_RESULTS, "document_topics_en.csv")
INPUT_DOC_TOPICS_IT = os.path.join(DATA_RESULTS, "document_topics_it.csv")

# Output chart files
OUTPUT_CHART_EN = os.path.join(DATA_RESULTS, "topic_distribution_en.png")
OUTPUT_CHART_IT = os.path.join(DATA_RESULTS, "topic_distribution_it.png")

# --- TOPIC LABELS ---
# Single source of truth: Figures/fig_config.py (kept aligned with the
# canonical run's lda_topics_*.txt). Do not hardcode labels here.
from Figures.fig_config import TOPIC_LABELS_EN, TOPIC_LABELS_IT

# --- ANALYSIS & VISUALIZATION FUNCTION ---
def analyze_and_visualize_distribution(filepath, topic_labels, language, output_filename):
    """
    Loads data, computes topic distribution by source, and creates a stacked bar chart.
    """
    print(f"\n--- Starting Distribution Analysis for: {language.upper()} ---")
    
    try:
        df = pd.read_csv(filepath)
        # Identify the correct column name
        topic_col = [col for col in df.columns if 'topic_dominante' in col][0]
        print(f"  File loaded: {filepath}. Found topic column: '{topic_col}'")
    except FileNotFoundError:
        print(f"  ERROR: File '{filepath}' not found. Skipping.")
        return
    except IndexError:
        print(f"  ERROR: No 'topic_dominante' column found in '{filepath}'. Skipping.")
        return
    except Exception as e:
        print(f"  ERROR loading '{filepath}': {e}")
        return

    # Clean data and map topic IDs to readable labels
    df.dropna(subset=[topic_col], inplace=True)
    df['topic_label'] = df[topic_col].map(topic_labels)
    df.dropna(subset=['topic_label'], inplace=True) # Remove rows without a label
    
    # Calculate percentage distribution per source
    percentage_dist = pd.crosstab(df['fonte'], df['topic_label'], normalize='index') * 100
    
    print("\nTable: Percentage Distribution of Topics per Source (%)")
    rounded_dist = percentage_dist.round(2)
    print(rounded_dist)

    # Export to LaTeX for the Thesis
    latex_filename = output_filename.replace('.png', '.tex')
    try:
        with open(latex_filename, 'w', encoding='utf-8') as f:
            try:
                latex_str = rounded_dist.style.format(precision=2).to_latex(
                    caption=f"Percentage Distribution of Topics per Source ({language} Corpus)",
                    label=f"tab:topic_dist_{language.lower()}"
                )
            except AttributeError:
                latex_str = rounded_dist.to_latex(
                    caption=f"Percentage Distribution of Topics per Source ({language} Corpus)",
                    label=f"tab:topic_dist_{language.lower()}",
                    float_format="%.2f"
                )
            f.write(latex_str)
        print(f"LaTeX table successfully exported to '{latex_filename}'")
    except Exception as e_latex:
        print(f"Error during LaTeX export: {e_latex}")

    # Visualization
    print("\nCreating stacked bar chart...")
    try:
        sns.set_theme(style="whitegrid")
        
        percentage_dist.plot(
            kind='bar', 
            stacked=True, 
            figsize=(16, 10), 
            colormap='viridis'
        )

        plt.title(f'Topic Distribution per Source (Language: {language.upper()})', fontsize=18, pad=20)
        plt.ylabel('Percentage of Documents (%)', fontsize=12)
        plt.xlabel('Data Source', fontsize=12)
        plt.xticks(rotation=45, ha="right")
        
        # Legend outside the plot
        plt.legend(title='Topic', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Chart successfully saved to '{output_filename}'")
    except Exception as e:
        print(f"An error occurred during visualization: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # English Analysis
    analyze_and_visualize_distribution(
        filepath=INPUT_DOC_TOPICS_EN,
        topic_labels=TOPIC_LABELS_EN,
        language="English",
        output_filename=OUTPUT_CHART_EN
    )

    # Italian Analysis
    analyze_and_visualize_distribution(
        filepath=INPUT_DOC_TOPICS_IT,
        topic_labels=TOPIC_LABELS_IT,
        language="Italian",
        output_filename=OUTPUT_CHART_IT
    )