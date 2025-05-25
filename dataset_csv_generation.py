import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def remove_duplicates_and_save_unique(original_tsv, generated_tsv, output_tsv):
    # Read both TSV files
    original_df = pd.read_csv(original_tsv, sep='\t')
    generated_df = pd.read_csv(generated_tsv, sep='\t')

    # Find columns to compare (intersection of columns)
    compare_cols = list(set(original_df.columns) & set(generated_df.columns))

    # Remove duplicates: keep only rows in generated_df not in original_df
    unique_generated = generated_df.merge(
        original_df[compare_cols], 
        on=compare_cols, 
        how='left', 
        indicator=True
    )
    unique_generated = unique_generated[unique_generated['_merge'] == 'left_only']
    unique_generated = unique_generated[generated_df.columns]  # Keep original column order

    # Save to output TSV without extra quotes
    unique_generated.to_csv(output_tsv, sep='\t', index=False, quoting=3)  # quoting=3 means csv.QUOTE_NONE



def plot_class_distribution(tsv_path, class_column):
    df = pd.read_csv(tsv_path, sep='\t')
    all_classes = ["not_hate", "implicit_hate", "explicit_hate"]
    class_counts = df[class_column].value_counts().reindex(all_classes, fill_value=0)
    class_counts.plot(kind='bar')
    plt.title(f'Class Distribution in {tsv_path}')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()



if __name__ == "__main__":
    original_tsv = "data/implicit-hate-corpus/implicit_hate_v1_stg1_posts.tsv"
    generated_tsv = "data/implicit-hate-corpus/generated_implicit_ONLY.tsv"
    output_tsv =  "test_data/test_result/result_test.tsv"
    class_column = "class"

    remove_duplicates_and_save_unique(original_tsv, generated_tsv, output_tsv)
    plot_class_distribution(output_tsv, class_column)
