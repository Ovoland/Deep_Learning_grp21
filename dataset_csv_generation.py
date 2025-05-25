import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import csv

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



def split_original_dataset_equal_classes_with_generated(
    original_tsv, 
    implicit_generated_tsv, 
    explicit_generated_tsv, 
    train_output_tsv, 
    test_output_tsv,
    test_ratio=0.3
):

    # Read the original dataset and generated datasets with quoting to handle special characters
    original_df = pd.read_csv(original_tsv, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8', dtype=str, engine='python', on_bad_lines='skip')
    implicit_gen_df = pd.read_csv(implicit_generated_tsv, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8', dtype=str, engine='python', on_bad_lines='skip')
    explicit_gen_df = pd.read_csv(explicit_generated_tsv, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8', dtype=str, engine='python', on_bad_lines='skip')
    class_column = "class"
    classes = ["not_hate", "implicit_hate", "explicit_hate"]

    # Desired test set sizes for each class
    desired_test_sizes = {
        "not_hate": 3987,
        "implicit_hate": 2130,
        "explicit_hate": 327
    }

    test_dfs = []
    train_dfs = []

    for cls in classes:
        cls_df = original_df[original_df[class_column] == cls]
        # Determine the number of samples for the test set based on the desired size
        n_test_per_class = min(len(cls_df), desired_test_sizes[cls])
        # Shuffle and split
        cls_test = cls_df.sample(n=n_test_per_class, random_state=42)
        cls_train = cls_df.drop(cls_test.index)
        test_dfs.append(cls_test)
        train_dfs.append(cls_train)

    # Concatenate and shuffle
    test_df = pd.concat(test_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = pd.concat(train_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    # Add generated samples to training set
    # For implicit_hate: add 4334 samples
    implicit_gen_samples = implicit_gen_df.sample(n=4334, random_state=42)
    # For explicit_hate: add 8541 samples
    explicit_gen_samples = explicit_gen_df.sample(n=8541, random_state=42)

    # Append generated samples to training set
    train_df = pd.concat([train_df, implicit_gen_samples, explicit_gen_samples], ignore_index=True)

    # Shuffle and trim to 9304 samples per class
    train_final_dfs = []
    for cls in classes:
        cls_train = train_df[train_df[class_column] == cls]
        cls_train = cls_train.sample(n=9304, random_state=42) if len(cls_train) > 9304 else cls_train
        train_final_dfs.append(cls_train)
    train_final_df = pd.concat(train_final_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to TSV with quoting=csv.QUOTE_NONE to avoid extra quotes, and escapechar to handle special characters
    train_final_df.to_csv(train_output_tsv, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    test_df.to_csv(test_output_tsv, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')






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

    implicit_generated_tsv = "data/implicit-hate-corpus/generated_implicit_ONLY.tsv"
    explicit_generated_tsv = "data/implicit-hate-corpus/generated_explicit_ONLY.tsv"
    train_output_tsv = "data/implicit-hate-corpus/augmented_explicit_only/FINAL_TRAINING_SET.tsv"
    test_output_tsv = "data/implicit-hate-corpus/augmented_explicit_only/FINAL_TESTING_SET.tsv"
    # train_output_tsv = "test/FINAL_TRAINING_SET.tsv"
    # test_output_tsv = "test/FINAL_TESTING_SET.tsv"
    test_ratio = 0.3
    
    
    split_original_dataset_equal_classes_with_generated(
        original_tsv, 
        implicit_generated_tsv, 
        explicit_generated_tsv, 
        train_output_tsv, 
        test_output_tsv,
        test_ratio
    )
    
    
    class_column = "class"
    plot_class_distribution(train_output_tsv, class_column)
    plot_class_distribution(test_output_tsv, class_column)
