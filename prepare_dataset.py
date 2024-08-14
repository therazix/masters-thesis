import math
import numpy as np
import pandas as pd
from pathlib import Path
from pandarallel import pandarallel

pandarallel.initialize()
pd.options.mode.chained_assignment = None

CHARS = 'aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž' + '0123456789' + '.!?"„“,:-();/&'


def extract_style(text: str):
    words = text.split()

    len_text = len(text)
    len_words = len(words)
    avg_len = np.mean([len(w) for w in words])
    num_short_w = len([w for w in words if len(w) < 3])
    per_digit = sum(c.isdigit() for c in text) / len_text
    per_cap = sum(1 for c in text if c.isupper()) / len_text
    richness = len(list(set(words))) / len_words
    frequencies = {char: sum(1 for c in text if c.lower() == char) / len_text for char in CHARS}

    return pd.Series(
        [avg_len, len_text, len_words, num_short_w,
         per_digit, per_cap, *frequencies.values(), richness]
    )


def insert_features(df: pd.DataFrame):
    df[['avg_len', 'len_text', 'len_words', 'num_short_w',
      'per_digit', 'per_cap', 'f_0', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6',
      'f_7', 'f_8', 'f_9', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15',
      'f_16', 'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24',
      'f_25', 'f_26', 'f_27', 'f_28', 'f_29', 'f_30', 'f_31', 'f_32', 'f_33',
      'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39', 'f_40', 'f_41', 'f_42',
      'f_43', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 'f_50', 'f_51',
      'f_52', 'f_53', 'f_54', 'f_55', 'f_56', 'f_57', 'f_58', 'f_59', 'f_60',
      'f_61', 'f_62', 'f_63', 'f_64', 'richness']] = df['text'].parallel_apply(lambda x: extract_style(x))  # noqa


def create(input_file: str | Path,
           output_dir: str | Path,
           num_of_authors: int,
           add_out_of_class: bool = False,
           add_text_features: bool = False,
           train_test_split: float = 0.75):
    input_file = Path(input_file) if isinstance(input_file, str) else input_file
    output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir

    df = pd.read_csv(input_file)
    df = df[['author', 'text']]
    df = df.rename(columns={'author': 'label'})
    df = df.drop_duplicates()

    # Get all author names sorted by the number of texts
    all_authors = df['label'].value_counts().index.tolist()

    selected_authors = all_authors[:num_of_authors]
    non_selected_authors = all_authors[num_of_authors:]

    # Initialize the train and test dataframes
    train = pd.DataFrame(columns=['label', 'text'])
    test = pd.DataFrame(columns=['label', 'text'])

    for author in selected_authors:
        author_df = df[df['label'] == author]  # Get all texts of the author
        author_df = author_df.sample(frac=1).reset_index(drop=True)  # Shuffle author's texts
        split_index = int(train_test_split * len(author_df))
        train_df = author_df.iloc[:split_index]
        test_df = author_df.iloc[split_index:]
        # Add author's texts to the train and test dataframes
        train = pd.concat([train, train_df], ignore_index=True)
        test = pd.concat([test, test_df], ignore_index=True)

    # Rename all authors
    replace_dict = {author: str(i) for i, author in enumerate(selected_authors)}
    train['label'] = train['label'].replace(replace_dict)
    test['label'] = test['label'].replace(replace_dict)

    if add_out_of_class:
        out_of_class_df = pd.DataFrame(columns=['label', 'text'])
        last_author_text_count = len(df[df['label'] == selected_authors[-1]])
        # math.ceil is used to ensure that the number of texts per author is at least 1
        texts_per_author = math.ceil(last_author_text_count / len(non_selected_authors))
        for author in non_selected_authors:
            author_df = df[df['label'] == author]
            author_df = author_df.sample(frac=1).reset_index(drop=True)
            author_df = author_df.iloc[:texts_per_author]
            out_of_class_df = pd.concat([out_of_class_df, author_df], ignore_index=True)

        out_of_class_df['label'] = str(len(selected_authors))
        out_of_class_df = out_of_class_df.sample(frac=1).reset_index(drop=True)
        split_index = int(train_test_split * len(out_of_class_df))
        train_out_of_class = out_of_class_df.iloc[:split_index]
        test_out_of_class = out_of_class_df.iloc[split_index:]
        train = pd.concat([train, train_out_of_class], ignore_index=True)
        test = pd.concat([test, test_out_of_class], ignore_index=True)

    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    if add_text_features:
        insert_features(train)
        insert_features(test)

    output_dir.mkdir(parents=True, exist_ok=True)

    ooc_suffix = 'withOOC' if add_out_of_class else 'withoutOOC'
    train.to_csv(output_dir / f'train_top{num_of_authors}_{ooc_suffix}.csv', index=False)
    test.to_csv(output_dir / f'test_top{num_of_authors}_{ooc_suffix}.csv', index=False)


def main():
    for ooc in [False, True]:
        for author_count in [5, 10, 25, 50]:
            create('datasets/csfd_new.csv',
                   f'datasets/csfd/top{author_count}',
                   author_count,
                   add_out_of_class=ooc,
                   add_text_features=True)


if __name__ == '__main__':
    main()
