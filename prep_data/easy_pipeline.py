import argparse

import pandas as pd
import spacy
from huggingface_hub.utils import tqdm
from tabulate import tabulate
from langid.langid import LanguageIdentifier, model

nlp = spacy.load("en_core_web_sm")


def return_sentences_batch(texts):
    """
    Process multiple texts in batch mode using SpaCy.

    :param texts: List of strings (documents)
    :return: List of lists, where each sublist contains sentences from a document
    """
    sentence_lists = []

    # Process texts in batch using nlp.pipe
    for doc in tqdm(nlp.pipe(texts, disable=["ner", "tagger"]), total=len(texts)):  # Disable components not needed for speed
        sentences = [sent.text for sent in doc.sents]
        sentence_lists.append(sentences)

    return sentence_lists


def remove_non_english_sentences(sentences):
    """
    This method removes non-english sentences from a list of sentences. This can be used to remove either sentences that
    are not English or sentences that only contain math texts or formulas. When we do feature extraction, we want to
    ignore such snippets of text.
    :param sentences: a [list] of sentences , each sentence is a [string].
    :return: a filtered list of sentences that are in English.
    """
    lang_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    english_sentences = []
    for sentence in sentences:
        lang, prob = lang_identifier.classify(sentence)
        if lang == "en":
            english_sentences.append(sentence)
    # print the number of sentences removed
    # print("Number of non-english sentences removed:", len(sentences) - len(english_sentences))
    return english_sentences


def create_sentence_windows(sentences, window_size):
    """
    This method creates windows of sentences from a list of sentences. The windows are created by sliding a window of
    size window_size over the list of sentences.
    :param sentences: [list] list of sentences, each sentence is a string.
    :param window_size: [int] size of the window.
    :return: a [list of lists] where each list is a window of sentences.
    """
    # create windows of sentences
    windows = []
    for i in range(0, len(sentences), window_size):
        window = sentences[i:i + window_size]
        windows.append(window)
    return windows


def read_in_data(data_path):
    """
    This method reads in data from a parquet file and returns a pandas dataframe. The dataframe is filtered to remove
    rows where the full_text is empty or None.
    :param data_path: [string] path to the parquet file.
    :return: a [pandas dataframe] containing the data.
    """
    df = pd.read_parquet(data_path)
    # drop all rows with empty full text, or None
    df = df.dropna(subset=["full_text"])
    df = df[df.full_text != ""]
    return df


def process_texts(df, window_size):
    """
    This method processes the full_text column in a pandas dataframe. It creates windows of sentences from the full_text
    column and returns a new dataframe with the windows. Other columns in the dataframe are duplicated for each window.
    That way we can keep track of the original paper that the window came from, however the ID column is unique for each
    window.
    :param window_size: [int] size of the window to create.
    :param df: [pandas dataframe] dataframe containing the data.
    :return: a [pandas dataframe] containing the windows of sentences.
    """
    new_dfs = []

    # Extract the full text column as a list for batch processing
    papers = df["full_text"].tolist()

    # Process all papers in batch
    sentences_list = return_sentences_batch(papers)  # Process sentences in batch
    english_sentences_list = [remove_non_english_sentences(sentences) for sentences in sentences_list]
    windows_list = [create_sentence_windows(sentences, window_size) for sentences in english_sentences_list]

    # Precompute the rows using list comprehensions
    new_dfs = [
        {**row._asdict(), "text": " ".join(window)}
        for row, windows in tqdm(zip(df.itertuples(index=False), windows_list), total=len(df))
        for window in windows
    ]

    # Create a DataFrame
    full_df = pd.DataFrame(new_dfs)
    full_df["ID"] = range(len(full_df))

    return full_df


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data_path", type=str, required=True, help="Path to the parquet file")
    argument_parser.add_argument("--window_size", type=int, default=5, help="Size of the window")
    argument_parser.add_argument("--output_path", type=str, default="windows.parquet",
                                 help="Output path for the windows")
    args = argument_parser.parse_args()
    process_texts(read_in_data(args.data_path), args.window_size).to_parquet(args.output_path)
