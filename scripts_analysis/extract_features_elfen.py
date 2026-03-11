import argparse

import polars as pl
from elfen.extractor import Extractor

if __name__ == '__main__':
    argumentparser = argparse.ArgumentParser()
    # add data path
    argumentparser.add_argument('--data_path', required=True, help="Path to the dataset")
    # add the column name of text
    argumentparser.add_argument('--text_column', required=True, help="Name of the text column")
    # add the output path
    argumentparser.add_argument('--output_path', required=True, help="Path to save the extracted features")
    # add data tpye (options are 'csv' or 'parquet')
    argumentparser.add_argument('--data_type', required=True, help="Type of the dataset")
    args = argumentparser.parse_args()

    # read the data from the path, depending on the data type
    if args.data_type == 'csv':
        df = pl.read_csv(args.data_path, separator='\t')
    elif args.data_type == 'parquet':
        df = pl.read_parquet(args.data_path)
    else:
        print("Data type not supported")
        exit()
    print("columns in the dataframe: ", df.columns)
    print("processing a dataframe of size %d" % len(df))
    extractor = Extractor(data=df,
                          language="en",
                          model="en_core_web_sm",
                          text_column=args.text_column)
    # save as parquet to the output path
    extractor.extract_features()
    feature_df = extractor.data
    # drop all columns that are Object type
    feature_df = feature_df.select(pl.exclude(pl.Object))
    print(feature_df.head())
    print("type of the feature dataframe: ", type(feature_df))
    print("shape of the feature dataframe: ", feature_df.shape)
    # save polars dataframe to the output path but convert it into parquet
    feature_df.write_parquet(args.output_path)