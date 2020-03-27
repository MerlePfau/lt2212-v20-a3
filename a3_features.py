import argparse
import numpy as np
import pandas as pd
from glob import glob
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.
    folders = glob("{}/*".format(args.inputdir))
    all_files = []
    author_list = []
    index_overview = {}
    list_of_sample_words = []
    i = 0
    # reads in documents and creates wordlists per documents + all words + word index overview + list of all authors
    for author in folders:
        files = glob("{}/*".format(author))
        for filename in files:
            author_list.append(author)
            filecontent = ""
            with open(filename, "r") as thefile:
                for line in thefile:
                    filecontent += line
            tokens = filecontent.split()
            cleaned_words = [w.lower() for w in tokens if w and w.isalpha()]
            for word in cleaned_words:
                if word not in index_overview:
                    index_overview[word] = i
                    i += 1
            all_files.append(cleaned_words)
    all_words = index_overview.keys()


    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    columns = len(all_words)
    rows = len(all_files)
    # initiate numpy array
    features = np.zeros((rows, columns))
    sample_number = 0
    # vectorize word counts per document
    for wordlist in all_files:
        bag_vector = np.zeros(len(all_words))
        for word in wordlist:
            index = index_overview[word]
            bag_vector[index] += 1
        features[sample_number, :] = bag_vector
        sample_number += 1
    X = features

    # dimensionality reduction and splitting train and test data
    svd = TruncatedSVD(n_components=args.dims, n_iter=7, random_state=42)
    X_dr = svd.fit_transform(X)
    y = author_list
    test = args.testsize/100
    X_train, X_test, y_train, y_test = train_test_split(X_dr, y, test_size=test, random_state=42)

    # build table
    table = pd.DataFrame()

    y_train = pd.DataFrame(y_train)
    y_train['train_test'] = 'train'
    y_test = pd.DataFrame(y_test)
    y_test['train_test'] = 'test'
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    table_1 = pd.concat([y_train, y_test])
    table_1['ID'] = np.arange(len(table_1))
    table_2 = pd.concat([X_train, X_test])

    table = pd.concat([table_1, table_2], axis=1)

    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.
    with open(args.outputfile, "w+") as thefile:
        table.to_csv(thefile, index=False)

    print("Done!")

