from model.config import Config
from model.DataSet import DataSet, build_vocabs, UNK, NUM, \
    build_glove_vocab, write_vocab, load_vocab, \
    export_trimmed_glove_vectors, process_vocab


def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config and processing of words
    config = Config(load=False)

    # Generators
    dev = DataSet(config.filename_dev)
    test = DataSet(config.filename_test)
    train = DataSet(config.filename_train)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = build_vocabs([train, dev, test])
    vocab_glove = build_glove_vocab(config.filename_glove)

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Trim GloVe Vectors
    vocab = load_vocab(config.filename_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                 config.filename_trimmed, config.dim_word)


if __name__ == "__main__":
    main()
