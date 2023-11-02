from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Tokenisation:

    def generate_token(self, data1):
        """
        Tokenize the textual data and prepare it for deep learning models.
        :param data1: Input data with text content
        :return: Word index and tokenized sequences
        """
        # Define the maximum number of words to be used (most frequent)
        MAX_NB_WORDS = 2000

        # Define the maximum number of words in each content
        MAX_SEQUENCE_LENGTH = 600

        # Define the embedding dimension
        EMBEDDING_DIM = 100

        # Create a tokenizer with specific settings
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

        # Fit the tokenizer on the text content
        tokenizer.fit_on_texts(data1['content'].values)

        # Get the word index
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        # Tokenize the content
        X = tokenizer.texts_to_sequences(data1['content'].values)

        # Pad the sequences to a fixed length
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

        return word_index, X
