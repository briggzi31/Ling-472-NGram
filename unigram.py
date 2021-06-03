# !/usr/bin/python3

import pandas as pd
import math
import counts

# TODO: Implement a Laplace-smoothed unigram model :)
class LanguageModel:

    def __init__(self):
        self.word_count = {}
        self.sent_prob = {}
        self.df = pd.DataFrame()
        self.total_count = 0

    def train(self, train_corpus):

        # Read in and cleans the training corpus
        sentences = counts.clean_file_contents(train_corpus)

        # Keeps a dictionary of the vocabulary and the counts of individual words and the total counts in the corpus.
        self.word_count, self.total_count = counts.count_unigrams(sentences)

        # Removing the start and end token from the vocabulary and from the total counts 
        self.word_count.pop("<sos>")
        self.word_count.pop("<eos>")
        self.total_count -= 3 * len(sentences)

        # Calculate log2 Probability of individual words with Laplace Smoothing
        prob_df = [] 
        for word in self.word_count:
            word_prob = (self.word_count[word] + 1) / (self.total_count + len(self.word_count))
            log2_prob = math.log2(word_prob)
            prob_df.append([word, self.word_count[word], word_prob, round(log2_prob, 3)])


        column_names = ["word", "Count", "Probability", "log2 Probability"]

        # Sample way of creating a dataframe. This assumes that "data" is a LIST OF LISTS.
        self.df = pd.DataFrame(data=prob_df, columns=column_names)
        # Sorting first by log2 Probability in descending order and then by alphabetically
        self.df.sort_values(by=["log2 Probability", "word"], ascending=[False, True], inplace=True)

        unigram_prob = self.df[['word', 'log2 Probability']]
        self.df = unigram_prob.set_index('word')
        print(self.df)

        # Saving to a file:
        corpus = train_corpus.replace(".txt", "")
        corpus = corpus.replace("data/", "")
        save_as_csv = "unigram_train_for_" + str(corpus) + ".csv"
        self.df.to_csv(save_as_csv)

    def score(self, test_corpus):

        # Read in and cleans the test/dev .txt files
        sentences = counts.clean_file_contents(test_corpus)

        # Create UNK tokens for OOV words
        unk = counts.unk_sentences(sentences, self.word_count)

        # Deletes first two unks, which were start tokens, and last unk, which was an end token.
        # Calculate log2 probability for each sentence.
        # prob = []

        for i in range(len(unk)):
            unk_sent = unk[i]
            reg_sent = sentences[i]

            del unk_sent[0:2]
            del reg_sent[0:2]
            unk_sent.pop()
            reg_sent.pop()

            sentence_prob = 0
            sent = ''
            for j in range(len(unk_sent)):
                sent += reg_sent[j] + ' '
                current_word = self.df.loc[unk_sent[j], 'log2 Probability']
                # current_word = self.df.loc[self.df["word"] == word]
                # sentence_prob += float(current_word["log2 Probability"])
                sentence_prob += current_word

            sent = sent.strip()
            self.sent_prob[sent] = sentence_prob

            # prob.append([sentence, round(sentence_prob, 3)])

        column_names = ["Sentence", "log2 Probability"]
        test_prob = pd.DataFrame(data=list(self.sent_prob.items()), columns=column_names)
        # Sample way of creating a dataframe. This assumes that "data" is a LIST OF LISTS.
        # test_prob = pd.DataFrame(data=prob, columns=column_names)
        test_prob = test_prob.set_index('Sentence')

        print(test_prob)
        
        # Saving to a file:
        corpus = test_corpus.replace(".txt", "")
        corpus = corpus.replace("data/", "")
        save_as_csv = "unigram_test_for_" + str(corpus) + ".csv"
        test_prob.to_csv(save_as_csv)

        # Calculate perplexity
        sum_prob = test_prob["log2 Probability"].astype("float").sum()
        perplexity = round(2 ** (-(1 / self.total_count) * sum_prob), 3)
        print()
        print('Perplexity: ' + str(perplexity))
