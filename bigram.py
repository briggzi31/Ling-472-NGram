# !/usr/bin/python3
import math
import pandas as pd
import counts

UNK = '<UNK>'

def _print_sorted(s):
    for line in s:
        bigram = line[0] + ' ' + str(line[1])
        print(bigram)



class LanguageModel:

    def __init__(self):
        self.unked_sentences = []
        self.bigram_probs = {}
        self.unigram_count = {}
        self.bigram_count = {}
        self.sent_probs = {}
        self.V_num = 0
        self.N_tot = 0

    def _calculate_bigram_probabilities(self):
        '''

        :param sentences:
        :param word_count:
        :param bigram_count:

        :return:
        '''

        # don't include <sos> in our vocabulary size
        self.V_num = len(self.unigram_count) - 1

        # calculate bigram probability for each (w_i, w_i-1) pair
        for sentence in self.unked_sentences:
            for i in range(2, len(sentence)):
                word_i_1 = sentence[i-1]
                word_i = sentence[i]

                cur_pair = word_i_1 + ' ' + word_i

                count_w_w_1 = 0
                if cur_pair in self.bigram_count:
                    count_w_w_1 = self.bigram_count[cur_pair]

                count_w = 0
                if word_i_1 in self.unigram_count:
                    count_w = self.unigram_count[word_i_1]

                if cur_pair not in self.bigram_probs:
                    bigram_prob = (count_w_w_1 + 1)/(count_w + self.V_num)
                    logged_bigram_prob = math.log2(bigram_prob)
                    self.bigram_probs[cur_pair] = round(logged_bigram_prob, 3)

    def _print_sent_prob(self):
        for sent in self.sent_probs:
            print(sent, self.sent_probs[sent])

    def _calculate_sentences_prob(self, sentences, unked_sentences):
        for i in range(len(sentences)):
            regular_sent = sentences[i]
            unk_sent = unked_sentences[i]
            sent_prob = 0
            sent = ''

            for j in range(2, len(unk_sent)):
                word_j_1 = unk_sent[j-1]
                word_j = unk_sent[j]

                cur_bigram = word_j_1 + ' ' + word_j

                sent += ' ' + regular_sent[j]

                if cur_bigram in self.bigram_probs:
                    cur_bigram_prob = self.bigram_probs[cur_bigram]
                else:
                    cur_bigram_prob = 1/((self.unigram_count[word_j_1]) + self.V_num)
                    cur_bigram_prob = round(math.log2(cur_bigram_prob), 3)

                if cur_bigram_prob == 0:
                    raise ValueError('cur_bigram_prob is 0 when it should be a non-zero')

                sent_prob += cur_bigram_prob

            sent = sent[:-6]
            self.sent_probs[sent] = round(sent_prob, 3)

    def _calculate_perplexity(self):
        perplexity = 0
        for sentence, prob in self.sent_probs.items():
            perplexity += prob

        return round(2 ** ((perplexity * -1)/self.N_tot), 3)



    def train(self, train_corpus):

        # clean the text
        sentences = counts.clean_file_contents(train_corpus)

        # get unigram counts
        self.unigram_count, self.N_tot = counts.count_unigrams(sentences)
        self.N_tot -= len(sentences)
        self.unigram_count['<sos>'] -= len(sentences)

        # print('original: ')
        # for i in range(10):
        #     print(sentences[i])

        self.unked_sentences = counts.unk_sentences(sentences, self.unigram_count)

        # print()
        # print('unked: ')
        # for i in range(10):
        #     print(self.unked_sentences[i])

        # get bigram counts
        self.bigram_count = counts.count_bigrams(self.unked_sentences)

        # print()
        # print('bigram count:')
        # print(list(self.bigram_count.items())[:4])

        # calculate logged probabilities of each bigram
        self._calculate_bigram_probabilities()

        # print()
        # print('bigram probs: ')
        # print(list(self.bigram_probs.items())[:4])

        # sort and print the probabilities
        sorted_prob = sorted(self.bigram_probs.items(), key=lambda x: (-x[1], x[0]))

        columns = ['Bigram', 'Probability']
        bigram_probs_df = pd.DataFrame(data=sorted_prob, columns=columns)
        bigram_probs_df = bigram_probs_df.set_index('Bigram')

        print(bigram_probs_df)

        corpus = train_corpus.replace(".txt", "")
        corpus = corpus.replace("data/", "")
        save_as_csv = "bigram_train_for_" + str(corpus) + ".csv"
        #bigram_probs_df.to_csv('bigram_probabilities.csv')
        bigram_probs_df.to_csv(save_as_csv)
        # _print_sorted(sorted_prob)



    def score(self, test_corpus):
        # clean the text
        sentences = counts.clean_file_contents(test_corpus)


        # return unked copy of sentences
        unked_sentences = counts.unk_sentences(sentences, self.unigram_count)

        # calcuate probabilities of the sentences
        self._calculate_sentences_prob(sentences, unked_sentences)

        # print(self.sent_probs)

        columns = ['Sentence', 'Probability']
        sentences_probs_df = pd.DataFrame(data=list(self.sent_probs.items()), columns=columns)
        sentences_probs_df = sentences_probs_df.set_index('Sentence')

        print(sentences_probs_df)
        corpus = test_corpus.replace(".txt", "")
        corpus = corpus.replace("data/", "")
        save_as_csv = "bigram_test_for_" + str(corpus) + ".csv"
        #sentences_probs_df.to_csv('bigram_sentence_probabilities.csv')
        sentences_probs_df.to_csv(save_as_csv)

        # self._print_sent_prob()
        sum_prob = sentences_probs_df["Probability"].astype("float").sum()
        perplexity = round(2 ** (-(1 / self.N_tot) * sum_prob), 3)
        # perplexity = self._calculate_perplexity()
        print()
        print('Perplexity =', perplexity)
