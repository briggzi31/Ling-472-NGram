# !/usr/bin/python3
import counts
import math
import pandas as pd

# TODO: Implement a Laplace-smoothed trigram model :)
class LanguageModel:

    def __init__(self):
        self.unked_sentences = []
        self.unigram_count = {}
        self.bigram_count = {}
        self.trigram_count = {}
        self.trigram_probs = {}
        self.sent_probs = {}
        self.V_num = 0

    def _calculate_trigram_probabilities(self):
        # trigram_df_maker = []
        # trigram_vocab = []

        for sentence in self.unked_sentences:
            for i in range(2, len(sentence)):
                word_i_2 = sentence[i - 2]
                word_i_1 = sentence[i - 1]
                word_i = sentence[i]

                cur_trigram = word_i_2 + ' ' + word_i_1 + ' ' + word_i
                bigram_pair = word_i_2 + ' ' + word_i_1

                count_w_w_2 = 0
                if cur_trigram in self.trigram_count:
                    count_w_w_2 = self.trigram_count[cur_trigram]

                count_w_w_1 = 0
                if bigram_pair in self.bigram_count:
                    count_w_w_1 = self.bigram_count[bigram_pair]

                trigram_prob = 0
                logged_trigram_prob = 0

                # if cur_trigram not in self.trigram_probs:
                if cur_trigram not in self.trigram_probs:
                    trigram_prob = (count_w_w_2 + 1) / (count_w_w_1 + self.V_num)
                    logged_trigram_prob = math.log2(trigram_prob)
                    self.trigram_probs[cur_trigram] = round(logged_trigram_prob, 3)

    def _calculate_sentence_probs(self, sentences):
        # trigram_df_maker = []

        for i in range(len(sentences)):
            regular_sent = sentences[i]
            unk_sent = self.unked_sentences[i]
            sent_prob = 0
            sent = ''

            for j in range(2, len(unk_sent)):
                word_j_2 = unk_sent[j - 2]
                word_j_1 = unk_sent[j - 1]
                word_j = unk_sent[j]

                cur_trigram = word_j_2 + ' ' + word_j_1 + ' ' + word_j
                cur_bigram = word_j_2 + ' ' + word_j_1

                cur_bigram_count = 0
                if cur_bigram in self.bigram_count:
                    cur_bigram_count = self.bigram_count[cur_bigram]

                sent += ' ' + regular_sent[j]

                if cur_trigram in self.trigram_probs:
                    cur_trigram_prob = self.trigram_probs[cur_trigram]
                else:
                    cur_trigram_prob = 1 / (cur_bigram_count + self.V_num)
                    cur_trigram_prob = round(math.log2(cur_trigram_prob), 3)

                if cur_trigram_prob == 0:
                    raise ValueError('cur_bigram_prob is 0 when it should be a non-zero')

                sent_prob += cur_trigram_prob

            sent = sent[:-6]
            self.sent_probs[sent] = sent_prob

            # trigram_df_maker.append([unk_sent, sent_prob])


    def train(self, train_corpus):
        sentences, n_tot = counts.clean_file_contents(train_corpus)

        self.unigram_count = counts.count_unigrams(sentences)

        self.unked_sentences = counts.unk_sentences(sentences, self.unigram_count)

        self.bigram_count = counts.count_bigrams(self.unked_sentences)

        self.trigram_count = counts.count_trigrams(self.unked_sentences)

        #print()
        #print(list(self.trigram_count.items())[0:100])

        self.V_num = len(self.unigram_count)

        self._calculate_trigram_probabilities()

        columns = ['Trigram', 'Probability']
        df_trigram_probs = pd.DataFrame(data=list(self.trigram_probs.items()), columns=columns)
        #sorted_prob = sorted(self.trigram_probs.items(), key=lambda x: (-x[1], x[0]))

        # columns = ['Trigram', 'Count', 'Probability', 'log2 Probability']

        #trigram_probs_df = pd.DataFrame(data=sorted_prob, columns=columns)

        # trigram_probs_df = pd.DataFrame(trigram_df_maker, columns = columns)

        df_trigram_probs.sort_values(by=["Probability", "Trigram"], ascending=[False, True], inplace=True)

        df_trigram_probs = df_trigram_probs.set_index('Trigram')

        print(df_trigram_probs)




        #trigram_probs_df = trigram_probs_df.set_index('Trigram')


        corpus = train_corpus.replace(".txt", "")
        corpus = corpus.replace("data/", "")
        save_as_csv = "trigram_train_for_" + str(corpus) + ".csv"
        df_trigram_probs.to_csv(save_as_csv)


        # _print_sorted(sorted_prob)

    def score(self, test_corpus):
        sentences, n_tot = counts.clean_file_contents(test_corpus)

        self.unked_sentences = counts.unk_sentences(sentences, self.unigram_count)

        self._calculate_sentence_probs(sentences)

        columns = ['Sentence', 'Probability']
        df_sentence_probs = pd.DataFrame(data=list(self.sent_probs.items()), columns=columns)
        df_sentence_probs = df_sentence_probs.set_index('Sentence')
        print(df_sentence_probs)

        
        # columns = ['Sentence', 'log2 Probability']
        #trigram_probs_df = pd.DataFrame(data=sorted_prob, columns=columns)
        # trigram_probs_df = pd.DataFrame(trigram_df_maker, columns = columns)
        #trigram_probs_df.sort_values(by=["log2 Probability", "Trigram"], ascending=[False, True], inplace=True)
        #trigram_probs_df = trigram_probs_df.set_index('Trigram')

        corpus = test_corpus.replace(".txt", "")
        corpus = corpus.replace("data/", "")
        save_as_csv = "trigram_test_for_" + str(corpus) + ".csv"
        df_sentence_probs.to_csv(save_as_csv)



        # Calculate perplexity
        sum_prob = df_sentence_probs["Probability"].astype("float").sum()
        perplexity = round(2 ** (-(1 / n_tot) * sum_prob), 3)
        print()
        print('Perplexity: ' + str(perplexity))

