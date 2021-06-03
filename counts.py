import string
import re

UNK = '<UNK>'

def clean_file_contents(f):
    """
    :param: Takes in a file name containing text
    Cleans the text, stripping the text of punctuation
    and splitting it into tokens. Adds <sos> and <eos> tokens for the start and end of the sentence
    respectively. This is an in-place operation. Note: this is an expensive operation
    :return: A 2-D List where rows are sentences and elements are individual tokens without white space
    """
    sentences = []
    with open(f, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        cur_sent = lines[i]
        cur_sent = cur_sent.translate(str.maketrans('', '', string.punctuation))
        cur_sent = re.sub(r'\s+', ' ', cur_sent)
        lines[i] = '<sos> <sos> ' + cur_sent.strip() + ' <eos>'
        words = lines[i].split()
        sentences.append(words)

    return sentences

def count_unigrams(sentences):
    unigram_count = {UNK: 0}
    N_tot = 0

    # calculate individual word counts
    for sentence in sentences:
        # for word in sentence:
        for i in range(0, len(sentence)):
            N_tot += 1
            word = sentence[i]

            if word not in unigram_count:
                unigram_count[word] = 1
            else:
                unigram_count[word] += 1

    # Getting rid of words only appearing once and replacing it as an UNK token.
    remove = [w for w in unigram_count if unigram_count[w] == 1]

    for word in remove:
        unigram_count.pop(word)

    # setting count of <UNK> token
    unigram_count[UNK] = len(remove)

    return unigram_count, N_tot


def count_bigrams(sentences):
    # getting bigram counts
    bigram_count = {}

    for sentence in sentences:
        for i in range(2, len(sentence)):
            word_i_1 = sentence[i-1]
            word_i = sentence[i]

            bigram = word_i_1 + ' ' + word_i
            if bigram not in bigram_count:
                bigram_count[bigram] = 1
            else:
                bigram_count[bigram] += 1

    return bigram_count

def count_trigrams(sentences):
    # getting trigram counts
    trigram_count = {}

    for sentence in sentences:
        for i in range(2, len(sentence)):
            word_i_2 = sentence[i-2]
            word_i_1 = sentence[i-1]
            word_i = sentence[i]

            trigram = word_i_2 + ' ' + word_i_1 + ' ' + word_i

            if trigram not in trigram_count:
                trigram_count[trigram] = 1
            else:
                trigram_count[trigram] += 1

    return trigram_count


def unk_sentences(sentences, unigram_count):
    unked_sentences = []

    for sentence in sentences:
        unked_sent = []
        for token in sentence:
            if token not in unigram_count:
                unked_sent.append(UNK)
            else:
                unked_sent.append(token)

        unked_sentences.append(unked_sent)

    return unked_sentences