# Ling-472-NGram

Prachatorn Joemjumroon and Sam Briggs

University of Washington

Ling 472 Spring 2021

In this project, we create an NGram model trained on text from Jane Austen's works. 
We create an unigram, bigram, as well as a trigram model.

**What is an NGram?**

An NGram is a language model which uses a probabilistic approach to language.
We predict the probability that a certain word *w<sub>i* appears given the N-1 words that  
appear before *w<sub>i*.
For example a trigram calculates:

P(w<sub>i</sub> | word<sub>i-1</sub>, word<sub>i-2</sub>)

and a 4-Gram calculates:

P(w<sub>i</sub> | word<sub>i-1</sub>, word<sub>i-2</sub>, word<sub>i-3</sub>)


To estimate the probabilities, we used these formula:

Let N denote the total number of words in the given corpus and V denote the number of unique words in the given corpus

Then for a unigram we estimate:

![unigram equation](unigram_equation.jpg?raw=true "Unigram Equation")

For a bigram we estimate:

![bigram_equation](bigram_equation.jpg?raw=true "Bigram Equation")

For a trigram we estimate:

![trigram_equation](trigram_equation.jpg?raw=true "Trigram Equation")

For perplexity, we used the equation

![perplexity_equation](perplexity_equation.jpg?raw=true "Perplexity Equation")


**Dataset:**

We used the English versions of three different Jane Austen's books (Emma, Persuastion, and Sense and Sensibility), obtained through [NLTK's selection of Project Gutenberg texts](http://www.nltk.org/book/ch02.html). The data given to us had the sentences across the three books shuffled and then assigned 80% of the sentences to the training set, 10% to the development set, and 10% to the test set.

**Implementation:**

In our training dataset, we replaced words that only appear once in our corpus with an <UNK> token and calculate the log based 2 probability with the <UNK> tokens. Also we used the Laplace Smoothing technique to handle new words that did not appear in our trained dataset. In our test and developemnt dataset, we replaced Out-of- volcabulary (OOV) words with <UNK> tokens and calculate the log based 2 probability, including the <UNK> tokens.

**Results:**

Perplexity:

|         | Devlopment | Test  |
|---------|------------|-------|
| Unigram | 2.219      | 2.216 |
| Bigram  | 2.231      | 2.231 |
| Trigram | 2.533      | 2.529 |

**Discussion of Results:**

The perplexity is increasing as we increased the N number of grams in our model. Why the perplexity increases as we increases the number of grams is because the probability of a string of words is less likely as we increased the number of words in the string. But the perplexity is still very low, meaning that the test and development data uses similar words as the train data.  This shows that the test and develpoment data would likely to occur when we train our model with the train data. This is expected to happen because the train, test, and development data all came from the same author and the same three books. If we were to change the test, train, of development data with different authors and books, then we would expect to have a higher perplexity. If we were to continue this work in the future, we would want to generalize the Ngrams, so that we can pass in any N, to calculate the Ngram probabilities and see the trend of increasing perplexity is still true as we increase the number of N in the Ngram.

