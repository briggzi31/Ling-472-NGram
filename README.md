# Ling-472-NGram

Prachatorn Joemjumroon and Sam Briggs

University of Washington

Ling 472 Spring 2021

In this project, we create an NGram model trained on text from Jane Austen's works. 
We create an unigram, bigram, as well as a trigram model.

What is an NGram?

An NGram is a language model which uses a probabilistic approach to language.
We predict the probability that a certain word *w<sub>i* appears given the N-1 words that  
appear before *w<sub>i*.
For example a trigram calculates:

P(w<sub>i</sub> | word<sub>i-1</sub>, word<sub>i-2</sub>)

and a 4-Gram calculates:

P(w<sub>i</sub> | word<sub>i-1</sub>, word<sub>i-2</sub>, word<sub>i-3</sub>)


To estimate the probabilities, we used these formula:

Let T denote the total number of words in the given corpus

Then for a unigram we estimate:

P(w<sub>i</sub>) := count(w<sub>i</sub>) / T

For a bigram we estimate:

P(w<sub>i</sub>) := count()

Dataset

Results

Discussion of Results

Data Statement

This creates an NGram model trained on text from Jane Austen's works.


