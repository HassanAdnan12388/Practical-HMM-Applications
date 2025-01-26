#!/usr/bin/env python
# coding: utf-8

# # PA2.2 Part A - Hidden Markov Model (HMM) Based Part Of Speech Tagging
# 
# ### Introduction
# 
# In this notebook, you will be making a Part of Speech (POS) Tagger using an HMM via Supervised Matrix/Parameter Initialization and Viterbi Decoding.
# 
# Hidden Markov Model (HMM) is a statistical model used to describe a system that transitions between a set of hidden states over time, generating observable outcomes.
# 
# In the context of Part-of-Speech (POS) tagging, HMMs are used to model the sequence of words in a sentence as a sequence of hidden states representing the POS tags. The observable outcomes are the actual words in the sentence.
# 
# ## Terminology
# 
# __Supervised Matrix/Parameter Initialization__: involves setting the initial values of the parameters in the model, specifically the emission probabilities matrix and the transition probabilities matrix. In the case of supervised learning tasks, where the HMM is trained on annotated data (e.g., for Part-of-Speech tagging), the initialization involves estimating the initial values of these matrices based on the observed training data. This initialization is crucial as it provides the starting point for the model to learn and adjust its parameters during the training process. The matrices are typically initialized using statistical information derived from the frequency of transitions and emissions observed in the training dataset.
# 
# __Viterbi Decoding__: is a dynamic programming algorithm used for finding the most likely sequence of hidden states (POS tags) given the observed sequence of words. It efficiently calculates the probability of a sequence of states by considering both emission and transition probabilities. Viterbi decoding helps identify the most probable sequence of POS tags for a given sentence based on the trained HMM parameters.
# 
# ![The 3 Main Questions of HMMs](hmm_questions.png)
# 
# ## Resources
# 
# For additional details of the working of HMMs, Matrix Initializations and Viterbi Decoding, you can also consult [Chapter 8](https://web.stanford.edu/~jurafsky/slp3/8.pdf) of the SLP3 book as reference.
# 
# For a more colorful tutorial, you can also refer to this guide [Hidden Markov Models - An interactive illustration](https://nipunbatra.github.io/hmm/)
# 
# Another hands-on approach to Viterbi Decoding, can be found in this medium article [Intro to the Viterbi Algorithm](https://medium.com/mlearning-ai/intro-to-the-viterbi-algorithm-8f41c3f43cf3) and can be supplemented by the following slide-set  
# [HMM : Viterbi algorithm -a toy example](https://www.cis.upenn.edu/~cis2620/notes/Example-Viterbi-DNA.pdf) from the UPenn CIS 2620 course.
# 
# ### Instructions
# 
# - Follow along with the notebook, filling out the necessary code where instructed.
# 
# - <span style="color: red;">Read the Submission Instructions, Plagiarism Policy, and Late Days Policy in the attached PDF.</span>
# 
# - <span style="color: red;">Make sure to run all cells for credit.</span>
# 
# - <span style="color: red;">Do not remove any pre-written code.</span>
# 
# - <span style="color: red;">You must attempt all parts.</span>
# 
# For this notebook, in addition to standard libraries i.e. `numpy`, `nltk`, `collections` and `tqdm`, you are permitted to incorporate supplementary libraries, but it is strongly advised to restrict their inclusion to a minimum. However, other HMM toolkits or libraries are strictly prohibited.

# In[5]:


import nltk
#Download the dataset and tagset from below:
nltk.download('conll2000')
nltk.download('universal_tagset')
from nltk.corpus import conll2000


# In[6]:


import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm


# ### POS HMM Class
# The __POS_HMM__ Class contains the following methods:  
# 
# 1. `__init__(self, corpus)`: initializes the __POS_HMM__ and prepares it for the parameter initialization phase, contains:
#     - a corpus, which is further 85-15 split into training and test sets.
#     - a list of all the (word, tag) pairs in the training set (all the sentences are concatenated hence sequence is maintained).
#     - a tuple of unique words, tags in the training set
#     - a dictionary for mapping words and tags to its unique integer identifier.
#     - some additional variables to reduce code redundancy in latter parts such as len()
#     - Transition, Emission and Initial State Probability Matrices which are initialized to Zeros.
#     - Tag Occurance Probability matrix, which is initialized with tag probabilities i.e. 
#         (count of a single tag / count of all the tags) in the training set similar to a unigram. This is used for assigning tags to new or __unknown words__ by randomly sampling from all the tag probabilities.
# 2. `init_params(self)`: (__To Be Implemented__) initializes the transition, emission and initial state probability matrices via supervised matrix/parameter learning.  \
#     __Parts to Complete__:
#     - a __method__ `word_given_tag(word, tag)`, which takes as input a word and a tag, and __*returns the count of all instances where the word had the assigned tag as its label / count of the tag*__ (i.e. a probability estimate of its occurence).
#     - a __method__ `next_tag_given_prev_tag(tag2, tag1)`which takes as input two tags, and __*returns the count of all instances where the tag 2 and was preceeded by tag 1 / count of the first tag*__ (also a probability estimate).
#     - __add code__ for populating the initial state probability matrix. This essentially contains the probabilities of all POS tags occuring at the start of the sentence.
#     - __add code__ to normalize each rows of the transition, emission and initial state probability (only has one row) matrices.
# 3. `viterbi_decoding(self, sentence)`: (__To Be Implemented__) returns the mostly likely POS Tags for each word/numeral/punctuation in the sentence.
# 4. `evaluation(self, debug = False)`: evalutes the performance of the POS Tagger on the test set and returns the testing accuracy (as a %age).

# In[15]:


class POS_HMM:
    def __init__(self, corpus): #DO NOT MODIFY THIS FUNCTION

        #-----------------DO NOT MODIFY ANYTHING BELOW THIS LINE-----------------
        self.corpus = corpus
        self.train_set, self.test_set = train_test_split( self.corpus, train_size=0.85,random_state = 777)        
        
        # Extracting Vocabulary and Tags from our training set
        self.all_pairs = [(word, tag) for sentence in self.train_set for word, tag in sentence] #List of tuples (word, POS tag) or a flattened list of tuples by concatenating all sentences
        self.vocab = tuple(set(word for (word, _) in self.all_pairs))

        self.all_tags = [tag for (_, tag) in self.all_pairs] #List of all POS tags in the trainset
        self.tags = tuple(set(self.all_tags)) #List of unique POS tags
        self.word_tag_counts = Counter(self.all_pairs)

        # Mapping vocab and tags to integers for indexing
        self.vocab2index = {word: i for i, word in enumerate(self.vocab)}
        self.tag2index = {tag: i for i, tag in enumerate(self.tags)}

        self.vocab_len = len(self.vocab) #Total Number of Vocab (Unique Words)
        self.all_tag_lengths = len(self.all_tags) #Number of tags in the trainset
        self.tag_len = len(self.tags) #Number of unique tags

        # Initialize transition and emission matrices (Default: Zeros)
        self.transition_mat = np.zeros((self.tag_len, self.tag_len))
        self.emission_mat = np.zeros((self.tag_len, self.vocab_len))
        self.initial_state_prob = np.zeros(self.tag_len)

        # Initialize POS Tag occurance probabilities for getting most likely POS Tags for unknown words
        self.tag_occur_prob= {} #Dictionary of POS Tag occurance probabilities
        all_tag_counts = Counter(self.all_tags)
        for tag in self.tags:
            self.tag_occur_prob[tag] = all_tag_counts[tag]/self.all_tag_lengths
        self.tag_counts = Counter(tag for _, tag in self.all_pairs)
        #-----------------Add additional variables here-----------------
        def word_given_tag(word, tag):
            count_tag = self.all_tags.count(tag)
            count_w_given_tag = self.all_pairs.count((word, tag))
            return count_w_given_tag / count_tag if count_tag != 0 else 0

        def next_tag_given_prev_tag(tag2, tag1):
            count_t1 = self.all_tags.count(tag1)
            count_t2_t1 = sum(1 for i in range(len(self.all_tags) - 1) 
                              if self.all_tags[i] == tag1 and self.all_tags[i + 1] == tag2)
            return count_t2_t1 / count_t1 if count_t1 != 0 else 0


    def init_params(self): #Initialize transition and emission matrices via Supervised Learning (Counting Occurences of emissions and transitions observed in the data).
        all_pairs_array = np.array(self.all_pairs)
        tags_array = np.array(self.all_tags)

        #------------------- Space provided for any additional data structures that you may need or any process that you may need to perform-------------------
        for tag in self.tags:
            tag_index = self.tag2index[tag]
            self.initial_state_prob[tag_index] = self.tag_occur_prob[tag]

            for next_tag in self.tags:
                next_tag_index = self.tag2index[next_tag]
                count_t1_t2 = sum(1 for i in range(len(self.all_pairs) - 1) 
                                  if self.all_pairs[i][1] == tag and self.all_pairs[i + 1][1] == next_tag)
                self.transition_mat[tag_index, next_tag_index] = count_t1_t2 / self.tag_counts[tag]

            for word in self.vocab:
                word_index = self.vocab2index[word]
                self.emission_mat[tag_index, word_index] = self.word_tag_counts[(word, tag)] / self.tag_counts[tag]

        # Normalize matrices
        self.normalize_matrices()

    def normalize_matrices(self):
        self.transition_mat /= self.transition_mat.sum(axis=1, keepdims=True)
        self.emission_mat /= self.emission_mat.sum(axis=1, keepdims=True)
        self.initial_state_prob /= self.initial_state_prob.sum()


        #-----------------DO NOT ADD ANYTHING BELOW THIS LINE-----------------

        def word_given_tag(word, tag):
            count_tag = self.all_tags.count(tag)
            count_w_given_tag = self.all_pairs.count((word, tag))
            return count_w_given_tag / count_tag if count_tag != 0 else 0
    
        def next_tag_given_prev_tag(tag2, tag1):
            count_t1 = self.all_tags.count(tag1)
            count_t2_t1 = sum(1 for i in range(len(self.all_tags) - 1) 
                            if self.all_tags[i] == tag1 and self.all_tags[i + 1] == tag2)
            return count_t2_t1 / count_t1 if count_t1 != 0 else 0
        
        
        # Compute Transition Matrix
        for i, t1 in enumerate(tqdm(list(self.tag2index.keys()), desc="Populating Transition Matrix", mininterval=10)):
            for j, t2 in enumerate(list(self.tag2index.keys())):
                self.transition_mat[i, j] = next_tag_given_prev_tag(t2, t1)

    # Compute Emission Matrix
        for i, tag in enumerate(tqdm(list(self.tag2index.keys()), desc="Populating Emission Matrix", mininterval=10)):
            for j, word in enumerate(list(self.vocab2index.keys())):
                self.emission_mat[i, j] = word_given_tag(word, tag)
        
        #-------------------Add your code here-------------------
        for i, tag in enumerate(tqdm(list(self.tag2index.keys()), desc="Populating Initial Probability Matrix", mininterval=10)):
            self.initial_state_prob[i] = self.all_tags.count(tag) / len(self.all_tags) if len(self.all_tags) != 0 else 0

        # Normalize matrices
        self.transition_mat = self.transition_mat / self.transition_mat.sum(axis=1, keepdims=True)
        self.emission_mat = self.emission_mat / self.emission_mat.sum(axis=1, keepdims=True)
        self.initial_state_prob = self.initial_state_prob / self.initial_state_prob.sum() if self.initial_state_prob.sum() != 0 else self.initial_state_prob
        # Compute Initial State Probability
                
        #The below code may help. You can modify it as per your requirement.
        #for i, tag in enumerate(tqdm(list(self.tag2index.keys()), desc="Populating Initial Probability Matrix", mininterval = 10)):
        #    self.initial_state_prob[i] = None

        
        # Normalize matrices i.e. each row sums to 1
                

        
        #-----------------DO NOT MODIFY ANYTHING BELOW THIS LINE-----------------



    def viterbi_decoding(self, sentence): #Sentence is a list words i.e. ["Moon", "Landing", "was", "Faked"]
        
        pred_pos_sequence = []  # Implement the Viterbi Algorithm to predict the POS tags of the given sentence

        #-------------------Add your code here-------------------
        # Number of words in the sentence and number of tags
        n_words = len(sentence)
        n_tags = len(self.tags)

        # Probability matrix, initialized with zeros
        # Each cell prob_matrix[t][i] holds the probability of the most probable tag sequence ending in tag t for the first i words in the sentence.
        prob_matrix = np.zeros((n_tags, n_words))

        # Back pointers matrix to reconstruct the path of tags
        back_pointers = np.zeros((n_tags, n_words), dtype=int)

        # Initialization step (i=0 for the first word in the sentence)
        for t in range(n_tags):
            # Probability of tag 't' being the first tag and the first word being emitted by tag 't'
            prob_matrix[t][0] = self.initial_state_prob[t] * self.emission_mat[t][self.vocab2index.get(sentence[0], -1)]
            back_pointers[t][0] = 0

        # Dynamic programming forward pass
        for i in range(1, n_words):
            for t in range(n_tags):
                # Calculate the maximum probability for each tag 't' at position 'i' in the sentence
                # And also find the tag from the previous position that contributes to this maximum probability
                max_prob, max_state = max((prob_matrix[t_prev][i - 1] * self.transition_mat[t_prev][t] * self.emission_mat[t][self.vocab2index.get(sentence[i], -1)], t_prev) for t_prev in range(n_tags))
                prob_matrix[t][i] = max_prob
                back_pointers[t][i] = max_state

        # Find the final best path through backtracking
        best_path = []
        # Start with the most probable last tag
        max_prob_last_tag = np.argmax(prob_matrix[:, n_words - 1])
        best_path.append(max_prob_last_tag)

        # Follow the back pointers to retrieve the best path
        for i in range(n_words - 1, 0, -1):
            best_tag_prev = back_pointers[best_path[-1]][i]
            best_path.append(best_tag_prev)

        # Reverse the path to get it in the correct order
        best_path.reverse()

        # Convert tag indices back to tag names
        pred_pos_sequence = [list(self.tag2index.keys())[tag_index] for tag_index in best_path]

        return pred_pos_sequence





        #-----------------DO NOT MODIFY ANYTHING BELOW THIS LINE-----------------

        return pred_pos_sequence
    
   
    def evaluation(self, debug=False): #DO NOT MODIFY THIS FUNCTION
        # Evaluate the model on the test set
        correct, total = 0, 0
        pred_pos_sequences = []

        for test_sentence in self.test_set:
            test_sentence_words, test_sentence_tags = zip(*test_sentence)
            pred_pos_tags = self.viterbi_decoding(test_sentence_words)
            pred_pos_sequences.extend(pred_pos_tags)

            correct += sum(1 for true_tag, pred_tag in zip(test_sentence_tags, pred_pos_tags) if true_tag == pred_tag)
            total += len(test_sentence_words)
        
        accuracy = correct / total if total > 0 else 1

        if debug:
            test_words, test_tags = zip(*[(word, tag) for test_sentence in self.test_set for word, tag in test_sentence])
            print(f"Sentence (first 20 words): {test_words[:20]}")
            print(f"True POS Tags (first 20 words): {test_tags[:20]}")
            print(f"Predicted POS Tags (first 20 words): {pred_pos_sequences[:20]}")

        print(f"Test Accuracy: {accuracy * 100:.4f}%")


# ### Model Evaluation

# In[16]:


pos_hmm = POS_HMM(corpus = conll2000.tagged_sents(tagset='universal'))
pos_hmm.init_params()
pos_hmm.evaluation(debug = False)

