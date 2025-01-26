#!/usr/bin/env python
# coding: utf-8

# # PA2.2 Part B - Hidden Markov Model (HMM) Based Music Generator
# 
# ### Introduction
# 
# In this notebook, you will be generating Music (no vocals though) using an HMM via [Baum-Welch](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm) Training Algorithm.
# 
# In the context of Music Generation, the states might represent underlying musical concepts (like note pitches or chord types), and observations could be specific notes or chords played at a given time. Hence, generating music involves moving through the states based on transition probabilities and producing musical notes based on the emission probabilities associated with each state. The emission probabilities dictate how likely it is to observe each possible note or chord (observation symbol) when in a given state.
# 
# ## Terminology
# 
# __Baum Welch__: is an __*Unsupervised*__ training algorithm that involves adjusting the HMM's parameters (transition, emission, and initial state probabilities) to best account for the observed sequences.The training process involves:
# - Expectation step (E-step): Estimate the likely sequences of hidden states (could be something implicit like musical concepts like chords or rhythms) given the current parameters of the model and the observed data.
# - Maximization step (M-step): Update the model's parameters to maximize the likelihood of the observed data, based on the estimated sequences of hidden states.
# 
# ![Baum Welch](unsupervised_learning.png)
# 
# ## Resources
# 
# For additional details of the working of Baum-Welch Training you can consult these medium articles [Baum-Welch algorithm](https://medium.com/mlearning-ai/baum-welch-algorithm-4d4514cf9dbe) and [Baum-Welch algorithm for training a Hidden Markov Model — Part 2 of the HMM series](https://medium.com/analytics-vidhya/baum-welch-algorithm-for-training-a-hidden-markov-model-part-2-of-the-hmm-series-d0e393b4fb86) as reference.
# 
# A more technical overview is covered by Rabiner in his paper on [A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition](http://www.stat.columbia.edu/~liam/teaching/neurostat-fall17/papers/hmm/rabiner.pdf).
# 
# If the above link is a bit difficult to digest, you can consult the following slides by Stanford's Dan Jurafsky in his course [LSA 352: Speech Recognition and Synthesis](https://nlp.stanford.edu/courses/lsa352/lsa352.lec7.6up.pdf).
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
# For this notebook, in addition to standard libraries i.e. `numpy`, `tqdm`, `hmmlearn` and `muspy`, you are permitted to incorporate supplementary libraries, but it is strongly advised to restrict their inclusion to a minimum. However, other HMM toolkits or libraries are strictly prohibited.

# In[4]:


#!pip install muspy
#!pip install hmmlearn
#muspy.download_musescore_soundfont()

import numpy as np
from tqdm.notebook import tqdm as tqdm
import muspy
from hmmlearn.hmm import CategoricalHMM


# **MusPy** is an open source Python library for symbolic music generation. It provides essential tools for developing a music generation system, including dataset management, data I/O,  
# data preprocessing and model evaluation.  
# **Documentation**: https://salu133445.github.io/muspy/

# In[5]:


#List of available datasets in muspy (could also add your own datasets)
#Link: https://salu133445.github.io/muspy/datasets/datasets.html
print(muspy.list_datasets())


# Since Baum-Welch is known for its slow convergence, we'll take the lightest dataset available from muspy datasets called the __HaydnOp20__ Dataset consisting of 1.26 hours of recordings \
# comprising of 24 classical songs

# In[8]:


my_dataset = muspy.datasets.HaydnOp20Dataset(root = '/Users/HassanAdnan/Desktop/PA2.2', download_and_extract=True)
my_dataset = my_dataset.convert()


# ## Section 1: (HMM + Baum Welch From Scratch) [80 Marks]

# Muspy offers all datasets in 4 different representations as mentioned below: 
# 
# ![Alt text](muspy_representations.png)
# 
# Initially, we are only interested in modelling through time and to keep it simple, we'll begin with the __Pitch Representation__. More details here:
# 
# ![text](muspy_to_pitch.png)

# In[9]:


music_collection = []
for music in my_dataset:
    music_collection.append(muspy.to_pitch_representation(music, use_hold_state=True))


# ### Singing HMM Class
# The __Singing_HMM__ Class contains the following methods:
# 
# 1. `__init__(self, corpus)`: initializes the __POS_HMM__ and prepares it for the parameter initialization phase, contains:
#     - a corpus, consisting of unlabeled  sequences of musical units (i.e. all the music songs are flattened and concatenated)
#     - a hidden_state_size (default to 10), higher values capture more variability but converge slowly.
#     - a tuple of all the unique 
#     - a dictionary for mapping the pitches to its unique integer identifier.
#     - some additional variables to reduce code redundancy in latter parts such as len()
#     - Transition, Emission and Initial State Probability Matrices which are initialized to Zeros.
# 
# 2. `init_mat(self, init_scheme='uniform')`: __(Can Be Modified)__ initializes the transition, emission and probability matrices either with a 'uniform' value or values sampled randomly from a uniform distribution and normalizes the matrice row wise.
# 
# 3. `forward(self, sequence)`: __(To Be Implemented)__ implements the Forward stage of the Forward-Backward Algorithm. 
# - Feel free to modify function signature and return values.
# - Do not change the function name.
# 4. `backward(self, sequence)`: __(To Be Implemented)__ implements the Forward stage of the Forward-Backward Algorithm. 
# - Feel free to modify function signature and return values.
# - Do not change the function name.
# 6. `baum_welch(sequence, alpha, beta)`: __(To Be Implemented)__ implements the Baum Welch Training Algorithm. 
# - Feel free to modify function signature and return values.
# - Do not change the function name.
# 7. `softmax(self, x, temperature=1.0)`: calculates the softmax of a given input x adjusting the sharpness of the distribution based on a temperature parameter.
# 
# 8. `temperature_choice(self, probabilities, temperature=1.0)`: applies a temperature scaling to a set of probabilities and selects an index based on the adjusted probabilities.
# 
# 9. `sample_sequence(self, length, strategy = "temperature", temperature = 1.0)`: __(Can Be Modified)__ generates a sequence of elements based on a given strategy (probabilistic or temperature) and a specified length. Strategies consists of:
# * `probabilistic` strategy:
#     -  Samples the initial state based on initial state probabilities.
#     -  Iterates over the desired sequence length, sampling an observation based on the current state's emission probabilities, appending the observation to the sequence, and then transitioning to the next state based on the current state's transition probabilities.
# * `temperature` strategy:
#     -  Similar to the probabilistic strategy but applies temperature scaling to the choice of initial state, observation sampling, and state transitions to adjust the randomness of the choices.

# ##### __READ THIS BEFORE YOU BEGIN__:
# - The functions `init_mat` and  `sample_sequence` are although pre-defined and will work properly, but if you have a better strategy feel free to add or experiment. Just make sure not to overwrite the pre-existing code.
# - You are allowed to make helper functions, just make sure they are neatly structured i.e. have self-explanatory variable names, have an algorithmic flow to them and are sufficiently commented.
# - Make sure not to change any exisiting code unless allowed by the TA.
# 
# __Tips for Baum-Welch Implementation__:
# 
# 1. Write the code for simple/vanilla Baum Welch Implementation first.
# 2. You have the option to either go over the whole concatenated sequence or each music seperately (in a nested for loop) per iteration.
# 3. If your vanilla Baum Welch Implementation compiles, most likely you would get overflow errors, arising from division by 0. This is due to long sequences yielding  \
# smaller values of alpha and beta. Hence, wherever division occurs, the denominator variable (which is a result of multiplication with alpha or beta) is close to 0.
# 
# I'll now suggest some ways with which the third point can be alleviated __(the hacky ways might/might not work, so be wary)__:
# 
# - Hacky way #1 (Working with smaller chunks of observed sequences): For every iteration, rather than going over the concatenated music sequences or each music sequence, you can further break down your musical sequences into even smaller chunks and go over those instead.
# 
# - Hacky way #2 (Add a small epsilon value to the denominator): Add a small episilon value like 1e-12 to the denominator wherever the division by 0 error occurs. 
# 
# - Proper way #1 (The [log-sum-exp](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) trick): For an HMM, the smaller values can be dealt with by passing them through
#     log and converting the multiplications to additions and then brought back via exponentiating them.
# 
#     - Another [intro](https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/) for the log-sum-exp, if the previous one was unclear.
#     - [Hidden Markov Models By Herman Kemper](https://www.kamperh.com/nlp817/notes/05_hmm_notes.pdf) illustrates the use of log-sum-exp technique in Baum Welch Implementation (particularly Forward and Backward Passes).
#     - [Recition 7: Hidden Markov Models](https://www.cs.cmu.edu/~mgormley/courses/10601-s23/handouts/hw7_recitation_solution.pdf) gives an idea of the usage of log-sum-exp in the forward-backward algorithm.
#     - This HMM github [repo](https://github.com/jwmi/HMM/blob/master/hmm.jl) has implemented the log-sum-exp trick in julia language.
#     - The following [blog post](https://gregorygundersen.com/blog/2020/11/28/hmms/#implementation) might also be helpful for implementation of baum-welch using log-sum-exp trick.
#     - The following paper on [Numerically Stable Hidden Markov Models](http://bozeman.genome.washington.edu/compbio/mbt599_2006/hmm_scaling_revised.pdf) gives pseudocodes for working in the log domain for the HMMs (although not necessarily the log-sum-exp trick as is).
# 
# - Proper way #2 (Scaling Factors): involves scaling the alpha and beta values to avoid underflows.
#     - The following blog post explains the maths behind scaling [Scaling Factors for Hidden Markov Models](https://gregorygundersen.com/blog/2022/08/13/hmm-scaling-factors/)
#     - This stackexchange post [Scaling step in Baum-Welch algorithm](https://stats.stackexchange.com/questions/274175/scaling-step-in-baum-welch-algorithm) contains two answers which can also be consulted.
# 
# __How do you know the HMM is converging?__:
# 
# Since Baum Welch algorithm guarantees convergence to the local (not global) maxima, near zero values are difficult to achieve.  \
# Hence, a convergening HMM would have the log likelihoods going towards 0 (although still far from it). You can find a sample cell output  \
# below showing the log likelihoods decreasing. Another way is to see is that the post-convergence generated music would be better than the  \
# starting HMM (which has uniform or randomly initialized matrices).
# 
# __How do you know the HMM has converged?__:
# 
# One way is to monitor the difference between two successive log likelihoods and stop when the differences goes below a certain threshold. This has already been implemented for you.

# In[137]:


import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm

class Singing_HMM:
    def __init__(self, corpus, hidden_state_size=10):
        self.corpus = [seq.flatten().tolist() for seq in corpus]
        self.hidden_state_size = hidden_state_size
        self.music_seq = [note for seq in self.corpus for note in seq]
        self.vocab = tuple(set(self.music_seq))
        self.vocab2index = {note: i for i, note in enumerate(self.vocab)}
        self.vocab_len = len(self.vocab)
        
        self.transition_mat = np.zeros((self.hidden_state_size, self.hidden_state_size))
        self.emission_mat = np.zeros((self.hidden_state_size, self.vocab_len))
        self.initial_state_prob = np.zeros(self.hidden_state_size)

    def init_mat(self, init_scheme='uniform'): # Can be optionally modified for another initialization scheme (not necessary for the assignment)
        if init_scheme == 'uniform':
            self.transition_mat = np.ones((self.hidden_state_size, self.hidden_state_size))
            self.emission_mat = np.ones((self.hidden_state_size, self.vocab_len))
            self.initial_state_prob = np.ones(self.hidden_state_size)
        elif init_scheme == 'random':
            self.transition_mat = np.random.rand(self.hidden_state_size, self.hidden_state_size)
            self.emission_mat = np.random.rand(self.hidden_state_size, self.vocab_len)
            self.initial_state_prob = np.random.rand(self.hidden_state_size)
        
        self.transition_mat /= self.transition_mat.sum(axis=1, keepdims=True)
        self.emission_mat /= self.emission_mat.sum(axis=1, keepdims=True)
        self.initial_state_prob /= self.initial_state_prob.sum()
    
    def forward(self, sequence):
        """
        Forward algorithm for calculating the probabilities of a sequence.
        """
        seq_len = len(sequence)
        alpha = np.random.randn(self.hidden_state_size, seq_len)
        
        # Initialization
        for i in range(self.hidden_state_size):
            alpha[i, 0] = np.log(self.initial_state_prob[i]) + np.log(self.emission_mat[i, self.vocab2index[sequence[0]]])
        
        # Induction
        for t in range(1, seq_len):
            for j in range(self.hidden_state_size):
                log_alpha_sum = logsumexp(alpha[:, t-1] + np.log(self.transition_mat[:, j]))
                alpha[j, t] = log_alpha_sum + np.log(self.emission_mat[j, self.vocab2index[sequence[t]]])
        
        # Debugging print statements
        print("Alpha:")
        print(alpha)
        
        return alpha


    def backward(self, sequence):
        """
        Backward algorithm for calculating the probabilities of a sequence.
        """
        T = len(sequence)
        beta = np.random.randn(self.hidden_state_size, T)

        # Initialization step
        for s in range(self.hidden_state_size):
            beta[s, T-1] = 0

        # Recursion step
        for t in range(T-2, -1, -1):
            for s in range(self.hidden_state_size):
                log_beta_sum = logsumexp(np.log(self.transition_mat[s, :]) + np.log(self.emission_mat[:, self.vocab2index[sequence[t+1]]]) + beta[:, t+1])
                beta[s, t] = log_beta_sum
        
        # Normalize beta values
        beta -= logsumexp(beta)
        beta = np.exp(beta)

        return beta
    
    def baum_welch(self, n_iter=100, tol=1e-4):
        """
        Perform Baum-Welch training to update the model's parameters.
        """

        prev_log_likelihood = float('-inf')  # Initialize with negative infinity (DO NOT CHANGE THIS VARIABLE)

        for iteration in tqdm(range(n_iter), desc="Training Progress", leave=True):
            log_likelihood = 0 # Log likelihood for this iteration (DO NOT CHANGE THIS VARIABLE)
            #----------------Add Your Code Here----------------
            alpha_list = []
            beta_list = []
            xi_list = []
            gamma_list = []

            for seq in self.corpus:
                seq_len = len(seq)
                alpha = self.forward(seq)
                beta = self.backward(seq)
                alpha_list.append(alpha)
                beta_list.append(beta)
                
                # Compute xi and gamma
                xi = np.zeros((seq_len - 1, self.hidden_state_size, self.hidden_state_size))
                gamma = np.zeros((seq_len, self.hidden_state_size))
                for t in range(seq_len - 1):
                    for i in range(self.hidden_state_size):
                        for j in range(self.hidden_state_size):
                            xi[t, i, j] = alpha[i, t] + np.log(self.transition_mat[i, j]) + np.log(self.emission_mat[j, self.vocab2index[seq[t+1]]]) + beta[j, t+1]
                    xi[t] -= logsumexp(xi[t])
                    gamma[t] = logsumexp(xi[t], axis=1)
                
                for i in range(self.hidden_state_size):
                    gamma[seq_len-1, i] = alpha[i, seq_len-1] + beta[i, seq_len-1]
                gamma[seq_len-1] -= logsumexp(gamma[seq_len-1])

                xi_list.append(xi)
                gamma_list.append(gamma)
                
                log_likelihood += logsumexp(alpha[:, seq_len-1])

                # Print values for debugging
                print("Alpha:")
                print(alpha)
                print("Beta:")
                print(beta)
                print("Xi:")
                print(xi)
                print("Gamma:")
                print(gamma)

            # Update transition matrix, emission matrix, and initial state probability
            for i in range(self.hidden_state_size):
                self.initial_state_prob[i] = np.mean([np.exp(gamma[0, i]) for gamma in gamma_list])
                for j in range(self.hidden_state_size):
                    numerator = np.sum([np.exp(xi[t, i, j]) for xi in xi_list for t in range(len(xi))])
                    denominator = np.sum([np.exp(gamma[t, i]) for gamma in gamma_list for t in range(len(gamma)-1)])
                    self.transition_mat[i, j] = numerator / denominator
                    
            for j in range(self.hidden_state_size):
                for k in range(self.vocab_len):
                    numerator = 0
                    denominator = 0
                    for seq, gamma in zip(self.corpus, gamma_list):
                        for t in range(len(seq)):
                            if seq[t] == self.vocab[k]:
                                numerator += np.exp(gamma[t, j])
                            denominator += np.exp(gamma[t, j])
                    self.emission_mat[j, k] = numerator / denominator
            
            # Print log likelihood for debugging
            print(f"Iteration {iteration + 1}: Log Likelihood: {log_likelihood}")

            #----------------Do Not Modify The Code Below This Line----------------

            if iteration == 0:
                convergence_rate = convergence_diff = np.nan  # Print nan for the first iteration
            else:
                convergence_diff = np.abs(log_likelihood - prev_log_likelihood)
                convergence_rate = convergence_diff / np.abs(prev_log_likelihood)
            
            #Note that Log Likelihoods would be negative and would increase (i.e. go in the direction of 0) as the model converges.
            # Log Likelihoods may be far from 0, but the increasing trend should remain present.
            tqdm.write(f"Iteration {iteration + 1}: Log Likelihood: {log_likelihood}, Convergence Difference: {convergence_diff} , Convergence Rate: {convergence_rate}")
            
            if iteration > 0 and convergence_rate < tol:
                tqdm.write("Convergence achieved.")
                break
            
            prev_log_likelihood = log_likelihood



    def softmax(self, x, temperature=1.0):
        '''Compute softmax values for each set of scores in x.'''
        e_x = np.exp((x - np.max(x)) / temperature)
        return e_x / e_x.sum()

    def temperature_choice(self, probabilities, temperature=1.0):
        '''Apply temperature to probabilities and make a choice.'''
        adjusted_probs = self.softmax(np.log(probabilities + 1e-9), temperature)  # Adding epsilon to avoid log(0)
        return np.random.choice(len(probabilities), p=adjusted_probs)
                        
    def sample_sequence(self, length, strategy = "temperature", temperature = 1.0):
        sequence = []
        if strategy == 'probabilistic':
            # Sample the initial state
            state = np.random.choice(self.hidden_state_size, p=self.initial_state_prob)
            for _ in range(length):
                # Sample an observation (note) based on the current state
                note = np.random.choice(self.vocab, p=self.emission_mat[state])
                sequence.append(note)
                # Transition to the next state
                state = np.random.choice(self.hidden_state_size, p=self.transition_mat[state])
        elif strategy == 'temperature':
            # Sample the initial state with temperature
            state = self.temperature_choice(self.initial_state_prob, temperature)
            for _ in range(length):
                # Apply temperature to emission probabilities and sample a note
                note = self.temperature_choice(self.emission_mat[state], temperature)
                sequence.append(self.vocab[note])
                # Transition to the next state with temperature
                state = self.temperature_choice(self.transition_mat[state], temperature)
        return sequence


# In[138]:


#Specify values and run the code to test your implementation
pos_hmm = Singing_HMM(corpus=music_collection, hidden_state_size=10)
pos_hmm.init_mat(init_scheme='uniform')  # Choose the initialization scheme
pos_hmm.baum_welch(tol=1e-4, n_iter=100)  # Set the tolerance and number of iterations


# In[139]:


notes_seq = pos_hmm.sample_sequence(1024, strategy= "probabilistic") #Feel free to experiment with the sampling strategy
synthetic_music = muspy.from_pitch_representation(np.array(notes_seq), resolution=24, program=0, is_drum=False, use_hold_state=True, default_velocity=64)
muspy.write_midi('/Users/HassanAdnan/Desktop/PA2.2/pitch_based.mid', synthetic_music) #Specify the path to save the MIDI file, name it "pitch_based.mid"


# You can visualize your results here: https://cifkao.github.io/html-midi-player/  
#   
# Remember to brag about your generated music on Slack (You can use online __MIDI to WAV/MP3__ Converters)
# 
# __*P.S*__: You can use muspy.write_audio to convert the music object directly to wav file but that requires installation of a few softwares (not worth the hassle).
# 
# *You might notice that the results are although better than random but they are not as awe-inspiring as intended.
# The reason being that our model is unable to capture the  \
# variability of the different music styles (our dataset comprises of). However, there is a way to generate better music, that is taking a sufficiently long MIDI  \
# (could be other formats as well) sound track(s) of a single artist (EDM or any music which has repetitiveness in it) and refitting your HMM.  \
# The relevent function here would be muspy.read_midi().  \
# __After training on the notebook provided dataset, you are more than welcome to try it with your own curated dataset and see the results.  \
# THIS IS OPTIONAL AND NOT MANDATORY__.*

# ## Section 2: Synthetic Music Generation via __HMMLearn__ [20 Marks]

# #### __Note:__ For any model that you train/fit, remember to set __verbose = True__

# In[97]:


hidden_states = 32 #Number of hidden states in the HMM model (Feel free to change or experiment with this value)
sythetic_music_sequence_length = 128 #Length of the synthetic music sequence to be generated (could be either a time step, an event or a note)


# For starters, let's replicate what we did above manually with our HMM Library. Since, we already did pitch based representation,  
# let's do it for **Event Based Representation** (which is essentially denotes music as a sequence of events). So while pitch based representation  
# is between 0-128 unique pitch values, the event based representation is between 0-387 unique events.

# In[160]:


import muspy
import numpy as np
from hmmlearn import hmm



# Define the model parameters
num_components = 5  # Number of hidden states
num_iterations = 500  # Number of iterations for training
tolerance = 0.01  # Tolerance for convergence
resolution = 24  # MIDI resolution
program = 0  # MIDI program number (instrument)
is_drum = False  # Whether it's a drum track
use_start_end = False  # Whether to use start and end tokens in the note representation
encode_velocity = True  # Whether to encode note velocity
default_velocity = 64  # Default velocity if velocity is not encoded in the note representation

# Flatten the music collection to fit the model
flattened_music = np.concatenate([muspy.to_note_representation(music, use_start_end=use_start_end, encode_velocity=encode_velocity, dtype=int) for music in music_collection])

# Initialize and fit a Gaussian HMM model
model = hmm.GaussianHMM(n_components=num_components, covariance_type="diag", n_iter=num_iterations, tol=tolerance, verbose=True)
model.fit(flattened_music)




# In[161]:


#Sampling a sequence of music from the model and save as a MIDI file, name it "event_based.mid"
# Generate a sequence of notes
generated_sequence = model.sample(500)[0]  # Generate 500 notes

# Round to nearest integers and ensure non-negative times
generated_sequence = np.rint(generated_sequence).astype(int)
generated_sequence[:, 0] = np.abs(generated_sequence[:, 0])  # Ensure time is non-negative

# Sort the notes by time
generated_sequence = generated_sequence[generated_sequence[:, 0].argsort()]

# Convert the generated sequence back to a muspy Music object
generated_music = muspy.from_note_representation(generated_sequence, resolution=resolution, program=program, is_drum=is_drum, use_start_end=use_start_end, encode_velocity=encode_velocity, default_velocity=default_velocity)

# Save the generated music as a MIDI file
muspy.write_midi("event_based.mid", generated_music)


# To add some fun, lets take it up a notch and go for the __Note Based Representation__.  
# More on that here: 
# 1. https://muspy.readthedocs.io/en/v0.3.0/representations/note.html 
# 2. https://salu133445.github.io/muspy/classes/note.html

# **Hint:** This is a bit tricky since we have 4 features per observation. We'll leave it to you to devise a way to deal with it.  \
# There are alot of approaches which can be used. As some features are categorical, and some are continuous, hence you can try different HMMs types or a single HMM to rule them all. Just generate some good music. \
# Before you start, do take a peek of the available HMM models in the HMMlearn library __(you are allowed to import additional models if you want)__

# In[162]:


#Write your code here and save the sampled sequence as MIDI file, name it "note_based.mid"z
 

import muspy
import numpy as np
from hmmlearn import hmm

music_collection = []
for music in my_dataset:  
    note_representation = muspy.to_note_representation(music, use_start_end=False, encode_velocity=True, dtype=int)
    music_collection.append(note_representation)
    
model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=500, tol=0.01, verbose=True)
# Flatten your music collection to fit the model
flattened_music = np.concatenate(music_collection)
model.fit(flattened_music)


generated_sequence = model.sample(500)[0]  

# Round to nearest integers and ensure non-negative times
generated_sequence = np.rint(generated_sequence).astype(int)
generated_sequence[:, 0] = np.abs(generated_sequence[:, 0])  

# Sort the notes by time
generated_sequence = generated_sequence[generated_sequence[:, 0].argsort()]

generated_music = muspy.from_note_representation(generated_sequence, resolution=24, program=0, is_drum=False, use_start_end=False, encode_velocity=True, default_velocity=64)

muspy.write_midi("note_based.mid", generated_music)

