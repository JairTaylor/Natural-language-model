# Natural-language-model
This was the final project for Natural Langue Processing at University of Washington, 
Natural Language Processing - Winter 2016

Jair Taylor
Donny Huang
Elizabeth Clark

(See project.pdf for project guidelines)


(i) Language Model:


Our model was an interpolated n-gram model, for n = 1,2,3,4,5.  This method hopefully prevents overfitting to certain rarely-appearing long n-grams appearing in the history that our model recognizes: there is always a chance to ignore some of the history and resort to an n-gram model for smaller n.  That is, to generate a character c given the history h, we first examine the history for the largest k< 5 so that the last k letters of h form a k-gram that is known, that is, has been seen in the training data.  Let h_k denote the last k letters of h.  Then with some fixed probabiity p, e.g., p = .75, we output a character c with probability proportional to the number of times the (k+1)-gram h_k + c has been seen in training data.  With probability (1-p) we ignore the first letter of h_k, and instead output a character c with probability proportional to the number of times the k-gram h_(k-1) + c has been seen, and so forth.  We also perform alpha-smoothing by having a fixed probability q (e.g., q = .001) of ignoring all the training data and emitting a Unicode character uniformly at random.


The interpolation of scores on the unigram, bigram, trigram, etc. level parallels smoothing techniques like interpolated stupid backoff. By taking a weighted combination of multiple models, the model can better handle out-of-vocabulary problems and overfitting. The addition of the last term prevents the probability from going to 0 in the case where the character is not in the vocabulary. By adding this same probability to all characters, this is essentially add-k smoothing. 


(It is unfortunate that we ultimately resorted to ngrams due to the constraints of the way the project is ultimately evaluated, as we were initially trying something pretty interesting.


Our initial approach was to use a word-based recurrent neural network (LSTM) model, where the input is either one-hot vectors of word clusters trained through brown clustering, or dense word-to-vec vectors. Our intuition is that whereas individual English characters, for example, lack semantic meaning, complete English words do instead have semantic meaning. This means that neural networks trained on words should be able to capture the semantic relationships within sentences, instead of just character distributions. Due to the number of words across all languages, we need to reduce the dimensionality of the input, which led us to first attempt to cluster words using brown-clustering, before ultimately deciding to use dense word-to-vec vectors.


However, there are a few issues that prevented us from completing this model for the final evaluation. The main issue is runtime. Currently, our neural network model takes as input and outputs 50 dimensional word-to-vec vectors. To go from a 50 dimensional word-to-vec vector to character probabilities, we needed to perform operations summing across all words, which becomes prohibitively expensive given the number of words we have to handle. Even when we were just experimenting in English alone, our dictionary consisted of more than 10,000 words.)


(ii) Training Data:


Our training data is composed primarily of data gathered from Tatoeba and Wikipedia. 


Tatoeba consists of a large collection of sentences and their translations, with sentences in 285 languages. In our case, while we do not care about which sentences are translations of each other, the sentences in all the different languages themselves are a good source of data for training our language model. Therefore, we randomly selected 500,000 sentences from Tatoeba as training data for our model. (As a result, the distribution of languages follows that in Tatoeba, with (1) English at 12.4%, (2) Esperanto at 8.9%, (3) Turkish at 8.6%, etc.)


For Wikipedia, we downloaded random articles across 243 different languages, and used their introductory summary paragraph as training data for our language model. The data was cleaned of hyperlinks, so it is just consists of basic text. The proportion by which we downloaded articles of different languages from wikipedia follows the number of users registered for each language. (So the distribution follows (1) English at 45.5%, (2) Spanish at 6.7%, (3) French at 4.1%, etc.) In total, we used training data gathered from 50,000 articles.




(iii) Libraries Used:
 
Our final program is implemented in Python, and uses the Numpy library. (Though throughout the course of the project, we also used TensorFlow, NLTK, BioPython, polyglot, gensim among other libraries.)
