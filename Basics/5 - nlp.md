# Natural Language Processing

## Work Embeddings

Traditionally text are represented using discrete tokens or sparse one-hot encoded vectors, but this was inefficient as it created and used sparse vectors that fail to capture semantic relationship between words

Word embeddings solve this by mapping the sparse high dimensional textual tokens into a low dimensional continuous vector space. Within this space, words that share similar semantic meanings or structural contexts are positioned close to one another.

Mathematically the similarity of 2 words, in higher dimensional embedding space, can be found using **Cosine Similarity**  
$$Cosine Similarity (A,B) = \cos{\theta}=\frac{A.B}{||A||.||B||}=\frac{\sum_i=1^d{A_iB_i}}{\sqrt{\sum_{i=1}^dA_i^2}\sqrt{\sum_{i=1}^dB_i^2}}$$
> Word embeddings are mapped using real numbers, the metric outputs a bound scalar

- ($\theta = 0^\circ$)  
If the vectors point in the exact **same direction**. The words are semantically identical or perfect synonyms.  
- ($\theta = 90^\circ$)  
The vectors are **orthogonal**. The words are completely independent and share no contextual relationship.  
- ($\theta = 180^\circ$)
The vectors point in diametrically **opposite directions**. The words are exact opposites.

> Standard euclidean or any other linear distance metric isn't suitable as the distance between two words can be very great in magnitude, hence we use angle between their vectors.

The dense vectors can then be passed through RNN/Transformer architecture to solve complex-understanding tasks, such as:

- Sentiment analysis
- Topic classification
- Question answering (QA)

## Classical Embeddings Algorithms

1. Word2Vec (Google): Uses local context window to learn word vectors based on the distrbution hypothesis. It operates via two distinct training architectures:
    - Continuous Bag of Words (CBOW): Predicts targets given it's surroundings context words
    - Skip-Gram: Predicts the surrounding context words given a singular target word.
2. GloVe (Global Vectors for Word Representation) (Stanford): Combines the advantages of local context window methods with global matrix factorization. It trains explicitly on the global word-word co-occurance matrix of massive text corpus
