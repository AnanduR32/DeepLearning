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

## Encoder Decoder

- encoder: Compresses a variable length source sequence into a single, fixed-size context vector
  Source Text: "I love coding"  
  
    | Text/Input | $x_i$ | Network/Internally | Output
    | :--- | :--- | :--- | :---
    | I | [$x_1$] | RNN ($x_1+h_0$ - initial state) | [$h_1$]
    | Love | [$x_2$] | RNN ($x_1 + h_1$) | [$h_2$]
    | Coding | [$x_3$] | RNN ($x_2 + h_2$) | [$h_3$] - Context vector

- decoder: Unpacks the context vector to generate a completely new, variable-length target sequence.
  Target Text: \<SOS> J'aime le code
  > SOS - Start of sequence

    | Input | Network/Internally | Output | Text
    | :--- | :--- | :--- | :---
    $h_3$ | RNN ($h_3 + y_0$) <br/>where $y_0$ = '\<SOS>' <br/>i.e. $s_1=f_{dec}(y_0,h_3)$ | $y_1$ = softmax($s_1$) | J'aime
    | $y_1$, $s_1$ <br/>(Updated internal state,<br/>same as $h_1$ in concept) | RNN ($s_2=f_{dec}(y_1,s_1)$) | $y_2$ = softmax($s_2$) | le
    | $y_2$, $s_2$ | RNN ($s_3=f_{dec}(y_2,s_2)$) | $y_3$ = softmax($s_3$) | code

    Thus,  
    $$s_t=\sigma(W_s.[s_{t-1},y_{t-1}] + b)$$
    $$P(y_t=j|) = softmax(Vs_t + c)_j$$

    and Loss:  
    $$L(\theta)=\sum_{t=1}^TL_t(\theta)$$
    $$L_t(\theta)=-\log P(y_t=I_t | y\lt{t},X)$$
