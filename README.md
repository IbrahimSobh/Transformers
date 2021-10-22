# Transformers

## What is wrong with RNNs and CNNs
Learning Representations of Variable Length Data is a basic building block of sequence-to-sequence learning for Neural machine translation, summarization, etc
- **Recurrent Neural Networks (RNNs)** are natural fit variable-length sentences and sequences of pixels. But sequential computation inhibits parallelization. No explicit modeling of long and short-range dependencies.
- **Convolutional Neural Networks (CNNs)** are trivial to parallelize (per layer) and exploit local dependencies. However, long-distance dependencies require many layers.

## Attention!
The Transformer archeticture was proposed in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). As mentioned in the paper: 
- "_We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely_"
- "_Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train_"

Machine Translation (MT) is the task of translating a sentence x from one language (the source language) to a sentence y in another language (the target language). One basic and well known neural network architecture for NMT is called sequence-to-sequence [seq2seq](https://arxiv.org/pdf/1409.3215v3.pdf) and it involves two RNNs.
- **Encoder**: RNN network that encodes the input sequence to a single vector (sentence encoding)
- **Decoder**: RNN network that generates the output sequences conditioned on the encoder's output. (conditioned language model)

The problem of the vanilla seq2seq is information bottleneck, where the encoding of the source sentence needs to capture all information about it in one vector.

As mentioned in the paper [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf):
- "_A potential issue with this encoder–decoder approach is that a neural network needs to be able to compress all the necessary information of a source sentence into a fixed-length vector. This may make it difficult for the neural network to cope with long sentences, especially those that are longer than the sentences in the training corpus._"

![attention001.gif](images/attention001.gif)

Attention provides a solution to the bottleneck problem.
- **Core idea**: on each step of the decoder, use a direct connection to the encoder to focus on a particular part of the source sequence.
Attention is basically a technique to compute a **weighted sum** of the values (in the encoder), dependent on another value (in the decoder).

## Query and Values

- In the seq2seq + attention model, each decoder hidden state (query) attends to all the encoder hidden states (values)
- The weighted sum is a **selective summary** of the information contained in the values, where the query determines which values to focus on.
- Attention is a way to obtain a fixed-size representation of an arbitrary set of representations (the values), dependent on some other representation (the query).

## Transformers for computer vision 

## Multi-head attention


## References and more information
- Article [Transformers without pain](https://www.linkedin.com/pulse/transformers-without-pain-ibrahim-sobh-phd/)
- Article [Attention for Neural Machine Translation (NMT) without pain](https://www.linkedin.com/pulse/attention-neural-machine-translation-nmt-without-pain-sobh-phd/)
- Article [Anatomy of the Beast with many heads](https://www.linkedin.com/pulse/anatomy-beast-many-heads-code-ibrahim-sobh-phd/)
- [CS224n: Natural Language Processing with Deep Learning Stanford / Winter 2019](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/)
- [Natural Language Processing with Deep Learning (Winter 2017)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
- [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt)





