
### Revisiting RNNs

Based on the [limitations of RNNs](RNN#Limitations), we need the following functionalities: 
1. Continuous stream of data.
2. Parallelization
3. Long Term Memory

### Naïve approach

Eliminate the time steam by concatenating every time step into one vector. We can develop a feed forward network to complete this process.
![[Pasted image 20230807153425.png]]

But, it is not scalable and there is no order information. There is no long-term memory.

Idea: Identify and attend to what's important.

### Attention

Attending to most important parts of an input.
From an image, we scan the image from eyes and the brains extract the features which requires higher attention.
What do we do:
1. Encode position information
   ![[Pasted image 20230807154029.png]]
2. Extract query, key, value for search. Get the positional encoding, apply a linear layer and output the query. Similarly for key and value. We use separate neural networks for each of three process. We compare then to check what is self-important.
   ![[Pasted image 20230807154312.png]]
3. Compute the attention weighting. Compute the similarity between key and query. We use cosine similarity.
   ![[Pasted image 20230807154456.png]]
   We can then pass the value to a SoftMax function.
4. We will then extract features with high attention.
   ![[Pasted image 20230807154703.png]]

These operations together form a self-attention head that can plug into a larger network. Each head attends to a different part of input.