### How does RNN work?

A typical **neural network** involves following steps:
1. Input from the dataset.
2. Take the data and apply complex computations to it using randomly initialized variables. (weights and biases)
3. Predict result.
4. Compare the value to expected result to get error.
5. Backpropagate the error to adjust the variables.
6. Repeat step 1-5 until convergence (a satisfiable value is achieved).
7. Predict an input by applying these variables.

In RNN we input one example at a time and predict one result. The difference with a feedforward network comes in the fact that we also need to be informed about the previous inputs before evaluating the result. So you can view RNNs as multiple feedforward neural networks, passing information from one to the other.

![[Pasted image 20230807140742.png]]

So we need the following equations:
1. $h_t=f(W^{(hh)}h_{t-1}+W^{(hx)}x_{t})$ : This holds the information about the previous layer. We apply a non-linear activation function (tanh or sigmoid). 
2. $y_t=softmax(W^{(S)}h_t)$: Calculates the predicted word vector at a given time step t.
3. $J^{(t)}(\theta)=\sum_{i=1}^{\left|V\right|}(y_{t_i}log(y_{t_i}))$: cross-entropy Loss function

_$W$’s_ are, each of them represents the weights of the network at a certain stage. As mentioned above, the weights are matrices initialized with random elements, adjusted using the error from the loss function. We do this adjusting using back-propagation algorithm which updates the weights.

### Places where RNN can be used:
1. Many to One (Sentiment Classification)
2. One to Many (Text or Image generation)
3. Many to Many (Music Generation or Translation and forecasting)
![[Pasted image 20230807142025.png]]

### Design Criteria for Sequence Modelling.
1. Handle variable lengths
2. Track long-term dependency.
3. Maintain information about order.
4. Share parameters across sequences.

### Backpropagation through time

![[Pasted image 20230807150217.png]]

Take the cumulative loss, backpropagate through individual time steps and do the same process across the time steps, i.e., backpropagate from current time to when the process started in the past.

### Problems with RNN

The simplest RNN model has a major drawback, called **vanishing gradient problem,** which prevents it from being accurate.  The problem comes from the fact that at each time step during training we are using the same weights to calculate $y_t$. That multiplication is also done during back-propagation. The further we move backwards, the bigger or smaller our error signal becomes. This means that **the network experiences difficulty in memorizing words from far away in the sequence** and makes predictions based on only the most recent ones. 
One way to solve that is through ReLU activation function. It prevents shrinking as the derivative is 1 when $x>0$. 
Another way is to Initialize weights to Identity and biases to 0 vector.
We can use Gated Cells. Basically, use gates to selectively add or remove information within each recurrent unit. One example is through Long Short Term Memory (LSTMs) as it replies on a gated cell to track information throughout many time steps. It maintains a cell state. Use gates to control the flow of information by using _Forget_ gate which gets rid of irrelevant information, _storing_ relevant information from current input, selectively _updating_ the cell state and lastly, _output_ gate returns a filtered version of the cell state.

### Limitations

1. Encoding bottleneck. Encode a lot of content into a single output. Thus, there is no way to ensure that all the information was maintained and learned. 
2. They are slow and no easy way to parallelize.
3. No long memory.