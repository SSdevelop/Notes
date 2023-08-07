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
### Problems with RNN

The simplest RNN model has a major drawback, called **vanishing gradient problem,** which prevents it from being accurate.  The problem comes from the fact that at each time step during training we are using the same weights to calculate $y_t$. That multiplication is also done during back-propagation. The further we move backwards, the bigger or smaller our error signal becomes. This means that **the network experiences difficulty in memorizing words from far away in the sequence** and makes predictions based on only the most recent ones.