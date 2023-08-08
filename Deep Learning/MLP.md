### Limitations of Linear Models
1. Monotonicity: Linear models assume a monotonic relationship between features and the model's output. This means that an increase or decrease in a feature should always result in a corresponding increase or decrease in the model's output. However, in many real-world scenarios, this assumption does not hold true, and the relationship between features and the output is more complex.
2. Nonlinear relationships: Linear models struggle to capture nonlinear relationships between features and the output. In cases where the relationship is not linear, using a linear model can lead to inaccurate predictions.
3. Contextual dependencies: Linear models fail to account for complex contextual dependencies, especially in tasks like image classification. The significance of a pixel in an image depends on its context (the values of surrounding pixels), and linear models cannot capture these interactions effectively.
4. Need for complex preprocessing: While some problems violating linearity assumptions can be addressed with preprocessing techniques, such as transforming features or adding additional features, there are cases where finding an appropriate preprocessing fix is challenging or unknown.
5. Alternative approaches: Nonlinear modeling techniques, such as decision trees, kernel methods, nonparametric spline models, and deep neural networks, have been developed to address the limitations of linear models. These approaches can capture complex relationships and interactions among features more effectively.

### Converting to Non-Linear
Let us denote a matrix $\mathcal{X}\in\mathbb{R}^{n\times d}$ which is a minibatch of $n$ examples with each example having $d$ inputs or features. If there is one hidden layer, we denote $\mathcal{H}\in\mathbb{R}^{n\times h}$ the outputs of hidden layer. We can thus have hidden-layer weights as $\mathcal{W}^{(1)}\in\mathbb{R}^{d\times h}$ and for output-layer to be $\mathcal{W}^{(2)}\in\mathbb{R}^{h\times q}$ .
Thus, the output layer would be: $\mathcal{O}\in\mathbb{R}^{n\times q}$ . We can also define the respective biases as: $\mathcal{b}^{(1)}\in \mathbb{R}^{1\times h}\text{ and } \mathcal{b}^{(2)}\in \mathbb{R}^{1\times q}$. So, the equations become:
$$
\begin{split}\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}\end{split}
$$
We can add any non-linear activation function $\sigma$ to the hidden layers. So, the equations will now become:
$$
\begin{split}\begin{aligned}
    \mathbf{H} & = \sigma\left(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}\right), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}\end{split}
$$
Since each row in $\mathbf{X}$ corresponds to an example in the minibatch, with some abuse of notation, we define the nonlinearity $\sigma$ to apply to its inputs in a row-wise fashion, i.e., one example at a time.