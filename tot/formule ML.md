# Overfitting: 

Given a hypotheses space H, a hypotheses $h \in H$, is said to overfit the training data if there exists some $h' \in H$, such that: 

$$ \text{error}_s(h) < \text{error}_s(h') $$
$$ \text{error}_d(h) > \text{error}_d(h') $$

### Sample error
$$ \text{error}_s(h) = \frac{1}{n} \sum_{x \in S} \delta(f(x) \not ={h(x)})$$


### True Error
$$ \text{error}_D(h) = \Pr_{x \in D}(f(x) \not ={h(x)})$$


## Overfitting Decision Trees

A decision three that perfectly classify training examples could lead to overfitting when there is either noise or/and not enough samples in the dataset. 

### Solutions: 
1. Stop growing the tree when data splitting not more statistically signficant 
2. Grow the full tree and then post prune 


# Bayesian Learning

## Maximum a Posteriori Hypothesis

$$Vmap = \argmax P(h \mid D) = argmax \frac{P(D\mid h)P(h)}{P(D)}$$
$$ = argmax {P(D\mid h)P(h)}$$

## Maximum Likelihood

$$V_{ML} = \argmax P(h \mid D) = \argmax P(D\mid h)$$

under the assumption P(hi) = P(hj), $j\not ={i}$


## Naive and Approximation to Optimal

### Optimal
The optimal Bayes Classifier is defined as: 

$$ \text{Vob} = \argmax_{vj \in V} \sum_{hi \in H} P(vj \mid x, hi)P(hi \mid D)$$

The optimal classifier is an optimal learner that maximizes the probs that a class vj is correctly classified, is very powerful (an hypothesis might be a linear combination) but it represents just an upperbound for not optimal classifier. However, It is not fairly practical in real situation, especially due to its computationally intensive nature.


### Naive 

On the other hand the naive bayes classifier is a good approximation based on conditional indipendence. 
Indeed, given a Dataset D an instance is defined as: 
$$ x = <a1, a2, .... , an > $$ 
so computing the map we get: 

$$ \text{V}_{map} = \argmax P(vj \mid <a1, a2, .... , an>, D) $$

$$ = \argmax P(<a1, a2, .... , an> \mid vj, D)P(vj \mid D) $$

Under the naive bayes assumption: 

$$ \text{V}_{map} = \argmax P(vj \mid D) \prod_{i=1}^{n} P(ai \mid vj, D)$$

Even that given the assumption is not always true (less accurate), it is more feasible than an optimal classifier.


### Document classification

$$ \text{doc}_{i} = \{abstract \cup title \cup author \cup pubblication \}$$

We can now set up a dataset vocabulary.

So given a new doc doc_i we want to compute: 

$$ \text{V}_{NB} = \argmax P(cj \mid D) \prod_{i=1}^{n} P(ai \mid cj, D)$$
$$ = \argmax P(cj \mid D) P(d \mid cj, D)$$

We use an approaching bag of words (BoW) based on a multinomial distribution for multiclass problem. 

We represent a doc as a fixed-lenght feature vector d given that we have

$$ d = <d1, ..., dn>$$

where d_i = k if word i occurs k time in doc_i

So for each feature we compute: 

$$ P(d|c_j, D) = \frac{n!}{d_1! \dots d_n!} \prod_{i=1}^{n} P(w_i|c_j, D)^{d_i} $$

Maximum-likelihood solution:

$$ \hat{P}(w_i|c_j, D) = \frac{\sum_{\text{doc} \in D} tf_{i,j} + \alpha}{\sum_{\text{doc} \in D} tf_j + \alpha \cdot |V|} $$

dove:

1. $$  \text{tf}_{i,j} : \text{numero di occorrenze di \( w_i \) nel documento doc della classe \( c_j \)}.$$ 
2. $$  \text{tf}_{j} : \text{frequenza di tutti i termini del documento \( \text{doc} \) della classe \( c_j \)}.$$ 
3. $$  \alpha\text{: parametro di smoothing}.$$ 


# K-NN 

## Main step: 

Given a learning function and a labeled Dataset D, for a new instance x': 

1. Find K-Nearest neighbors of new instance x'
2. Assign to x' the most common label among the k-nearest neighbors to x'

Likelihood of Class C for a new instance x': 

$$ P(C|x,D, K) = \frac{1}{k} \sum_{xn \in Nk(xn,D)} I(tn=C)$$


# Backpropagation

## Forward step

(![Alt text](12-ANN_2p.jpeg))

## Backward step
![Alt text](12-ANN_3-1.jpeg)


## SGD

![Alt text](image.png)



# Linear Regression

Given: 

$$ y(x,w) = w0 + w1x1 + ... + wdxd = w^Tx$$

## Batch Mode: 

If we claims that the target values is given by: 

$$ t = y(x,w) + \epsilon$$

and under the assumption that the epsilon error is Gaussian: 

$$ P(t \mid x, w, \beta) = N(t\mid y(x,w), \beta^{-1})$$

so under the assumption that observations are i.i.d: 

$$ P(t \mid x, w, \beta) = \prod_{i=1}^{n} N(t\mid w^Txn, \beta^{-1})$$

In order to maximize this quantity we use the log-likehood: 

$$ \ln P(t \mid x, w, \beta) = \frac{N}{2}\beta - \frac{N}{2}2\pi - \beta * E_D(W)$$

where: 

$$ E_d(W) = \frac{1}{2} \sum_{i=1}^{n} ({tn - w^Txn})^2$$

and we remember that maximize that quantity is like minimize a sum-of-squares, therefore: 

$$ \nabla \ln P(t \mid x, w, \beta) = \sum_{i=1}^{n} ({tn - w^Txn})xn^T$$

setting to zero: 

$$ 0 = \sum_{i=1}^{n} tnxn^T - w^T (\sum_{i=1}^{n}{x_{n}x_{n}^T})$$ 

solving to w, Maximum log-Likelihood Solution

$$w = (X^TX)^{-1} X^T t $$ 

## Sequential Learning

$$ w \leftarrow w + \Delta wi $$

$$ \Delta wi = - \eta \frac{\partial E}{\partial wi} = \eta \sum_{i=1}^{n} ({tn - w^Txn}) (-x_{i,n})$$


# Gram Matrix and Linear kernelized

## Gram Matrix

A Gram Matrix is an $N \times N$ symmetric matrix with elements: 

$$ Knm = x_{n}^Tx_{m} = K(xn,xm)$$
$$
K = \begin{bmatrix}
    K(x_1, x_1) & K(x_1, x_2) & \cdots & K(x_1, x_n) \\
    K(x_2, x_1) & K(x_2, x_2) & \cdots & K(x_2, x_n) \\
    \vdots  & \vdots  & \ddots & \vdots  \\
    K(x_n, x_1) & K(x_n, x_2) & \cdots & K(x_n, x_n)
\end{bmatrix}
$$

## Kernelized version for regression



# K-Means

1. Set a K Value
2. Take K single clusters and assigns N-K samples to them based on distance between centroids and point. After each assignment recompute the centroid
3. Take each sample and compute its distance from the centroid of the clusters, if the samples it is not in the centroid closest clusters switch it. Recompute the centroid
4. Repeat 3 until convergence 



# Logistic Regression 2 class problem

Given a binary classification problem: 

#### The likelihood is: 

$$ P(t \mid w) = \prod_{i=1}^{n} yn^{tn}(1-yn)^{1-tn}$$

so: 

$$yn = P(y=1 \mid a,s) = \sigma(w0 + w0a + w0s)$$

#### Error function: 

$$ E(w) = \ln P(t \mid w) = -\sum_{i=1}^{3} [tn \ln yn + (1-tn)\ln (1-yn)]$$

#### Maximum likelihood Solution: 

$$ w = \argmin E(w)$$ 

Iterative reweighted least squares. In a nutshell SGD but with Hessian Matrix 



# CNN

$$w_{out} = \frac{w_{in} - w_{k} + 2p}{s}+1 $$
$$h_{out} = \frac{h{in} - h_{k} + 2p}{s}+1 $$
$$ \mid \theta \mid = w_{k} \times h_{k} \times d_{in} \times d_{out} + d_{out} $$

Padding necessario: 
$$ \left\lfloor \frac{wk}{2} \right\rfloor $$

# ANN 
## Cost function

Model implicitly defines a conditional distribution $( p(t|x, \theta) )$

Cost function: Maximum likelihood principle (cross-entropy)

$$ J(\theta) = \mathbb{E}_{x,t \sim D} [-\ln(p(t|x, \theta))] $$

Example:
Assuming additive Gaussian noise we have

$$ p(t|x, \theta) = \mathcal{N}(t|f(x; \theta), \beta^{-1}I) $$

and hence

$$ J(\theta) = \mathbb{E}_{x,t \sim D} \left[ \frac{1}{2} \| t - f(x; \theta) \|^2 \right] $$

Maximum likelihood estimation with Gaussian noise corresponds to mean squared error minimization.

## Output units activation functions

### Regression

Linear units: Identity activation function
$$ y = W^T h + b $$

Use a Gaussian distribution noise model
$$ p(t|x) = \mathcal{N}(t|y, \beta^{-1}) $$

Loss function: maximum likelihood (cross-entropy) that is equivalent to minimizing mean squared error.

Note: linear units do not saturate.


### Binary classification

Output units: Sigmoid activation function
$$ y = \sigma(W^T h + b) $$

Loss function: Binary cross-entropy
$$ J(\theta) = \mathbb{E}_{x,t\sim D} [- \ln p(t|x)] $$

The likelihood corresponds to a Bernoulli distribution

Output unit saturates only when it gives the correct answer.


### Multi-class classification


Output units: Softmax activation functions
$$ y_i = \text{softmax}(\alpha^{(i)}) = \frac{\exp(\alpha^{(i)})}{\sum_j \exp(\alpha_j)} $$

Loss function: Categorical cross-entropy
$$ J_i(\theta) = \mathbb{E}_{x,t \sim D} [- \ln \text{softmax}(\alpha^{(i)})] $$

with $( \alpha^{(i)} = w_i^T h + b_i )$.

Likelihood corresponds to a Multinomial distribution

Output units saturate only when there are minimal errors.




# PCA 

### Express the points in M: 

$$ \overline{x}_{n} = \sum_{i=1}^{n} (x_{n}^T u_{i})u_{i}$$

### Intrinsic dimension: 

Minimum dimension to represent the dataset. 


# SVM
## SVM Classification

Maximum Likelihood Solution: 

$$ w^*, w0^* = argmin \frac{1}{2}||w||^2$$

subject to: 

$$ tn(w^Tx + w0) \geq 1$$

Classification new instance: 

$$y(x') = sign(\sum_{xk\in SV} a^*t_{k}x'^Txk + w0^*)$$


### Soft margins
Slack Variables in order to manage noise in the dataset: 

1. $\xi_{n}=0$ if point on or inside the correct margin boundary
2. $0<\xi_{n}\leq1$ if point inside the margin but correct side
3. $\xi_{n}>1$ if point on wrong side of the boundary

subject to soft margin constraint: 
$$ t_{n}y(x_{n}) \geq 1-\xi_{n}$$ 

Maximum Likelihood Solution: 
$$w^*, w0^* = \argmin \frac{1}{2}||w||^2 + C \sum_{n=1}^{N} \xi_{n}$$


## SVM Kernelized
$$
J(w, C) = C \sum_{i=1}^{N} L_{\epsilon}(t_i, y_i) + \frac{1}{2} \| w \|^2,
$$

$$
L_{\epsilon}(t, y) = 
\begin{cases} 
0 & \text{se } |t-y|<\epsilon \\
|t-y|-\epsilon & \text{altrimenti}
\end{cases}
$$


The main idea is to use a kernelized regression method computing the optimal solution without the Gramm Matrix that is computationally intensive. 
So we use di $\epsilon-insensitive$ error function. Is not differentiable

So we've introduced slack variables. 

$$ \xi^+ > 0 \lrArr tn>y(xn) + \epsilon $$

$$ \xi^- > 0 \lrArr tn<y(xn) - \epsilon $$

So the error function: 
$$
J(w, C) = C \sum_{i=1}^{N} (\xi^+ + \xi^-) + \frac{1}{2} \| w \|^2,
$$

# Least Squares

The linear model: 

$$ y(x) = W^Tx$$

1 of K coding scheme for t: 

$$ tk = 1 \rArr x \in Ck, tj = 0 \\s.t. \text{  } j \not ={k}$$


#### The error function
$$ E(w) = \frac{1}{2}Tr\{{(XW - T)^T(XW - T)}\} $$

The solution: 

$$ w = (X^T X)^{-1}X^T T$$


# Gaussian Mixture Model

Gaussian Mixture Model is defined as: 

$$ P(x \mid \pi, \mu, \Sigma) = \sum_{k=1}^{K} \pi_{k}N(x \mid \mu_{k}, \Sigma_{k})$$

### Maximum log-likelihood: 

$$\ln P(x \mid \pi, \mu, \Sigma) = \sum_{n=1}^{N} \ln(\sum_{k=1}^{K} \pi_{k}N(x \mid \mu_{k}, \Sigma_{k}))$$


### Expectation Maximization:

$$\pi_{k} = \frac{Nk}{N}$$
$$N_{k} = \sum_{n=1}^{N} \gamma(Z_{nk})$$
$$\mu_{k} = \frac{1}{Nk}\sum_{n=1}^{N} \gamma(Z_{nk}){x_{n}}$$
$$\Sigma_{k} = \frac{1}{Nk}\sum_{n=1}^{N} \gamma(Z_{nk})(xn - \mu_{k})(xn - \mu_{k})^T$$

where: 

$$\gamma(Z_{nk}) = \frac{\pi_{k} N(x\mid \mu_{k}, \Sigma_{k})}{\sum_{j=1}^{K} \pi_{j}N(x|\mu_{j}, \Sigma_{j})} $$


# Ensemble 

## AdaBoost

Given $( D = \{ (x_1, t_1), \ldots, (x_N, t_N) \} ), where ( x_n \in X, t_n \in { -1, +1 } )$

1. Initialize $( w_n^{(1)} = \frac{1}{N}, n = 1, \ldots, N. )$
2. For $( m = 1, \ldots, M )$:

   - Train a weak learner $( y_m(x) )$ by minimizing the weighted error function:
     $$
     J_m = \sum_{n=1}^{N} w_n^{(m)} I(y_m(x_n) \neq t_n),
     $$
     $I(e) = \begin{cases} 
     1 & \text{if } e \text{ is true} \\
     0 & \text{otherwise}
     \end{cases}$

   - Evaluate:
     $$
     \epsilon_m = \frac{\sum_{n=1}^{N} w_n^{(m)} I(y_m(x_n) \neq t_n)}{\sum_{n=1}^{N} w_n^{(m)}}
     $$
     and
     $$
     \alpha_m = \ln \left( \frac{1 - \epsilon_m}{\epsilon_m} \right)
     $$

   - Update the data weighting coefficients:
     $$
     w_n^{(m+1)} = w_n^{(m)} \exp[\alpha_m I(y_m(x_n) \neq t_n)]
     $$

