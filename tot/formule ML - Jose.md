# Machine learning

## Definition of a machine learning problem and its goal

The general Machine Learning problem, we have to approximate a function $f:X \rightarrow Y$, given a dataset $D$ containing informations about $f$.
The specific form of the datasets determines the category of ML problem that we want to solve.

## Probabilistic framework
<!-- https://dsp.stackexchange.com/questions/53128/why-is-random-noise-assumed-to-be-normally-distributed -->
A key thing that we have to keep in mind in the Machine Learning setting is the ineherently presence of noise.
When we try to model its probability distribution, using the Gaussian distribution is often a good choice.
Nontheless even if we model it with great accuracy, we still need to make predictions, thus approaching machine learning from a probabilistic perspective helps us quantify and manipulate uncerntainty directly.

## Supervised Learning

In supervised learning, the dataset $D$ comprises, of pairs of input output

$$D = \{<x,y>|x \in X, y \in Y \}$$

### Sample error

The sample error represents the error that we make on instances present in the training set.

$$ \text{error}_S(h) = \frac{1}{n} \sum_{x \in S} \delta(f(x) \not ={h(x)})$$

### True Error

We assume that different instances of $X$ presented to us in the dataset is drawn from an unknown distribution $\mathcal{D}$.
The true error is the error that the hypotesis make on any value choosen at random from $\mathcal{D}$.

$$ \text{error}_D(h) = \P_{x \in D}(f(x) \not ={h(x)})$$

The true error is impossible to compute, but we can estimate it.

### Overfitting

Given a hypotheses space $H$, a hypotheses $h \in H$, is said to overfit the training data if there exists some $h' \in H$, such that:

$$ \text{error}_S(h) < \text{error}_S(h') \cap \text{error}_{\mathcal{D}}(h) > \text{error}_{\mathcal{D}}(h') $$

Overfitting occurs when the hypotesis is too complex compared to the task it tries to solve.

### Overfitting Decision Trees

The depth of a tree controls it's complexity.
A decision tree that perfectly classify training examples could lead to overfitting when there is either noise or/and not enough samples in the dataset.
There are two possible approach this problem

1. Stop growing the tree before it starts to overfit. The decision tree stops generating nodes when there is no good attribute to split on.
2. Grow the full tree and then post prune using a statistical significant test.

The second approach is to be preferred, because in the first is possible that at one given point no particular attribute is the best, but there are a combination that are informative.

### Bayesian learning

The uncertainty is modeled in a bayesian framework.
This means that our beliefs are updated as soon as new data is presented to us.
The bayes theorem is as follows:

$$P(A|B) = \frac{P(B|A)P(A)}{P(D)}$$

$P(A|B)$ is the called posterior, beacuse is the updated belief about $A$ after taking into consideration the evidence, $B$.
$P(B|A)$ is the likelihood and express how well the observed data supports our hypotesis.
$P(A)$ is the prior, reflects the prior beliefs we had before evidence was presented to us.

### Maximum a Posteriori Hypothesis

Is usual that a particular learning algorithm returns not a single hypotesis but a set of hypotesis.
Our objective would be to determine the most probable hypotesis $h$ given the data at hand $D$.
In other word we would like to determine hypotesis that maximizes the posterior.

$$h_{MAP} = \underset{h \in H}{\text{argmax}} P(h | D) \\$$
$$ \stackrel{(1)}{=} \underset{h \in H}{\text{argmax}} \frac{P(D|h)P(h)}{P(D)} \\$$
$$ \stackrel{(2)}{=} \underset{h \in H}{\text{argmax}}  {P(D|h)P(h)}$$

(1) Is given by Bayes Theorem.
(2) Is given by the fact that $P(D)$ is a constant and $\text{argmax}$ is invariant to constant multiplication.

### Maximum Likelihood

Assuming we know the prior probability of each hypotesis $h$, we can determine the most probable hypotesis by computing the $h_{MAP}$.
However the knowledge of $P(h)$, might not be available, thus we have no reason to think that a particular hypotesis must be preferred over another, so we model it as a uniform distribution, thus $P(h)$ can be ignored, because it becomes a constant value.

$$h_{ML} =\underset{h \in H}{\text{argmax}} P(h \mid D) = \argmax P(D\mid h)$$

### Optimal Bayes Classifier

While the $h_{MAP}$ is the most probable hypotesis, given the data, it's classification might not be the most probable.
The Bayes Optimal Classifier on the other hand classify each instance with its most probable value.
The BOC, determines the most probable value by making a weighted sum of the probability of a specific value $v$, assuming that $h$ is true, weighted by the probability of $h$ being true given the data $D$.  

$$ v_{obc} = \underset{h \in H}{\text{argmax}} \sum_{hi \in H} P(v | h)P(h | D)$$

The BOC, takes it's name from the fact that under the same hypotesis space, and with the same a priori knowledge no other method outperforms it on average.
It is, however, not practical in real situations, due to its computationally intensive nature.

### Naive Bayes classifier

The Naive Bayes Classifier is a practical algorithm.
It can be used under the assumption that any value $v \in V$, where $V$ is a finite set, that we want to compute, can be expressed as a conjunction of its attributes.
Since every value can be described as a conjunction of its attributes, the naive Bayes algorithm determines $v_{map}$ in the following way:

$$ v_{MAP} = \underset{h \in H}{\text{argmax}}  P(v | a_1, a_2, .... , a_n) \\$$

$$ \stackrel{(1)}{=} \underset{h \in H}{\text{argmax}} P( a_1, a_2, .... , a_n | v) \\$$
$$ \stackrel{(2)}{=} \underset{h \in H}{\text{argmax}} P(v) \prod_i P(a_i | v) \\$$

(1) Comes from the combined application of Bayes Theorem and that $P(a_1 , \ldots , a_n)$ is a constant.
(2) Comes from the Independence assumption. The independence assumptions states that attributes values are conditionally independent given the target value.

### Naive Bayes as approximation of the OBC

Naive Bayes is considered an approximation of the Bayes optimal classifier because it simplifies the joint probability calculation by assuming conditional independence between features given the class.
While this assumption may not hold in every case, Naive Bayes remains a practical and effective classification algorithm, providing a computationally efficient way to approach the optimal Bayes classifier in situations where the independence assumption is reasonable.

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

### Linear Regression

Given:

$$ y(x,\bold{w}) = w_0 + w_1x_1 + ... + w_dx_d = \bold{w}^Tx$$

Due to noise, we can say that:

$$ t = y(x,\bold{w}) + \epsilon$$

The optimal value of $\bold{w}$ can be determined using a maximum likelihood approach.
Assuming $\epsilon$ is Gaussian error, and that observations are i.i.d we can define the likelihood function of $\bold{t}$ as:

$$ P( \bold{t} \mid \bold{x}, \bold{w}, \beta) = \prod_{i=1}^{n} N(t_i\mid y(x_i,\bold{w}), \beta^{-1})$$

Insted of maximizing this quantity we can maximize the log likelihood:

$$ \underset{\bold{w}}{\text{argmax}}( \log( P( \bold{t} \mid \bold{x}, w, \beta)))  \\$$
$$= \underset{\bold{w}}{\text{argmax}} (\frac{N}{2} \log {\beta} - \frac{N}{2}\log (2\pi) - \beta E_D(\bold{w}))$$

$$ \stackrel{(1)}{=} \underset{\bold{w}}{\text{argmax}} (E_D(\bold{w}))$$
$$ \stackrel{(2)}{=} \underset{\bold{w}}{\text{argmin}} (E_D(\bold{w}))$$

(1) Comes from the fact that argmax is invariant to constant addition and multiplication.
(2) Come from the fact that maximization of the log likelihood is equivalent to the minimization of negative log likelihood.
Where

$$ E_d(\bold{w}) = \frac{1}{2} \sum_{i=1}^{N} ({t_i - \bold{w}^Tx_i})^2$$

In order to find the minimum we need to differentiate:

$$ \nabla E_d(\bold{w}) = \sum_{i=1}^{N} (t_i - \bold{w}^T x_i)x_i^T$$

setting to zero:

$$ 0 = \sum_{i=1}^{N} t_ix_i^T - \bold{w}^T (\sum_{i=1}^{N}{x_{i}x_{i}^T})$$

Gives us the maximum likely solution, which can be written in closed form.

$$\bold{w}_{ML} = (\bold{X}^T\bold{X})^{-1} \bold{X}^T \bold{t} $$

Determining the optimal values using the closed form can be costly, due to the fact that, it needs to process the entire dataset.
We can define an iterative approach, by updating the weights based on a subset (mini batch) or only a single element (sequential) through the following learning rule.

$$ \bold{w} ^{\tau} \leftarrow \bold{w}^{\tau - 1} - \eta \nabla{E_n}$$

Where $E_n$ is the error averaged over the subset.

<!-- Checkpoint -->
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

Typical Question:

$y(x) = \sum_{i=1}^{n} \alpha_{i}x_{i}^Tx$

Now, if a vector input x appears in an algorithm only in the form of inner product we can replace the inner product with some kernel k(x,x'), where:

$k$ is a kernel function defined as a real-valued function k(x,x') where x,x' $\in$ X. A similarity measure between instances x and x'.

Tipically simmetryc and non negative, **BUT NOT STRICTLY REQUIRED**

After kernel trick we obtain:

$y(x) = \sum_{i=1}^{n} \alpha_{i}k(x_{n},x)$

### Kernel Trick

The "kernel trick" is a technique used in machine learning algorithms that allows operations in high-dimensional spaces without explicitly computing the transformations of data into those spaces. It leverages kernel functions to compute the dot product (a measure of similarity) between vectors in a higher-dimensional space, enabling linear algorithms to solve non-linear problems. This makes it possible, for example, to separate classes of data that aren't linearly separable in the original space.

## Unsupervised learning

### K-NN

Given a learning function and a labeled Dataset D, for a new instance x':

1. Find K-Nearest neighbors of new instance x'
2. Assign to x' the most common label among the k-nearest neighbors to x'

Likelihood of Class C for a new instance x':

$$ P(C\midx,D, K) = \frac{1}{k} \sum_{xn \in Nk(xn,D)} I(tn=C)$$

## Neural Networks

### Backpropagation

Forward step

(![Alt text](12-ANN_2p.jpeg))

Backward step

![Alt text](12-ANN_3-1.jpeg)

### SGD

![Alt text](image.png)

# K-Means

1. Set a K Value
2. Take K single clusters and assigns N-K samples to them based on distance between centroids and point. After each assignment recompute the centroid
3. Take each sample and compute its distance from the centroid of the clusters, if the samples it is not in the centroid closest clusters switch it. Recompute the centroid
4. Repeat 3 until convergence

### Logistic Regression 2 class problem

Given a binary classification problem:

#### The likelihood is

$$ P(t \mid w) = \prod_{i=1}^{n} yn^{tn}(1-yn)^{1-tn}$$

so:

$$yn = P(y=1 \mid a,s) = \sigma(w0 + w0a + w0s)$$

#### Error function

$$ E(w) = \ln P(t \mid w) = -\sum_{i=1}^{3} [tn \ln yn + (1-tn)\ln (1-yn)]$$

#### Maximum likelihood Solution

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

Model implicitly defines a conditional distribution $( p(t \mid x, \theta) )$

Cost function: Maximum likelihood principle (cross-entropy)

$$ J(\theta) = \mathbb{E}_{x,t \sim D} [-\ln(p(t \mid x, \theta))] $$

Example:
Assuming additive Gaussian noise we have

$$ p(t \mid x, \theta) = \mathcal{N}(t \mid f(x; \theta), \beta^{-1}I) $$

and hence

$$ J(\theta) = \mathbb{E}_{x,t \sim D} \left[ \frac{1}{2} \ \mid  t - f(x; \theta) \ \mid ^2 \right] $$

Maximum likelihood estimation with Gaussian noise corresponds to mean squared error minimization.

## Output units activation functions

### Regression

Linear units: Identity activation function
$$ y = W^T h + b $$

Use a Gaussian distribution noise model
$$ p(t \mid x) = \mathcal{N}(t \mid y, \beta^{-1}) $$

Loss function: maximum likelihood (cross-entropy) that is equivalent to minimizing mean squared error.

Note: linear units do not saturate.

### Binary classification

Output units: Sigmoid activation function
$$ y = \sigma(W^T h + b) $$

Loss function: Binary cross-entropy
$$ J(\theta) = \mathbb{E}_{x,t\sim D} [- \ln p(t \mid x)] $$

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

# Autoencoder

What is an autoencoder?

1. A combination of two NN, an encoder and a decoder.
2. trained based on reconstruction loss
3. provides low dimensional representation
4. Bottleneck concept, which learn to reconstruct input minimizing a loss function
5. Autoencoders can be seen as a method for non-linear principal component analysis

# PCA

### Principali Usi

1. Dimensionality reduction
2. data compression
3. Data visualization
4. Feature Extraction

### Express the points in M

$$ \overline{x}_{n} = \sum_{i=1}^{n} (x_{n}^T u_{i})u_{i}$$

### Intrinsic dimension

Minimum dimension to represent the dataset.

### Goal: Maximize data variance after projection to some direction u1

Projected Points:
$x_{n}^Tu1$

Anzitutto fissiamo $\overline{x}$ come la media del nostro dataset, centriamo dunque il dataset sulla nostra media in maniera tale che esso abbiamo media 0.
A questo punto passiamo alla fase di massimizzazione della varianza calcolando la varianza come:

$$ \frac{1}{N} \sum_{n=1}^N [u_{1}^Tx_{n} - u_{1}^T\overline{x}]^2 = u_1^TSu_1$$

Il problema da risolvere diventa:

$$\max u_1^TSu_1$$

Da cui massimizzando e settando la derivata rispetto ad u1 a 0:

$$Su_1 = \lambda u_1$$
$$u_1^TSu_1 = \lambda_1$$

Chiamata first principal component.

S = matrice di Covarianza

# SVM

## SVM Classification

Maximum Likelihood Solution:

$$ w^*, w0^* = argmin \frac{1}{2} || w ||^2$$

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

### Maximum log-likelihood

$$\ln P(x \mid \pi, \mu, \Sigma) = \sum_{n=1}^{N} \ln(\sum_{k=1}^{K} \pi_{k}N(x \mid \mu_{k}, \Sigma_{k}))$$

### Expectation Maximization

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

# Perceptron

Combinazione lineare pesata dell'input

$$
o(x_1, \ldots, x_d) =
\begin{cases}
1 & \text{if } w_0 + w_1x_1 + \ldots + w_dx_d > 0 \\
-1 & \text{otherwise}
\end{cases}
$$

$$
o(\mathbf{x}) =
\begin{cases}
1 & \text{if } \mathbf{w}^T\mathbf{x} > 0 \\
-1 & \text{otherwise}
\end{cases}
= \text{sign}(\mathbf{w}^T\mathbf{x})
$$

## Error Function

$$E(W)=\frac{1}{2} \sum_{n=1}^{N}(tn-w^Tx_{n})^2$$

calcola i parametri attraverso SGD.

# Generative vs Discriminant Model

A generative model learn the class conditional densities of each class $P(x\mid Ck)$ trying to understand the joint distribution of the data.

Once we have this quantity we can easily compute the posterior $P(Ck \mid x)$ using Bayes Rule.

On the other hand a discriminative model compute directly P(Ck \mid x).

So we can define a generative model as:

$$P(Ck\mid x) = \frac{P(x \mid Ck)P(Ck)}{\sum_{j}P(x \mid Cj)P(Cj)} = \frac{exp(ak)}{\sum_{j}exp(aj)}$$

$a_{k} = \ln P(x \mid Ck)P(Ck)$

On the other hand a **discriminative model**:

$$P(Ck\mid x) =  \frac{exp(ak)}{\sum_{j}exp(aj)}$$

Maximum log-likelihood solution:

$$ w^* = argmax \ln P(t\mid w, x)$$

Note:
likelihood generative 2 classi utile per esercizio:
$$P(t \mid \pi_1, \mu_1, \mu_2, \Sigma, D) = \prod_{n=1}^{N}[\pi_1N(x_n \mid u_1, \Sigma)]^{t_a}[(1-\pi_1)N(x_n \mid u_2, \Sigma)]^{1-t_a}$$
