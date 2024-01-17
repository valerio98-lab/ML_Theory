# Reinforcment Learning

## Problem definition

A reinforcement learning problem is a special type of machine learning problem.
We have an agent and and environment, in a dynamic system.
The agent interacts with the environment by choosing and action, to which the environment responds changing the state.
Our goal is to determine an optimal policy $\pi : X \rightarrow A$ given a dataset $D = \{<x_1,a_1,r_1, \ldots x_n,a_n,r_n>^{i} \} $ composed by a sequence of states, action and rewards.

## Markov property

An assumption made in RL is the markov assumption, in other words we assume that the dynamic system does not depend on the history of states, observations, and actions as long as the current state is known.

## MDP

An MDP is process for decision making.
We assume that the states are fully observable, and we can act directly on them.
Since the states are observable all the information that we need are contained into them.
An MDP can be described ad a tuple:

$$\text{MDP} = <\bold{X},\bold{A},\delta,r>$$

$\bold{X}$ is the set of possibles states.$\bold{A}$ is the set of possibles actions. $\delta$ is a transition model and describes how the system evolve after executing each action. $r$ is a reward function, after each action the system recieves a numeric feedback signaling either a good or bad choice of action.

## Definition of K-Bandit problem
<!-- https://ai.stackexchange.com/questions/27694/what-are-the-major-differences-between-multi-armed-bandits-and-the-other-well-kn#:~:text=Multi%2DArmed%20Bandit%20is%20used%20as%20an%20introductory%20problem%20to%20reinforcement%20learning%2C%20because%20it%20illustrates%20some%20basic%20concepts%20in%20the%20field%3A%20exploration%2Dexploitation%20tradeoff%2C%20policy%2C -->
The K-bandit problem is a subclass of reinforcement learning problems. The problem goes as follows:

**"You are faced to choose between $k$ options, after you choose an option you recieve a reward, the goal is the maximize the expected reward over a period of time"**

The K-bandit problem is also called one-state MDP because you have only one state.
The importance of the k-armed bandit is due to the fact that illustrates some pillar concepts of reinforcement learning, such as the exploration-exploitation trade-off, the concept of optimal policy and more.

In a K-Bandit problem, we have no information about the rewards.
Thus in order to get the optimal policy we need to learn (gather informations about $r$ through simulation), determine an optimal policy and only then we can make decisions.

Depending on how the reward function behaves, and the type of informations about it we can define 4 solutions:

- If the rewards are deterministic, and known, the optimal solution is just picking always the action that yields the highest reward. No iterations are needed, we can skip the learning and go directly to making decisions.
- If the rewards are deterministic, but unknown, we need to test each action at least one time, in order to discover the value, thus we need at least $|A|$ iterations.
- If the rewards are non-deterministic, but known, the optimal solution can be found by picking the action which has the highest mean.
- If the rewards are non-deterministic and unknown, we have to iterate through
  - Initialize a data structure $\Theta$ that will gather informations about the reward.
    - For each time $t$, until termination:
      - Choose an action
      - Execute the choosen action
      - Collect the reward
      - Update $\Theta$
    - Choose an optimal policy $\pi^*$ according to $\Theta$

## Q-Learning

Is an algorithm to determine the optimal policy.
We define the Q-value as $Q(x,a)$ as the expected cumulative reward the agent will recieve by taking the action $a$ in state $x$ and following the policy there after.
For each state action pair, the Q-learning algorithm mantains a Q-value and updates iteratively.
The pseudo code is the following:

- For each state action pair initialize a table entry in the Q table.
- For each time:
- - Observe the state $x$.
  - Choose and action $a$.
  - Execute the action choosen.
  - Observe the new state.
  - Collect the reward.
  - Update the table relative to $x$ , $a$ with the new reward.
  - Move to the new state.
- Choose the oprimal policy according to the Q table.

If the reward functions are non deterministic or unknown the Q-value is just an estimate.
Our objective is to maximize the reward.
At any fiven time there must be an action that has the highest reward, this action is called greedy.
We can alwayas choose the action that yields the best result (exploitation).
However we might its also possible that by choosing some suboptimal action, the overall policy is better (exploration).
At any given time you cannot choose both exploration and exploitation thus you have a trade off.
An approach that can be embedded into the Q-learning algorithm, is that when choosing the action, we choose with a probability of $1-\epsilon$ greedy action and with a probability of $\epsilon$ a random action choosen uniformly.

## HMM

The Hidden Markov Models are a formulation of a Markov Process in which we assume that we cannot control the evolution of the system .
We assume that a state generates an event with a certain probability, we can observe the event but we cannot observe the state.

$$\text{HMM} = <\bold{X},\bold{Z}, \pi_0>$$

$\bold{X}$ is the set of states, which are not observable but we know that exists.
$\bold{Z}$ is the set of observations of the events generated by the states.
The transition model is described by $P(\bold{x_t}|\bold{x_{t-1}})$, meaning that the current state depends on the previous state.
The observation modle instead is $P(\bold{z_t}|\bold{x_{t-1}})$, meaning that the current observation depends from the previous state.
Last $\pi_0$ describes the probability of starting at exactly $x_0, x_0 \in \bold{X}$. Our goal is to reconstruct the states from the observations.

## POMDP

Partially Observable Markov Decision Process are a union of the elements of both MDP and HMM.
As in the HMM we don't assume a full observability of the states, however as in the MDP we can act on the environment.

$$\text{POMDP}= <\bold{X},\bold{A},\bold{Z}, \delta, r, o>$$

$\bold{X}$ is the set of states, which are not observable but we know that exists.
$\bold{Z}$ is the set of observations of the events generated by the states.
We have a probability distribution describing the probability of starting at state $x_0, x_0 \in \bold{X}$.
We can act on the environment by choosing an action in $A$.
The transition function $\delta$ is a probability distribution $P(x'|x,a)$ that describes the probability of ending up on $x'$ after executing $a$ on $x$.
$o$ is a probability distribution over distributions, $P(z'|x',a)$  describes the probability of observing $z'$, after we end up on $x'$ by the execution of $a$
