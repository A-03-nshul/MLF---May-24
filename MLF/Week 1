# Lecture 1: Introduction to Machine Learning - ML

### What is Machine Learning - ML?

The study of computer algorithms that **_Improve automatically_** through experience and by use of Data.

### What is a Task?

As we climb up the Task Hierarchy, the level of **Abstraction** increases, while Manual Labor decreases.

[Task Hierarchy .pdf](https://prod-files-secure.s3.us-west-2.amazonaws.com/a4b270fa-783b-4688-b76b-95e9569085ed/cb390cad-850d-4fe0-8980-7f162dc8e3aa/Task_Hierarchy_.pdf)

### Why even use ML?

We actually shouldn’t, if a task can be completed efficiently and sufficiently via Manual Labor or via Tool Usage - Programming, as ML is quite complicated.

**Example: Password Authentication**

Can very easily be achieved via Programming, so no need to involve ML here.

---

That said, they are quite prone to failure because:

- Speed, Scale & Cost of Human labor
- Inability to express **_Rules_** via language
- Sometimes, we don’t even know the exact rules transforming inputs to outputs.

ML may succeed in these areas because:

- It has huge _Example Data_ to refer to - to make (or figure out) its own rules
- It has a Structured idea on rules

**Example: Face Detection**

Now this requires ML.

Humans can tell a program what is or isn’t a face, with the help of multiple _Sample Images._ After that, the program can figure out its own **_Rules_** to produce desired Output for the Test Samples.

### Where is ML being used?

- **E-Mail Service**
  Email services these days provide In-Built Spam Filters. This is so successful only because of the intervention of ML.
- **Recommendation Systems**
  Like in E-Marketplace Shopping carts, or on OTT Platforms.
- **Voice Assistants**
  To accurately recognize the words being spoken to them - the commands being issued.

### What is Data?

In the most general sense:

<aside>
👉 It is a Collection of Bits or Bytes.

</aside>

It is a collection of Bits or Bytes.

In the context of ML, however:

<aside>
👉 Data is a **Collection of Vectors**

> METADATA is information on the Data
>
> A Computer is perfectly fine with just the data - It doesn’t need the MetaData. That is solely for the understanding of the Humans interacting with that Data, since unclassified vectors mean nothing to us.

</aside>

### What is a Model?

In the Scientific sense, A model is a **Mathematical Representation** (read Simplification) of **reality**.

In the context of ML though, we have 2 categories of Models:

- PREDICTIVE MODEL
- PROBABILISTIC MODEL
  ### PREDICTIVE MODELS:
  Their goal is to “Predict” something. The 2 popular models are:
  1. Regression
  2. Classification
     **REGRESSION:** Results in a _Continuous Numerical Value, that can take any range._
     **Example**: Model the price of a house based on Area and Distance form the metro.
     The Model:
     $$
     Price = 0.5*Area -Distance
     $$
     **CLASSIFICATION:** Results in a _Boolean_ type value - whether YES or NO for whatever is being Modeled
     **Example:** Model whether a house is less than 2km away from the Metro based on Price and Area (or Rooms).
     The Model:
     $$
     Distance = 2*Rooms - Price
     $$
     Thus, the house is:
  - Closer than 2km, if _Distance_ < 1
  - Farther, Otherwise
  ***
  ### PROBABILISTIC MODELS:
  Their goal is to evaluate the likeliness of an Event.

### What is a Learning Algorithm?

These are the tools that convert Data into Models.

They choose from a variety of models, with same structure but different parameters, to find the Best fit for the situation.

### Supervised Learning

**Notations** to be used throughout the course:

For the set of real numbers

$$
\reals
$$

d-dimensional Vector of reals

$$
\reals^d
$$

**x** : Vector

j-th coordinate of vector **x**

$$
x_j
$$

|| **x || :** Length of Vector **x**

Collection of **n** vectors

$$
x^1, x^2,...,x^n
$$

j-th coordinate of i-th vector **x**

$$
x_j ^ i
$$

j-th coordinate of vector **x** raised to **n**

$$
(x_j)^n
$$

Indicator:

- **1** ( Criteria) = 1 ⇒ Criteria is **TRUE**
- **1** ( Criteria) = 0 ⇒ Criteria is **FALSE**

**REGRESSION:**

In the simplest sense, it is **_Curve Fitting_**

Given

$$
\{(x^1, y^1), (x^2, y^2), (x^3,y^3),...(x^n, y^n)\}
$$

Model **_f_**

$$
\exists f(x^i) \to y^i
$$

Let

$$
x^i \in \reals^d
$$

$$
y^i \in \reals
$$

Then

$$
f: \reals^d \to \reals
$$

**Example:**

Training Data

$$
\{(x^1, y^1), (x^2, y^2), (x^3,y^3),...(x^n, y^n)\}
$$

$$
x^i \in \reals^d
$$

$$
y^i \in \reals
$$

$$
f: \reals^d \to \reals
$$

**Loss :** How far f(x) is from y - It should be **_minimized_** for Best results

$$
\frac{1}{n} * \sum \limits_{i=1}^n [f(x^i) - y^i]^2
$$

We need to measure loss, because we need

But this isn’t always possible, so we look for the “Next Best Thing”, which is bound to have some Losses.

$$
\exists f(x^i) \to y^i
$$

The Linear Regression Model:

$$
f(x) = w^\intercal x + b = \sum \limits_{j=1}^n (w_jx_j) +b
$$

The Labels **y** will be Numerical values that belong to real numbers.

**CLASSIFICATION:**

**Example:** Predict if Rooms in the house are > 3, based on Price and Area

The Labels **y** will be **+1 / -1**, for True /False.

> NOTICE THE DIFFERENCE IN TRAINING DATA USED FOR REGRESSION & CLASSIFICATION!

Let:

$$
x^i \in \reals^d
$$

$$
y^i \in \{+1, -1\}
$$

Algorithm Yields:

$$
f: \reals^d \to \{+1, -1\}
$$

**Loss**: Fraction of Misclassified Instances

$$
\frac{1}{n} * \sum \limits_{i=1}^n 1(f(x^i) \neq y^i)
$$

The Classification Model:

$$
f(x) =sign( w^\intercal x + b  )
$$

> The ‘sign’ is called Linear Separator.

### Evaluating Learned Models

The Learning Algorithm uses Training Data to get **_f_**

**_f_** mustn’t be tested using the same Training Data.

**We use _TEST DATA_ not included in the Training Data, for Model Evaluation.**

### Model Selection

Learning Algorithms just find the Best Model from the Collection provided to them.

**This poses the question: How to find the Right Collection?**

We do this using another, **_DISTINCT_** Subset of Data - **_VALIDATION DATA_**

> This step is basically choosing the Correct parameters, **w** and **b**.

### Unsupervised Learning

This has more to do with “_Understanding Data”._

Data :

$$
\{x^1,x^2,...,x^n\}
$$

$$
x^i \in \reals^d
$$

Build Models to _Compress, Explain & Group Data._

<aside>
👉 THIS IS MUCH MORE VAGUE THAN **SUPERVISED LEARNING.**

</aside>

**DIMENSIONALITY REDUCTION:** Compression and Simplification

**Example:** Represent a Million gene expression levels for 1 million people, using just 100 numbers per person.

Thus, we need to compress:

$$
(10^6)^6 \to (10^2)^6
$$

Encoder:

$$
f: \reals^d \to \reals^k \forall d>>k
$$

Decoder:

$$
g: \reals^k \to \reals^d \forall d>>k
$$

**Loss:**

$$
\frac{1}{n} * \sum \limits_{i=1}^n ||g(f(x^i)) - x^i||^2
$$

**DENSITY ESTIMATION:**

Output is a Probabilistic Model - A Scoring Function such that **Total score = 1.**

Data:

$$
\{x^1,x^2,...,x^n\}
$$

$$
x^i \in \reals^d
$$

Probability Mapping:

$$
P:\R^d \to \R_+
$$

Goal:

- **P(x)** is large if **x** is in the Data
- Low, Otherwise

Loss: Negative Log Likelihood

$$
Loss=\frac{1}{n}\sum\limits_{i=1}^n(-\log(P(x^i))
$$

For each Data Point **x**, we want **P(x)** to be as large as possible.

---

## -ANSHUL
