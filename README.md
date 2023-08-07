# plato

A very naive search engine for [Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/index.html)
## Query Likelihood Model with Jelinek-Mercer Smoothing
The model can be derived from negative KL-divergence:

$$\displaystyle\begin{align}
rank(d, q) &:= -D_{KL}(P(X \mid Q = q) \parallel P(X \mid D = d)) \\
&= -\sum_{w \in \mathcal{X}}P(X = w \mid Q = q) \log\frac{P(X = w \mid Q = q)}{P(X = w \mid D = d)} \\
&\equiv \sum_{w \in \mathcal{X}}P(X = w \mid Q = q) \log {P(X = w \mid D = d)} \\
&\approx \sum_{w \in \mathcal{X}}\frac{c(w, q)}{\vert q \vert} \log \left[(1 - \lambda)\frac{c(w, d)}{\vert d \vert} + \lambda \frac{c(w, C)}{\vert C \vert}\right] \\
&\equiv \sum_{w \in \mathcal{X}} c(w, q) \log \left[(1 - \lambda)\frac{c(w, d)}{\vert d \vert} + \lambda \frac{c(w, C)}{\vert C \vert}\right] \\
&= \sum_{w \in \mathcal{X}} c(w, q) \log\left[\left(\frac{1 - \lambda}{\lambda}\frac{c(w, d) \vert C \vert}{c(w, C) \vert d \vert} + 1\right) \cdot \lambda \frac{c(w, C)}{\vert C \vert}\right] \\
&= \sum_{w \in \mathcal{X}} c(w, q) \log\left(\frac{1 - \lambda}{\lambda}\frac{c(w, d) \vert C \vert}{c(w, C) \vert d \vert} + 1\right) + \sum_{w \in \mathcal{X}} c(w, q) \log\left(\lambda \frac{c(w, C)}{\vert C \vert}\right) \\
&\equiv \sum_{w \in \mathcal{X}} c(w, q) \log\left(\frac{1 - \lambda}{\lambda}\frac{c(w, d) \vert C \vert}{c(w, C) \vert d \vert} + 1\right) \\
\end{align}$$

The $\displaystyle\frac{c(w, d)}{c(w, C) \vert d \vert}$ term corresponds to TF-IDF and document length normalization.

## Example
```
python3 query.py

Please enter: natural language semantics

https://plato.stanford.edu/entries/natural-language-ontology/ (28.6821)
https://plato.stanford.edu/entries/montague-semantics/ (31.5394)
https://plato.stanford.edu/entries/word-meaning/ (32.6054)
https://plato.stanford.edu/entries/information-semantic/ (32.8631)
https://plato.stanford.edu/entries/compositionality/ (32.8701)
https://plato.stanford.edu/entries/meaning/ (33.0329)
https://plato.stanford.edu/entries/situations-semantics/ (33.0506)
https://plato.stanford.edu/entries/idiolects/ (33.147)
https://plato.stanford.edu/entries/tarski-truth/ (33.172)
https://plato.stanford.edu/entries/linguistics/ (33.3995)
https://plato.stanford.edu/entries/generalized-quantifiers/ (33.4796)
https://plato.stanford.edu/entries/dynamic-semantics/ (33.4993)
https://plato.stanford.edu/entries/games-abstraction/ (33.4996)
https://plato.stanford.edu/entries/logic-if/ (33.5564)
https://plato.stanford.edu/entries/computational-linguistics/ (33.6067)
https://plato.stanford.edu/entries/proof-theoretic-semantics/ (33.6346)
https://plato.stanford.edu/entries/logic-algebraic-propositional/ (33.703)
https://plato.stanford.edu/entries/language-thought/ (33.7869)
https://plato.stanford.edu/entries/two-dimensional-semantics/ (34.0089)
https://plato.stanford.edu/entries/logic-classical/ (34.0245)
```
