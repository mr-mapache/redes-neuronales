Habíamos visto que una dada una señal $\vec{s}$ una red neuronal podía aprender 

Supongamos ahora no una señal de activación sino una **secuencia** de longitud $l$ de señales $d$-dimensionales. 


$$
S = \begin{bmatrix} \vec{s}^1 \\ \vec{s}^2 \\ \vdots \\ \vec{s}^l \end{bmatrix} 
$$












### Multihead attention

  

In the transformers model, the attention mechanism is applied in parallel to multiple projections of the queries, keys and values. Each projection is called an "attention head". To define these projections, three weight matrices $W^Q$, $W^K$ and $W^V$ are used that are applied to the queries, keys and values respectively.

  

Let:

  

- $W^Q \in \mathbb{R}^{d \times d_q}$

- $W^K \in \mathbb{R}^{d \times d_k}$

- $W^V \in \mathbb{R}^{d \times d_v}$

  
  

With $d_q = d_k$. Given a tensor $X \in \mathbb{R}^{l \times d}$, we say that the products:

  

- $X W^Q \in \mathbb{R}^{l \times d_k}$

- $X W^K  \in \mathbb{R}^{l \times d_k}$

- $X W^VX  \in \mathbb{R}^{l \times d_v}$

  

Are the projections of the tensor $X$ in the query, key and value spaces respectively. We can then define the multi-head attention mechanism as:

  

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \cdots, \text{head}_h) W^O $$

$$ \text{head}_i = \text{Attention}(Q W^Q_i, K W^K_i, V W^V_i) $$

  
  

With $Q W^Q_i$, $K W^K_i$ and $V W^V_i$ the projections of the tensors $Q$, $K$ and $V$ in the query, key and value spaces respectively, for a head $\text{head}_i$, and $W^O$ is another transformation that is applied to the result of concatenating the outputs of each head. These transformations are responsible for generating the different "heads" from the queries, keys and original values.

  

Although in the definition of the multi-head attention mechanism layer, different views are generated for the input tensors $Q$, $K$ and $V$, in practice, it is simpler and computationally more efficient to generate a single projection of these tensors and then divide them into $h$ parts, so that the matrices $Q_i$, $K_i$ and $V_i$ are generated for each head $i$. This can be achieved as follows:

  

Given a projection $P \in \mathbb{R}^{l \times d}$, either $P = W^Q Q, W^K K$ or $W^V V$ we can divide each row of $P$ into $h$ parts of dimension $d/h$ and then group the vectors of each part into a matrix of dimension $l \times d/h$ in the same tensor by adding a dimension as follows:

  
  

$$ P = \begin{bmatrix}

  

    p^1_1 & p^1_2 & \cdots & p^1_d  \\

    p^2_1 & p^2_2 & \cdots & p^2_d   \\

    \vdots & \vdots & \ddots  & \vdots \\

    p^l_1 & p^l_2 & \cdots & p^l_d  \\

  

\end{bmatrix} \rightarrow \begin{bmatrix}

  

    \begin{bmatrix}

  

        p^1_1 & \cdots & p^1_{d/h}  \\

  

    \vdots & \vdots & \ddots  & \vdots \\

  
  

        p^1_{d\frac{(h-1)}{h}+1} &  \cdots & p^1_d  \\

  

    \end{bmatrix} \\

  

    \vdots \\

  

    \begin{bmatrix}

  

        p^l_1 & \cdots & p^l_{d/h}  \\

  

    \vdots & \ddots  & \vdots \\

  
  

        p^l_{d\frac{(h-1)}{h}+1} & \cdots & p^l_d  \\

  

    \end{bmatrix} \\

  

\end{bmatrix} \rightarrow \begin{bmatrix}

  

    \begin{bmatrix}

  

        p^1_1 & p^1_2 & \cdots & p^1_{d/h}  \\

  

    \vdots & \vdots & \ddots  & \vdots \\

  
  

        p^l_1 & p^l_2 & \cdots & p^l_{d/h}  \\

  

    \end{bmatrix} \\

  

    \vdots \\

  

    \begin{bmatrix}

  

        p^1_{d\frac{(h-1)}{h}+1}  & \cdots & p^1_d  \\

  

    \vdots & \vdots & \vdots \\

  
  

        p^l_{d\frac{(h-1)}{h}+1} & \cdots & p^l_d  \\

  

    \end{bmatrix} \\

  

\end{bmatrix}

  

$$

  

Where the first matrix is the first head, the second matrix is the second head and so on. The final result is a tensor of dimension $h \times l \times d/h$.

  

The concatenation of the outputs of each head is done in the dimension $d/h$ and is the inverse process to the one described for the "split" so that the final result is a tensor of dimension $l \times d_v$.

  

Finally, the output is multiplied by the matrix $W^O \in \mathbb{R}^{d_v \times d}$ to obtain the final result of the multi-head attention layer, which will have dimension $l \times d$.