One accelerator has a peak dense bf16 throughput of $P = 312$ TFLOP/s ($312\times10^{12}$ FLOP/s), an HBM bandwidth of $BW = 1.5$ TB/s ($1.5\times10^{12}$ bytes/s), and $40$ GB of HBM ($1$ GB $=10^9$ bytes throughout this page). Two or more such accelerators are connected by an interconnect that sustains $B_{\text{net}} = 400$ GB/s of send traffic and, independently and at the same time, $400$ GB/s of receive traffic between any pair of devices — a full-duplex link, so a device that is simultaneously sending and receiving is not throttled by sharing one budget. Every tensor in this problem — a weight, an activation, or a value in flight on the interconnect — is stored in bf16, $s = 2$ bytes/element, unless a part says otherwise.

For a matrix multiply $C = AB$ with $A \in \mathbb{R}^{m\times k}$, $B \in \mathbb{R}^{k\times n}$, $C \in \mathbb{R}^{m\times n}$:

- *FLOPs*: counting one multiply and one add as two FLOPs, $\text{FLOPs} = 2mkn$.
- *Bytes moved*: reading $A$ and $B$ from HBM once each and writing $C$ once, at $s$ bytes/element, $\text{bytes} = s(mk+kn+mn)$. Neither operand is re-read and nothing beyond $C$ is written.
- *Arithmetic intensity* $I = \text{FLOPs}/\text{bytes}$, in FLOP/byte.
- The *ridge point* $I^\star = P/BW$ is the intensity at which compute time and transfer time are equal. A matmul with $I > I^\star$ is *compute-bound*; one with $I < I^\star$ is *memory-bound*.
- *Predicted time* $t = \max(\text{FLOPs}/P,\ \text{bytes}/BW)$, assuming compute and HBM traffic fully overlap inside one kernel.

### Part 1 — Four matmuls

- Shape: A · $m$: 8 · $k$: 4,096 · $n$: 4,096
- Shape: B · $m$: 2,048 · $k$: 4,096 · $n$: 4,096
- Shape: C · $m$: 4,096 · $k$: 64 · $n$: 4,096
- Shape: D · $m$: 4,096 · $k$: 4,096 · $n$: 4,096

For each shape, give FLOPs, bytes moved, arithmetic intensity, the bound regime, and the predicted time. Then, for a general $(m,k,n)$, say whether growing $m$, $k$ or $n$ alone — holding the other two fixed — can ever move a matmul from compute-bound to memory-bound, or only ever the other way, and justify the answer for all three variables, not only the two that shapes A–D exercise directly.

### Part 2 — A 72-layer stack

A network has $L = 72$ layers, each a separate kernel with no fusion between layers. Layer $\ell$ takes an input $X_\ell$ and a weight $W_\ell$ and produces $X_{\ell+1}$; there is no attention, no normalisation, no residual connection and no bias, so $X_{\ell+1}$ is $X_\ell$'s only successor and $X_\ell$'s only source is $X_{\ell-1}$ (or, for $\ell = 1$, the network's own input). $X_{\ell+1}$ is written to HBM by layer $\ell$ and read back from HBM by layer $\ell + 1$, exactly as an ordinary matmul's output and next input. Layers alternate between two shapes, both built from $T = 20{,}480$ (the number of tokens processed together in this pass), $d = 8{,}192$ and $f = 32{,}768$:

- Odd $\ell$ (*type U*): $X_\ell \in \mathbb{R}^{T\times d}$, $W_\ell \in \mathbb{R}^{d\times f}$, $X_{\ell+1} = X_\ell W_\ell \in \mathbb{R}^{T\times f}$.
- Even $\ell$ (*type D*): $X_\ell \in \mathbb{R}^{T\times f}$, $W_\ell \in \mathbb{R}^{f\times d}$, $X_{\ell+1} = X_\ell W_\ell \in \mathbb{R}^{T\times d}$.

(A pointwise nonlinearity may sit between $X_\ell W_\ell$ and $X_{\ell+1}$; it is not a matmul and its cost is not counted.) All 72 weights are resident on one accelerator for the whole pass. Give the total compute time (every layer's FLOPs, divided by $P$) and total transfer time (every layer's bytes, divided by $BW$) for one forward pass over all 72 layers. Then determine exactly which tensors must be resident at once to run this pass, whether their total fits in the accelerator's 40 GB, and by how much it does or does not.

### Part 3 — Two devices, one input

The same 72-layer stack processes one $T = 20{,}480$-token input, now split across two identical accelerators of the kind above, connected by the interconnect. Compare two schemes.

*Pipeline parallelism*: device $A$ holds layers 1–36 and their weights; device $B$ holds layers 37–72 and theirs. Device $A$ computes its 36 layers in sequence, sends the resulting activation to device $B$ over the interconnect, and device $B$ computes its 36 layers.

*Tensor parallelism*: every layer's weight is split into 2 equal shards along its output axis — a type-U layer's $f$ columns, or a type-D layer's $d$ columns — one shard per device; both devices hold a full, identical copy of the network's input $X_1$. For each layer, a device computes its own output shard from the full input it already holds, with no communication needed for that; the two devices then exchange shards over the interconnect so that both again hold the layer's full output, before the next layer starts.

For each scheme, give the end-to-end latency for this one input, the peak memory used on either device — stating exactly which tensors are resident — and the total bytes either device sends over the interconnect during the whole pass. Say which scheme a single request should prefer, and why the other scheme is used in practice anyway.

### Part 4 — Sharding a feed-forward network

A feed-forward network computes $Y = \operatorname{act}(XW_1)W_2$ with $X \in \mathbb{R}^{T\times d}$, $W_1 \in \mathbb{R}^{d\times f}$, $\operatorname{act}(XW_1) \in \mathbb{R}^{T\times f}$, $W_2 \in \mathbb{R}^{f\times d}$, $Y \in \mathbb{R}^{T\times d}$, with $d = 8{,}192$ and $f = 32{,}768$ as in Part 2, and $\operatorname{act}$ a pointwise nonlinearity. Counting multiply-accumulates as two FLOPs and excluding $\operatorname{act}$ and any bias, this costs $4Tdf$ FLOPs, and the two weights together occupy $2dfs$ bytes ($s = 2$ for bf16). Consider two ways to split this block across $p$ devices; assume $p$ divides $f$ (tensor sharding) or $T$ (token sharding) exactly, as it does for every $p$ used below.

*Tensor sharding*: $W_1$'s $f$ columns and $W_2$'s $f$ rows are each split into $p$ equal shards, one per device; $X$ is replicated on every device. Device $i$ computes $H_i = \operatorname{act}(XW_{1,i}) \in \mathbb{R}^{T\times f/p}$ locally, then a partial output $Y_i = H_iW_{2,i} \in \mathbb{R}^{T\times d}$, also locally. The true $Y = \sum_i Y_i$ is formed by an all-reduce of the $p$ partial outputs, each of logical size $T \times d$ at $s_c$ bytes/element for the communicated values ($s_c = s = 2$ unless stated otherwise); an ideal ring all-reduce has every device send $2(p-1)/p$ times that logical size and receive the same amount, and this all-reduce cannot overlap with any of this block's own compute.

*Token sharding*: $W_1$ and $W_2$ are fully replicated on every device; $T$'s rows are split into $p$ equal groups, one per device, each computing the complete two-matmul block on its own $T/p$ rows, with no communication of its own inside this block.

Take $T = 2{,}048$ tokens (a batched decode step, smaller than Part 2's prefill). For $p = 1, 2, 4, 8, 16, 32, 64$, give each scheme's FLOPs, weight bytes, and predicted latency per device, and, for tensor sharding, the communication volume per device and its time on the $B_{\text{net}} = 400$ GB/s interconnect of Part 3. For each $p$, state whether tensor sharding is compute-, memory- or communication-bound, and do the same for token sharding. Repeat the bound classification for tensor sharding at $p = 2, 4, 16, 64$ on a slower, cross-node interconnect with $B_{\text{net}}' = 25$ GB/s in place of $B_{\text{net}}$. Finally, say what, if anything, would let an all-reduce's communication overlap with compute in a real deployment of this block, given that it cannot for the single block considered above.
