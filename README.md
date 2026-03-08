# scalar-autograd

![](assets/image.png)

A scalar-valued autograd engine built from scratch — no PyTorch, no NumPy, no magic.

Inspired by [Karpathy's micrograd](https://github.com/karpathy/micrograd). Written to understand backpropagation at the deepest level, not just use it.

---

## What this is

Two files. That's it.

- `engine.py` — the math engine. A single `Value` class that tracks operations and computes gradients automatically.
- `my_nn.py` — a neural network library built on top of the engine. `Neuron`, `Layer`, `MLP`.

Together they can train a multi-layer perceptron. The total code is under 100 lines.

---

## How it works

### The core idea: every operation remembers how to undo itself

When you write `a * b`, Python calls `Value.__mul__`. This creates a new `Value` for the result — but also stores a `_backward` closure that knows how to propagate gradients back to `a` and `b` using the chain rule:

```python
def _backward():
    self.grad += other.data * out.grad   # d(out)/d(self) = other
    other.grad += self.data * out.grad   # d(out)/d(other) = self
```

No gradient is computed yet. The closure just sits there, waiting.

### The computation graph

Every `Value` tracks its `_children` — the nodes that produced it. This builds a directed acyclic graph (DAG) as you do forward computations:

```
a = Value(2.0)
b = Value(3.0)
c = a * b       # c._children = {a, b}
d = c + Value(1.0)  # d._children = {c, Value(1.0)}
```

### Backward pass: topological sort + chain rule

`loss.backward()` does two things:

1. **Topological sort** — visits the graph depth-first, so every node is processed after all nodes that depend on it.
2. **Reverse traversal** — calls `_backward()` on each node from output to inputs, flowing gradients backward through the chain rule.

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    self.grad = 1  # dL/dL = 1
    for v in reversed(topo):
        v._backward()
```

Why `+=` instead of `=` when accumulating gradients? Because the same node can appear in multiple places in the graph (e.g. `a + a`). Each path contributes its own gradient — they must be summed.

### Supported operations

`+`, `*`, `**`, `relu`, `-`, `/`, `+=` (reflected ops for Python interop with plain floats)

---

## Neural network layer

`my_nn.py` builds on the engine:

- **`Neuron(nin)`** — `nin` weights initialized randomly in `[-1, 1]`, bias at 0. Computes `relu(w·x + b)` (or just `w·x + b` for the output layer).
- **`Layer(nin, nout)`** — `nout` neurons in parallel, each taking the same input.
- **`MLP(nin, nouts)`** — stacks layers. Output of each layer becomes input to the next.

```python
model = MLP(2, [4, 4, 1])
# 2 inputs → hidden layer of 4 → hidden layer of 4 → 1 output
```

The output layer has no ReLU — you want raw values for loss computation.

---

## Usage

```python
from scalar_autograd.engine import Value
from scalar_autograd.my_nn import MLP

# build model
model = MLP(2, [4, 4, 1])

# forward pass
x = [Value(1.0), Value(2.0)]
out = model(x)

# backward pass
out.backward()

# inspect gradients
for p in model.parameters():
    print(p.data, p.grad)

# gradient descent step
lr = 0.01
for p in model.parameters():
    p.data -= lr * p.grad
```

---

## Why gradients accumulate with `+=`

Consider `z = a + a`. Both uses of `a` contribute to the gradient of `z` with respect to `a`:

```
dz/da = dz/d(first a) + dz/d(second a) = 1 + 1 = 2
```

If `_backward` used `=` instead of `+=`, the second call would overwrite the first and the gradient would be wrong.

---

## Why the output layer has no ReLU

Stacking linear layers without a non-linearity collapses into a single linear transformation — no matter how many layers you add. ReLU breaks this by introducing a "kink" at zero, enabling the network to represent non-linear functions.

But the output layer produces a raw score used in loss computation. Clipping it with ReLU would destroy negative predictions and distort the loss signal.
