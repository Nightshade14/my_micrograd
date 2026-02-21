import math


class Value:
    def __init__(self, data, _prev=(), _op="", label=""):
        self.data = data
        self._prev = set(_prev)
        self.grad = 0.0
        self._backward = lambda: None
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), (
            "only supporting int/float powers for now"
        )
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += out.grad * other * (self.data ** (other - 1))

        out._backward = _backward
        return out

    def tanh(self):
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += out.grad * (1 - t**2)

        out._backward = _backward
        return out

    def relu(self):
        # out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")
        out = Value(max(0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += out.grad * (out.data > 0)

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def _build_topo(v):
            if v not in visited:
                visited.add(v)
                for node in v._prev:
                    _build_topo(node)
                topo.append(v)

        _build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Value(data = {self.data})"
