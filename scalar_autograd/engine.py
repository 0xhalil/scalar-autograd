class Value:
    def __init__(self, data, _children=(), op=''):
        self.data = data
        self._op = op
        self.grad = 0
        self._children = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(data=(self.data+other.data), _children = (self, other),op = "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(data=(self.data*other.data), _children = (self, other), op = "*")

        def _backward():
            self.grad += out.grad*other.data
            other.grad += out.grad*self.data

        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(data=(pow(self.data, other)), _children = (self, ), op = "**")

        def _backward():
            self.grad += other*(pow(self.data, other-1)) * out.grad

        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(data= (0 if self.data <= 0 else self.data), _children = (self,), op = "ReLU" )

        def _backward():
            self.grad += (self.data > 0 ) * out.grad
        out._backward = _backward

        return out

    
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
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
