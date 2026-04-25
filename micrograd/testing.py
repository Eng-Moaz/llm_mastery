from engine import Value
from visualizer import draw_dot

def testing() -> None:
    a = Value(4.0, label="a")
    b = Value(3.0, label="b")
    c = a / b
    c.label = "c"
    c.backward()
    print(c)
    print(c._prev)
    print(c.label)
    draw_dot(c).render('micrograd/graphs/micrograd_graph', format='svg', view=False)

if __name__ == "__main__":
    testing()