import scalarflow as sf

def main():
    with sf.Graph() as g:
        x = sf.Variable(0, name="x")
        y = sf.Variable(2, name="y")

        L0 = sf.Exp(x, name="L0")           # L0 = e^x
        L1 = sf.Pow(x, 2, name="L1")        # L1 = x^2
        L2 = sf.Pow(y, 2, name="L2")        # L2 = y^2
        L3 = sf.Add(L1, L2, name="L3")      # L3 = L1 + L2
        L4 = sf.Pow(L3, 3, name="L4")       # L4 = L3^3
        L5 = sf.Multiply(L0, L4, name="L5") # L5 = L0 * L4

        result = g.run(L5, compute_derivatives=True)

        g.gen_dot("autodiff_exercise.dot")

if __name__ == "__main__":
    main()
