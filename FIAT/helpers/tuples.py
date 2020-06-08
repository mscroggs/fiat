def mis(m, n):
    """Returns all m-tuples of nonnegative integers that sum up to n."""
    if m == 1:
        return [(n,)]
    elif n == 0:
        return [tuple([0] * m)]
    else:
        return [tuple([n - i] + list(foo))
                for i in range(n + 1)
                for foo in mis(m - 1, i)]
