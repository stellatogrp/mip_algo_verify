

class BoxSet(object):

    def __init__(self, linexprs, l, u):
        self.linexprs = linexprs  # this can be either LinExpr or LinExprStack
        self.l = l
        self.u = u
