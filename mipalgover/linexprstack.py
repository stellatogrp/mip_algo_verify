

class LinExprStack(object):

    def __init__(self,
                 linexprs):
        '''
            linexprs is a list of LinExpr objects for when we want to stack parameter sets
        '''
        self.linexprs = linexprs
