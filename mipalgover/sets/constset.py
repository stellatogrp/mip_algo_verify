from mipalgover.sets.boxset import BoxSet


class ConstSet(BoxSet):

    def __init__(self, linexprs, val):
        super().__init__(linexprs, val, val)
