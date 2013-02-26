

class DirichletBoundaryCondition(object):

    def __init__(self, var, x, value):

        self.var = var
        self.x = x
        self.value = value

        #
