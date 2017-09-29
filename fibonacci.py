from . import ChillstepCalculation
from aiida.orm.data.base import Int


class FibonacciCalculation(ChillstepCalculation):
    def start(self):
        self.ctx.n1 = 1
        self.ctx.n2 = 1
        self.ctx.iterations = self.inputs.parameters.value
        self.goto(self.iterate)


    def iterate(self):
        print "ITERATING", self.ctx.n1, self.ctx.n2
        self.ctx.n1, self.ctx.n2 = self.ctx.n2, self.ctx.n1+self.ctx.n2
        self.ctx.iterations -= 1
        if self.ctx.iterations < 1:
            self.goto(self.finalize)
    
    def finalize(self):
        
        self.goto(self.exit)
        return dict(output=Int(self.ctx.n2))

class FibonacciRecCalculation(ChillstepCalculation):
    def start(self):
        n = self.inputs.parameters.value
        print "n", n
        if n < 2:
            self.goto(self.exit)
            return dict(result=Int(1))
        else:
            self.goto(self.sumresults)
            return dict(
                    calc_1=FibonacciRecCalculation(parameters=Int(n-1)),
                    calc_2=FibonacciRecCalculation(parameters=Int(n-2)),
                )

    def sumresults(self):
        self.goto(self.exit)
        s = self.out.calc_1.out.result.value + self.out.calc_2.out.result.value
        print "s", s
        self.goto(self.exit)
        return dict(result=Int(s))
        




