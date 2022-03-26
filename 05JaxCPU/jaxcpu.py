from jax import grad
import jax.numpy as jnp

class JaxTest():
    """
      test = JaxTest()
      test.run_first_grad()
    """
    @staticmethod
    def linear_regression(x: float):
        """
        create a linear regress of float with slop 2.0 and intercept 1.0
        """
        return jnp.dot(x, 2.0) + 1.0 

    def __init__(self):
        self.config = {
            "x" : float(8.5)
            }
        self.default_x = float(3.0)
        self.test_function = JaxTest.linear_regression 
        
    def assign_test_function(self, f):
        self.test_function = f     


    def run_first_grad(self):
        print(f"value of my linear regression on x={self.config.get('x', self.default_x)} is {self.test_function(self.config.get('x', self.default_x))}")
        linear_grad = grad(self.test_function)
        print(f"first grad derivate of my function on x={self.config.get('x', self.default_x)} is {linear_grad(self.config.get('x', self.default_x))}") # evaluate it at x=3.0
