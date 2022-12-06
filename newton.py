import argparse
import sympy as sym
import numpy as np
import time


def derivative(func):
    my_func = sym.parse_expr(func)
    return sym.diff(my_func)


def newton_method(x0, iteration_number, f, df, accuracy, step):
    # print('Function: ')
    # print(f)
    # print('Starting point: ')
    # print(x0)
    # print('Number of steps: ')
    # print(iteration_number)
    # print('Accuracy: ')
    # print(accuracy)
    x = x0 # initial guess

    start = time.time()
    for i in range(int(iteration_number)):
        y = f(x)
        y_prime = df(x)

        if np.abs(y_prime) < 0.000001:  # Stop if the denominator is too small
            break

        x1 = x - y / (y_prime * step)  # do newton's computation

        if np.abs(x1-x) <= accuracy:    # Stop when the result is within the desired tolerance
            print('Close enough')
            end = time.time()
            elapsed_time = end - start
            return x, elapsed_time

        x = x1

    end = time.time()
    elapsed_time = end - start
    return x, elapsed_time


parser = argparse.ArgumentParser(description='finding the zero by Newtons method.')

parser.add_argument('func', action='store', help='A function whose zero will be calculated.')
parser.add_argument('-starting_point', type=int, action='store', help='Starting point.')
parser.add_argument('-number_of_steps', type=int, action='store', help='Number of steps.')
parser.add_argument('-accuracy', type=float, action='store', help='Accuracy.')
parser.add_argument('-step_size', type=float, action='store', help='Step should be between 0 and 1.')


args = parser.parse_args()
x = sym.Symbol('x')
f = sym.parse_expr(args.func)
print('Starting point: ' + str(args.starting_point))
print('Number of steps: ' + str(args.number_of_steps))
print('Accuracy: ' + str(args.accuracy))
print('Step size: ' + str(args.step_size))
print('Function: ' + args.func)

df = derivative(args.func)
print('Derivative: ' + str(df))
f = sym.lambdify(x, f, 'numpy')
df = sym.lambdify(x, df, 'numpy')


solution, elapsed_time = newton_method(args.starting_point, args.number_of_steps, f, df, args.accuracy, args.step_size)
print(solution)
print('Execution time:', elapsed_time, 'seconds')

