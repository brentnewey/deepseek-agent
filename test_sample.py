"""Sample Python file for testing the DeepSeek Agent"""

def fibonacci(n):
    """Calculate the nth Fibonacci number recursively"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def factorial(n):
    """Calculate factorial of n"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

if __name__ == "__main__":
    print("Testing fibonacci(10):", fibonacci(10))
    print("Testing factorial(5):", factorial(5))