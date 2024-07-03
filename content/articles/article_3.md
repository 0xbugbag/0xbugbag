Title: Learn Python #2
Date: 2024-07-03 11:20
Modified: 2024-07-03 11:20
Category: blog, python
Slug: learn-python-2
Summary: In this post, I have written code to find fibonacci-prime number.
Tags: python
Authors: 0xbugbag
Status: published



## **Code** ![Python Image]({static}/images/python-logo.png){: width="100" height="100" }

Syntax highlighting

```python

def get_fibo(fibo_nth):
    '''
    This function is used to get Fibonacci numbers.
    
    Parameters:
    - fibo_nth (int): Fibonacci numbers n-th F(n-th) as arguments
    
    Returns:
    - lst_fibo (int): the list consists of Fibonacci numbers
    '''
    fnum = 1 # n=1, F1
    snum = 1 # n=2, F2
    lst_fibo = [] # an empty list
    for i in range(0,fibo_nth):
        # The Fibonacci sequence
        sum = fnum + snum # According to the recurrence: sum, fnum and snum
        fnum = snum # Then, shift fnum to snum
        snum = sum # Shift snum to sum, the operation will loop until fn
        lst_fibo.append(sum) # Append the sum to lst_fibo
    return lst_fibo

def get_prime(num):
    '''
    This function is used to get prime numbers.
    
    Parameters:
    - num (int): fibonacci numbers as arguments
    
    Returns:
    - prime numbers (int): consist of prime numbers
    '''
    if num <= 1: # To eliminate 0 and 1, not prime
        return False
    for i in range(2,num):
        if num % i == 0: # To eliminate even numbers, not prime
            return False
    return True # Return prime numbers

def fibonacci_prime(head):
    '''
    This function is used to get the first `n` Fibonacci primes.
    If `n` is less than 1 the return is invalid, since Fibonacci and primes numbers cannot be negative integers
    
    Parameters:
    - head (int): first numbers n-th as arguments
    
    Returns:
    - lst (int): the first `n` of th Fibonacci prime numbers
    '''
    if head > 0:
        lst_first_n = [] # Empty list to store Fibonacci primes
        for i in range(len(fibo_prime)):
            if i == head:
                break
            lst_first_n.append(fibo_prime[i])        
    else:
        # Since Fibonacci and primes numbers cannot be negative integers
        return f'Invalid number. Should be greater than 0'
    
    return lst_first_n

# Define a varibale of Fibonacci numbers n-th F(n-th)
fibo_nth = 43

# Call get_fibo function
fibo_num = get_fibo(fibo_nth)

# Call get_prime function within list comprehension to get list of Fibonacci prime numbers
fibo_prime = [num for num in fibo_num if get_prime(num)]

# Input number to print result of n-th numbers
head = int(input('Input n-numbers: '))

# Call fibonacci_prime function to get the first `n` of th Fibonacci prime numbers
res = fibonacci_prime(head)

# Print result
print(f'fibonacci_prime(n = {head})\n')
print(res)
```