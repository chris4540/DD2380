# Question 1:
This problem can be formulated in matrix form.
Please specify the initial probability vector "pi",
the transition probability matrix "A" and the observation probability matrix B.

# Ans to Question 1:
The formula can be written as $\pi  A  B$

# Question 2:
> First, we need to multiply the transition matrix with our current
> estimate of states.

    What is the result of this operation?

# Ans to Question 2:
    pi * A is the probability distribution vector of each state at t = 1.

    In general we can write pi * A^n to tells the probability distribution
    of each stae at t = n for time-homogeneous transitional probability
    matrix "A"

# Question 3:
> In the following, the result must be multiplied with the observation matrix.

    What is the result of this operation?

# Ans to Question 3:
    pi * A * B is the probability distribution vector of each observation a t = 1.

# Question 4:
Why is it valid to substitute $O_{1:t} = o_{1:t}$ with $O_{t} = o_{t}$
when we condition on the state $X_t = x_i$ ?

# Ans to Question 4:
The question is why the following statement is true.

$$P(O_t = o_t| X_t = x_i, O_{1:t} = o_{1:t}) = P(O_t = o_t| X_t = x_i)$$

Given the condition that $X_t = x_i$, we can directly know the probability of
the observation at time $t$ by the observation probability matrix. This is one of
the HMM assumption that current observation is statistically independent of the 
previous observations given that $X_t = x_i$.

