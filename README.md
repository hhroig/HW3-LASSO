# First Order Algorithms to solve the LASSO problem.
### (HW3-Optimization)

by Harold A. Hernández Roig
(October, 2018)

### USAGE:

- Run script "main.m". It already have some predifined values to show how well the algorithms work.
- The central core of this work are the functions:
   - CD_BW_lasso: (Block-Wise) Coordinate Descent Algoritm for Penalized (LASSO) 
                   Least Square Regression.
   - coor_desc_lasso: Cyclical (Block-Wise) Coordinate Descent Algoritm for 
                      Penalized (LASSO) Least Square Regression. Usage of
                      soft-trhesholding included.		
   - SGD_lasso: Stochastic Gradient Descent for Penalized (LASSO)  Least Square
                Regression.	
   - SGD_Momentum_lasso: Stochastic Gradient Descent with Momentum for Penalized (LASSO) 
                         Least Square Regression.				
- There are also 2 secondary functions:
   - sample_data: generates a simulated (sparse) dataset
   - S_lambda: soft-trhesholding function for "coor_desc_lasso"   
