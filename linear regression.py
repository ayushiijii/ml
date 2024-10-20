from math import sqrt 
import seaborn as sns 
import matplotlib.pyplot as plt 

# Input the X and Y data
x = list(map(int, input("Enter X Data: ").split())) 
y = list(map(int, input("Enter Y Data: ").split())) 

n = len(x) 

# Calculate the means of x and y
xmean = sum(x) / n 
ymean = sum(y) / n 

# Calculate a and b arrays
a = [x[i] - xmean for i in range(n)] 
b = [y[i] - ymean for i in range(n)] 

# Calculate the required sums
ab = sum([a[i] * b[i] for i in range(n)]) 
asquare = sum([a[i]**2 for i in range(n)]) 
bsquare = sum([b[i]**2 for i in range(n)]) 

# Calculate the correlation coefficient (r)
r = ab / sqrt(asquare * bsquare) 

# Calculate standard deviations
dely = sqrt(bsquare) / sqrt(n - 1)
delx = sqrt(asquare) / sqrt(n - 1) 

# Calculate coefficients for the linear regression line
b1 = r * dely / delx 
b0 = ymean - b1 * xmean 

# Print the coefficients and the equation
print("B0 :", b0, "B1 :", b1) 
print("Equation : y = ", b0, " + ", b1, " * x") 

# Plot the data points and the regression line
sns.scatterplot(x=x, y=y, label='Data points') 

# Predict y values based on the regression line
xpred = [i for i in range(min(x), max(x) + 1)] 
ypred = [b0 + b1 * i for i in xpred] 

# Plot the regression line
sns.lineplot(x=xpred, y=ypred, color='red', label='Regression line') 

# Show the plot
plt.xlabel('X Data')
plt.ylabel('Y Data')
plt.title('Linear Regression Plot')
plt.legend()
plt.show()
