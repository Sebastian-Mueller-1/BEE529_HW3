import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import math
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize, least_squares

df = pd.read_csv("BEE529 Dataset BonnevilleChinook_2010to2015.csv")

#scrub
print("Do we have any NaN Values:", df.isnull().values.any()) #do we have any nan values -- yes
print("How many NaN values do we Have:", df.isnull().sum().sum()) #how many -- 147

# we have 147 NaN values which means at most we could have 147 rows with at least 1 NaN value. Since we have 2197 rows of data
# I feel comfortable with removing all rows with NaN and retaining the integrity of the data. 

df.dropna(inplace=True) # remove rows with at least 1 NaN

#assign columns as variables
count = df["Unnamed: 2"].values
outflow = df["Unnamed: 3"].values
temp = df["Unnamed: 4"].values
turbidity = df["Unnamed: 5"].values

# remove index 0 which is a column string
count = np.delete(count,0)
outflow = np.delete(outflow, 0)
temp = np.delete(temp,0)
turbidity = np.delete(turbidity,0)

# need to convert entries in numpy array to float.  Pandas defaults to importing all values as strings when hitting a NaN. This is an computationally inefficient way to 
# convert the arrays especially if we were using a larger data sets. If they were larger I may important to pandas, understand the data and then re-read the data 
# without NaN so pandas imports values as int and I wouldn't need to iterate across array again do this second conversion.

count = count.astype(np.float64)
outflow = outflow.astype(np.float64)
temp = temp.astype(np.float64)
turbidity = turbidity.astype(np.float64)

#model class
class SalmonDamModel():
    
    def __init__(self):
        """for simplicity will init all the variables from our data set"""
        self.count = count
        self.outflow =  outflow
        self.temp = temp
        self.turbidity = turbidity

        self.independent_var1 = None # should be init with independent variable for simple regressions
        self.independent_var2 = None
        self.independent_var3 = None

        self.dependent_var = count

        self.model_results_full = None
        self.model_results_parameters = None # holds coefficient values as array
        self.model_r2 = None
        self.model_yPredicted = None


    def simple_exponential_model(self, coefficient_params):
        """takes list for coefficients parameters and returns predicted output array"""
       
        x = self.independent_var1
        a = coefficient_params[0]
        b = coefficient_params[1]
        c = coefficient_params[2]
        
        yPredicted =  a*x**b+c
        self.model_yPredicted = yPredicted
        return yPredicted

    def multivariate_non_linear_model(self,coefficient_params):
        """takes list for coefficients parameters and returns predicted output array"""
        
        x_1 = self.independent_var1
        x_2 = self.independent_var2
        x_3 = self.independent_var3
        a = coefficient_params[0]
        b = coefficient_params[1]
        c = coefficient_params[2]
        d = coefficient_params[3]
        e = coefficient_params[4]
        f = coefficient_params[5]
        g = coefficient_params[6]

        yPredicted = a*x_1**b + c*x_2**d + e*x_3**f + g
        self.model_yPredicted = yPredicted
        return yPredicted

    def residuals_simple(self, coefficient_params):
        yPredicted = self.simple_exponential_model(coefficient_params)
        return self.dependent_var - yPredicted
    
    def residuals_multi(self, coefficient_params):
        yPredicted = self.multivariate_non_linear_model(coefficient_params)
        return self.dependent_var - yPredicted
   
    def calculate_r2(self, coefficient_params):
        """takes coefficient_parameters as list and find predicted output and calculates r-squared"""
        
        r_value = (1 - np.sum((self.dependent_var - self.model_yPredicted)**2)/np.sum((self.dependent_var - np.mean(self.dependent_var))**2))**0.5
        self.model_r2= r_value**2
        # print("this is r2 of", r_value**2)

    def calculate_simple_RMSE(self, coefficient_params):
        """takes list for coefficients parameters and returns RMSE of simple non linear model"""
        
        yPredicted = self.simple_exponential_model(coefficient_params)
        MSE = mean_squared_error(self.dependent_var, yPredicted)
        RMSE = math.sqrt(MSE)
        return RMSE

    def calculate_multi_RMSE(self, coefficient_params):
        """takes list for coefficients parameters and returns RMSE of multivariate non linear model"""
        
        yPredicted = self.multivariate_non_linear_model(coefficient_params)
        MSE = mean_squared_error(self.dependent_var, yPredicted)
        RMSE = math.sqrt(MSE)
        return RMSE

    def run_minimize_model_fit(self, initial_guess, simple, RMSE):
        """Optimize objective function, RMSE, takes initial parameter guess input as list and simple agreement as boolean T or F for simple or multi model """
        
        if simple == True:
            
            if RMSE == True:
                results = minimize(self.calculate_simple_RMSE, initial_guess)
                self.model_results_full = results
                self.model_results_parameters = results.x

            else:
                results = least_squares(self.residuals_simple, initial_guess)
                self.model_results_full = results
                self.model_results_parameters = results.x
        
        
        else:
            if RMSE == True:
                results = minimize(self.calculate_multi_RMSE, initial_guess)
                self.model_results_full = results
                self.model_results_parameters = results.x

            else:
                results = least_squares(self.residuals_multi, initial_guess)
                self.model_results_full = results
                self.model_results_parameters = results.x

    def visualize_cross_plot(self, *args):
        """input takes any amount of modeled yPredictions"""

        self.calculate_r2(self.model_results_parameters)
        counter = 1
        colors = ["b.","r.","g.","y."]
        for model in args:
            plt.plot(self.dependent_var, model, colors[counter], label='Model4' + '($r^2$=%.4f)'%self.model_r2)
            counter += 1
            
        plt.legend()
        plt.xlabel('Observations ($y_i$)'); plt.ylabel('Model Estimates ($\hat{y_i}$)') 
        plt.plot(count,count, 'k-')
        plt.show()

    def visualize_results(self, modeled_results, label_array):
        
        self.calculate_r2(self.model_results_parameters)
        plt.plot(self.independent_var1, self.dependent_var,'.', color='blue',alpha=0.1)   # add observation series
        plt.plot(self.independent_var1, modeled_results, 'o', color='red' )    ## add modeled series
        plt.legend( ['Observed', 'Predicted'],title ="r2=%.4f"%self.model_r2)
        plt.xlabel(label_array[0])
        plt.ylabel(label_array[1])
        plt.show()

hypothesis1 = SalmonDamModel()
hypothesis2 = SalmonDamModel()
hypothesis3 = SalmonDamModel()
hypothesis4 = SalmonDamModel()

# set parameters 
hypothesis1.independent_var1 = outflow
hypothesis2.independent_var1 = temp
hypothesis3.independent_var1 = turbidity
hypothesis4.independent_var1 = outflow
hypothesis4.independent_var2 = temp
hypothesis4.independent_var3 = turbidity

# run simple exponential model parameter optimization using LSE for hypo 1-3
hypothesis1.run_minimize_model_fit([0,0,0], True, False) 
hypothesis2.run_minimize_model_fit([0,0,0], True, False)
hypothesis3.run_minimize_model_fit([0,0,0], True, False)

# run multi exponential model parameter using LSE for hypo 4
hypothesis4.run_minimize_model_fit([1,1,1,1,1,1,1],False, False)


model1 = hypothesis1.simple_exponential_model(hypothesis1.model_results_parameters)
model2 =  hypothesis2.simple_exponential_model(hypothesis2.model_results_parameters)
model3 =  hypothesis3.simple_exponential_model(hypothesis3.model_results_parameters)
model4 = hypothesis4.multivariate_non_linear_model(hypothesis4.model_results_parameters)

hypothesis1.visualize_results(model1, ["outflow (kcfs)","count (fish/day)"])
hypothesis2.visualize_results(model2, ["temperature (celcius)","count (fish/day)"])
hypothesis3.visualize_results(model3, ["turbidity (secchi ft)","count (fish/day)"])

hypothesis4.visualize_cross_plot(model4)


print("hypo4:",hypothesis4.model_results_parameters)
print("hypo3:",hypothesis3.model_results_parameters)
print("hypo2:",hypothesis2.model_results_parameters)
print("hypo1:",hypothesis1.model_results_parameters)

#seaborn boxplot outlier exploration
import seaborn as sns
sns.boxplot(data=outflow)
plt.show()