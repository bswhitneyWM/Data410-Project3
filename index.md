# Project 3

This page investigates the theoretical advantages and disadvantages of multivariate regression analysis and boosting algorithims, and describes the analysis completed on the cars and Boston housing datasets using each of the methods. 

# Theroetical Discussion

## Multivariate Regression Analysis

## Gradient Boosting

## Extreme Gradient Boosting (xgboost)

# Analysis

## Cars Dataset

For the cars dataset I used used the `ENG, CYL, WGT` variables as the independent variables and `MPG` as the dependent variable. 
```python
cars = pd.read_csv('Data/cars.csv')
Xcars = cars[['ENG', 'CYL', 'WGT']].values
ycars = cars['MPG'].values
``` 

### Functions Defined
I decided to use the three kernels that we have been exploring in class (Tricubic, Epanechnikov, and Quartic) for the Locally Weighted Regression. I didn't make any changes to these kernels and were defined as follows: 

```python
def Tricubic(x):
    if len(x.shape) == 1:
        x = x.reshape(-1,1)
    d = np.sqrt(np.sum(x**2,axis=1))
    return np.where(d>1,0,70/81*(1-d**3)**3)

def Quartic(x):
    if len(x.shape) == 1:
        x = x.reshape(-1,1)
    d = np.sqrt(np.sum(x**2,axis=1))
    return np.where(d>1,0,15/16*(1-d**2)**2)

def Epanechnikov(x):
    if len(x.shape) == 1:
        x = x.reshape(-1,1)
    d = np.sqrt(np.sum(x**2,axis=1))
    return np.where(d>1,0,3/4*(1-d**2)) 
```

In terms of the Locally Wieghted Regresison and Boosted Regression, I also used the functions defined in class and past lectures. This way I knew the functions worked properly and could focus on other aspects such as paramter scanning, feature selection, and error analysis throughout the project. 

```python
def lw_reg(X, y, xnew, kern, tau, intercept): 
    n = len(X) # the number of observations
    yest = np.zeros(n)
    if len(y.shape)==1: # here we make column vectors
        y = y.reshape(-1,1)
    if len(X.shape)==1:
        X = X.reshape(-1,1)
    if intercept:
        X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
        X1 = X
    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
        f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
        f = LinearNDInterpolator(X, yest)
    output = f(xnew) # the output may have NaN's where the data points from xnew are outside the convex hull of X
    if sum(np.isnan(output))>0:
        g = NearestNDInterpolator(X,y.ravel()) 
        output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output

def boosted_lwr(X, y, xnew, kern, tau, intercept):
    Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
    new_y = y - Fx
    tree_model = DecisionTreeRegressor(max_depth=2, random_state=123)
    tree_model.fit(X,new_y)
    output = tree_model.predict(xnew) + lw_reg(X,y,xnew,kern,tau,intercept)
    return output  
```

Data was split into training and testing splits to be used in determining the best hyperparamters for the Locally Weighted Regression. I used a `random_state = 13` to ensure my results were reproducible. The `test_size` was set to 0.25 so 75% of the observations would be used to train the model and 25% would be used to validate the results. The code for this is shown below: 

```python
# Split the data for parameter selection
Xtrain_cars, Xtest_cars, ytrain_cars, ytest_cars = tts(Xcars, ycars, test_size=0.25, random_state=13)
Xtrain_boston, Xtest_boston, ytrain_boston, ytest_boston = tts(Xboston, yboston, test_size=0.25, random_state=13)
```

### Regression Analysis

- What parameters to scan
- Why we used sum of MSE and MAE
- HOW I did both methods at once


```python
# Parameters to scan
kernels = [Tricubic, Epanechnikov, Quartic]
taus = np.arange(0.1, 2.1, 0.1)

# Save MSE and MAE
best_params_lwr = tuple()
best_params_boost = tuple()
best_sum_lwr = 10**5
best_sum_boost = 10**5

#Scale the data
scale = StandardScaler()
Xtrain_cars_ss = scale.fit_transform(Xtrain_cars)
Xtest_cars_ss = scale.transform(Xtest_cars)

# Perform analysis
for kern in kernels:
    for tau in taus:
        # Locally Weighted Regression
        ypred_cars_lwr = lw_reg(Xtrain_cars_ss, ytrain_cars, Xtest_cars_ss, kern=kern, tau=tau, intercept=True)
        mse_lwr = mean_squared_error(ytest_cars, ypred_cars_lwr)
        mae_lwr = mean_absolute_error(ytest_cars, ypred_cars_lwr)
        
        # Boosted Locally Weighted Regression
        ypred_cars_boost = boosted_lwr(Xtrain_cars_ss, ytrain_cars, Xtest_cars_ss, kern=kern, tau=tau, intercept=True)
        mse_boost = mean_squared_error(ytest_cars, ypred_cars_boost)
        mae_boost = mean_absolute_error(ytest_cars, ypred_cars_boost)
        
        if(mse_lwr + mae_lwr < best_sum_lwr):
            best_sum_lwr = mse_lwr + mae_lwr
            best_params_lwr = (kern, tau)
            
        if(mse_boost + mae_boost < best_sum_boost):
            best_sum_boost = mse_boost + mae_boost
            best_params_boost = (kern, tau)

print(f'Best parameters for LWR are kern = {best_params_lwr[0]} and tau = {best_params_lwr[1]} for the Cars Dataset') 
print(f'Best parameters for Boosted LWR are kern = {best_params_boost[0]} and tau = {best_params_boost[1]} for the Cars Dataset') 
```
- Results (best params for each)
- Crossvalidation (Why I chose 5 splits, randmo seed and shuffle)

```python
# Perform Crossvalidation
kf = KFold(n_splits=5, shuffle=True, random_state=410)

# Save mse and mae
mse_lwr_cars = []
mae_lwr_cars = []
mse_boost_cars = []
mae_boost_cars = []
for idxTrain, idxTest in kf.split(Xcars):
    Xtrain, Xtest = Xcars[idxTrain], Xcars[idxTest]
    ytrain, ytest = ycars[idxTrain], ycars[idxTest]
    
    # Scale the data
    Xtrain_ss = scale.fit_transform(Xtrain)
    Xtest_ss = scale.transform(Xtest)
    
    # Locally Weighted Regression
    ypred_lwr = lw_reg(Xtrain_ss, ytrain, Xtest_ss, kern=best_params_lwr[0], tau=best_params_lwr[1], intercept=True)
    mse_lwr_cars.append(mean_squared_error(ytest, ypred_lwr))
    mae_lwr_cars.append(mean_absolute_error(ytest, ypred_lwr))
    
    # Boosted Locally Weighted Regression
    ypred_boost = boosted_lwr(Xtrain_ss, ytrain, Xtest_ss, kern=best_params_boost[0], tau=best_params_boost[1], intercept=True)
    mse_boost_cars.append(mean_squared_error(ytest, ypred_boost))
    mae_boost_cars.append(mean_absolute_error(ytest, ypred_boost))

# Print results
print("Crossvalidated LWR MSE for cars dataset", np.mean(mse_lwr_cars))
print("Crossvalidated LWR MAE for cars dataset", np.mean(mae_lwr_cars))
print("Crossvalidated Boosted LWR MSE for cars dataset", np.mean(mse_boost_cars))
print("Crossvalidated Boosted LWR MAE for cars dataset", np.mean(mae_boost_cars))
```

- Results of Crossvalidation 

### Cars Results

## Boston Housing Dataset

For the Boston Housing dataset, the same functions and kernels that were defined for the Cars dataset were used for analysis of the Boston Housin dataset. Again, the data was split into a training sample with 75% of the bservations and a testing set with 25% of the observations to be used for determining the ideal hyperparameters for the data. If you have any questions about these, please see the `Functions Defined` section above. 

### Feature Selection

### Regression Analysis

### Boston Results

# Final Results
- Compare against each other

# Conclusion
I was surprised through theoretical and class expectations
 - Highlihts that there is no universal laws, always good to try a couple methods
 - Perhaps randomness had to do with it as well, maybe random seeds would result in different results
 - why did it perform better in both cases in cars dataset
