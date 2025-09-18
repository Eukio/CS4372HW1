import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import statsmodels.api as sm

original_df = pd.read_csv("https://raw.githubusercontent.com/Eukio/CS4372HW1/9714f27a8c780017e82912c3ea0f718ec8903886/student-mat.csv", sep=';')
x = original_df[['Medu','Fedu','famrel','failures','absences']].copy() #'Pstatus', 
y = original_df['G3']
# using SGDRegressor 
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled, columns=x.columns)
#print(x_scaled.describe())
features = ['Medu','Fedu','famrel']
# x = x_scaled[features]
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)
# X_train.shape, X_test.shape
# sgd = SGDRegressor(max_iter=1000, tol=1e-3)
# sgd.fit(X_train, y_train)
# sgd.score(X_test, y_test)
# y_pred = sgd.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# ev = explained_variance_score(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(mse, mae, ev, r2)

# OLS Using StatsModels Library
x = x[features]
x = sm.add_constant(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)
mod = sm.OLS(y_train, X_train)
res = mod.fit()
print(res.summary())
y_test_predict = res.predict(X_test)
sm.tools.eval_measures.rmse(y_test, y_test_predict)