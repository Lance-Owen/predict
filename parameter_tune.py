#
# import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
#
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor as XGBR
# # rfr_best = RandomForestRegressor()
# rfr_best = XGBR()
# params ={'n_estimators':range(10,100,1)}
# gs = GridSearchCV(rfr_best, params, cv=4)
#
# df = pd.read_csv('淮南预测值对比.csv',encoding='utf8')
# X_train,Y_train = df[['zbkzj','project_type']],df['下浮率']
# gs.fit(X_train,Y_train)
#
# #查验优化后的超参数配置
# print(gs.best_score_)
# print(gs.best_params_)



# width = 10 均值上下浮动0.2


from tools import *
# df = pd.read_csv('lishui.csv',encoding='gbk')
df = huainan_data()
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(df[['zbkzj','project_type']],df['下浮率'], test_size=0.2, random_state=42)
rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': range(10, 100),
    'max_depth': [None, 5, 10],
    'min_samples_split': range(2, 20),
    'min_samples_leaf': [1, 2, 4],
    # 'max_features': ['auto', 'sqrt', 'log2']
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(grid_search.best_score_)
print(best_params)
