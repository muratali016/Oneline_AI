from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error,mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def process(model,X_train,X_test,y_train,y_test,):
    """
    This function gives you the best model and scores of the best model
    The given data must be separated to train and valid or tested and cleaned from non-numerical values

    """
    model.fit(X_train, y_train)
    print("first_model accuracy :{} ".format(model.score(X_test,y_test)))
    print("finding best params and applying to model ...")
    rf_grid={"n_estimators":np.arange(10,100,10),
    "max_depth":[None,3,5,10],
    "min_samples_split":np.arange(2,20,2),
    "min_samples_leaf":np.arange(1,20,2),
    "max_features":[0.5,1,"sqrt","auto"]}
    rs_model=RandomizedSearchCV(model,param_distributions=rf_grid,n_iter=50,
    cv=5,verbose=True)
    
    rs_model.fit(X_train, y_train)

    train_preds = rs_model.predict(X_train)
    val_preds=rs_model.predict(X_test)
    scores={"Training MAE ": mean_absolute_error(y_train,train_preds),
             "Valid MAE ": mean_absolute_error(y_test,val_preds),           
            "Training R^2":r2_score( y_train,train_preds),
            "Valid R^2 ":r2_score(y_test,val_preds) } 
    with open('best_model.pkl','wb') as f:
        pickle.dump(rs_model,f)
    print(scores)
    
    a=scores["Training MAE "]
    b=scores["Valid MAE "]
    dictt={"Training MAE":a,"Valid/Test MAE":b}
    comparison=pd.DataFrame(dictt,index=["accuracy"])
    comparison.T.plot(kind="bar",figsize=(10,6))
    return rs_model
