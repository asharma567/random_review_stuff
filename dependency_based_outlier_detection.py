import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class dependency_variable_outlier_detector(object):
    def __init__(
            self, 
            training_data, 
            target_variable_name, 
            model=None
        ):
        
        self.X, self.y = self._get_XY(training_data, target_variable_name)
        
        if not model:
            if self._is_binary(target_variable_name):
                self.model = RandomForestClassifier(random_state=1)
            else:
                self.model = RandomForestRegressor(random_state=1)
        
        self.model.fit(self.X, self.y)

    def _is_binary(self, vector):
        return len(set(vector)) == 2

    def _get_XY(self, training_data, target_variable_name):
        preprocessed_data = training_data[:]
        
        y = preprocessed_data.pop(target_variable_name)
        X = preprocessed_data
        X_dummitized = dummy_it(X)

        return X_dummitized, y

    def compute_errors(self):
        self.df_errors = pd.DataFrame(_compute_errors(self.model, self.X, self.y), columns=['outlier_score'])
        return self.df_errors

    def get_largest_negative_residual(self, errors):
        return np.argmin(self.df_errors.outlier_score)

    def get_largest_positive_residual(self, errors):
        return np.argmax(self.df_errors.outlier_score)

    def get_example_w_largest_negative_residual(self, df_original):
        '''
        df_original this assumes the indices line up with post preprocessing (X)
        '''
        return df_original.ix[self.get_largest_negative_residual]

    def get_example_w_largest_positive_residual(self, errors, df_original):
        '''
        df_original this assumes the indices line up with post preprocessing (X)
        '''
        return df_original.ix[self.get_largest_positive_residual]

    def score(self):
        #check if targ var is binary then score accordingly
        return score_regression_model_r2(self.model, self.X, self.y)


    def sort_examples_by_residuals(self, errors, df_orig): 
        return df_orig.ix[np.argsort(errors)]

    def get_rsquared_linear_model(self, X, y):
        import statsmodels.api as sm
        '''
        Get RMSE given response variable (y) and predictor set (X)
        USE CASE: helper function for VIF
        '''
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
     
        return model.rsquared

def _compute_errors(model, X, y):
    predictions = np.array(model.predict(X))
    errors = [truth - predictions[idx] for idx, truth in enumerate(y)]
    return errors


def score_regression_model_r2(model, X,y):
    '''
    just explains fit of the model not whether the model generalizes well
    cv woudl be the best best for that
    '''
    from sklearn.metrics import r2_score
    
    return r2_score(y, model.predict(X))


def dummy_it(input_df, linear_model=False):
    '''
    I: Pandas DataFrame with categorical features
    O: Same df with binarized categorical features
    *check the dummy variable trap thing
    '''
    
    # base_case empty DF to append to
    base_case_df = pd.DataFrame()
    categorical_variables = []
    dropped_variables = []
    
    # every column that's not a categorical column, we dummytize
    for col in input_df.columns:
        if str(input_df[col].dtype) != 'object':
            base_case_df = pd.concat([base_case_df, input_df[col]], axis=1)
        else:
            if linear_model:
                dropped_variables.append(pd.get_dummies(input_df[col]).ix[:, -1].name)
                #leaves the last one out
                base_case_df = pd.concat([base_case_df, pd.get_dummies(input_df[col]).ix[:, :-1]], axis=1)
            else:
                base_case_df = pd.concat([base_case_df, pd.get_dummies(input_df[col])], axis=1)
            categorical_variables.append(col)
    
    # print out all the categoricals which are being dummytized
    if categorical_variables:
        print ('Variables Being Dummytized: ')
        print ('=' * 10)
        print ('\n'.join(categorical_variables))

    if dropped_variables:
        print ('Dropped to avoid dummy variable trap: ')
        print ('=' * 10)
        print ('\n'.join(dropped_variables))

    return base_case_df


if __name__ == "__main__":

    outlier_scorer = dependency_variable_outlier_detector()
    outlier_scorer.compute_errors()

