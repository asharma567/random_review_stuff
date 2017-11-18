from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#get the cv thing to work
#refactor for readability
#find the intersection between the basic dependency outlier model
#send to charu
#ask him about the roc_auc

class attribute_wise_learning_for_outlier_detector(object):
    
    def __init__(
            self, 
            training_data, 
            model=None
        ):
        
        if not model:

            #to keep results consistent for comparison we've fixed the stochastic implementation
            self.model = RandomForestRegressor(random_state=1)

        self.training_set_continuous_only = training_data[training_data.describe().columns]
        
        #this is a hack so that we only iterate on continuous features 
        #but retain the information on the dummy cols
        #this should be removed if the code is fixed
        categorical_cols= get_names_of_categorical_variables(training_data)
        dummy_cols = list(set(training_data[categorical_cols].values[0]))
        self.training_set_dummies_only = dummy_it(training_data)[dummy_cols]

    def outlier_score_all_datapoints(self, transformation_to_errors=None):

        self.feature_weights, self.df_errors = self.compute_weights_and_errors_for_features(self.training_set_continuous_only)

        #question when do you apply this?
        #to all the outliers 
        if transformation_to_errors:
            df_transformed_errors = self.df_errors.applymap(transformation_to_errors)
            df_transformed_errors.columns = [col + '_transformed' for col in df_transformed_errors.columns]
            self.df_errors = pd.concat([self.df_errors, df_transformed_errors], axis=1)
        
        # scaling for comparison, adding them up
        df_errors_scaled = _standardize(self.df_errors)
        
        #apply_feature_relevance weights
        df_errors_scaled_wtd = df_errors_scaled[:]
        for col in df_errors_scaled.columns:
            df_errors_scaled_wtd[col] = df_errors_scaled[col] * self.feature_weights[col]
        
        self.df_errors['also_error'] = df_errors_scaled_wtd.sum(axis=1)
        
        self.df_errors['also_abs_error'] = self.df_errors['also_error'].apply(np.abs)
        self.df_errors['also_outlier_score'] = self.df_errors['also_abs_error']/self.df_errors['also_abs_error'].max()

        
        return self.df_errors['also_error']

    def compute_weights_and_errors_for_features(self, training_data):
        feature_relevence_weights_dict = {}
        errors_by_feature = {}

        for target_variable_name in training_data.columns:
            
            X, y = self._get_XY_and_add_dummies(training_data, target_variable_name)        
            self.model.fit(X, y)

            if not _vector_varies_in_values(y): 
                weight = 0            
                feature_relevence_weights_dict[target_variable_name] = weight
                continue

            #* the scoring functions here needs to be crossvalidated
            #the major differenc I could see here rmse is much more aggressive than
            #r2 that is it'll more likely call something a 0 weight because of the
            #threshold of 1
            
            if _is_binary(y): 
                from sklearn.metrics import roc_auc_score
                weight = _transform_to_scale_0_to_1(roc_auc_score(y, self.model.predict(X)))
                feature_relevence_weights_dict.append(weight)
            else:
                weight = _score_regression_model_r2(y, self.model.predict(X))
                
                #I've tried to add this however it only produces a negative score for total_price
                #perhaps try looking for another implementation, i've also tried adding makescorer 
                #function
                
                # weight = model_cv_score(
                #     self.model, 
                #     X, 
                #     y, 
                #     folds=10,
                #     scoring='r2'
                # )
                
                feature_relevence_weights_dict[target_variable_name] = weight
            
            # also compute the squared error
            errors_by_feature[target_variable_name] = _compute_errors(self.model, X, y)
    
        return feature_relevence_weights_dict, pd.DataFrame(errors_by_feature)

    def _transform_to_scale_0_to_1(self, roc_auc_score): 
        # [2*(item % 0.5) for item in roc_auc_score]
        return 2 * (roc_auc_score % 0.5)

    def _get_XY_and_add_dummies(self, training_data, target_variable_name):
        preprocessed_data = training_data[:]
        
        y = preprocessed_data.pop(target_variable_name)
        X = preprocessed_data
        
        #HACK until the classification scoring is fixed
        X = pd.concat([X, self.training_set_dummies_only],axis=1)
        return X, y





def model_cv_score(model, feat_matrix, labels, folds=10, scoring='roc_auc'):
    from sklearn import model_selection
    '''
    I: persisted model object, feature matrix (numpy or pandas datafram), labels, k-folds, scoring metric
    O: mean of scores over each k-fold (float)
    '''
    return np.median(model_selection.cross_val_score(model, feat_matrix, labels, cv=folds, scoring=scoring))



def calc_weight_with_regression_rmse(truths, preds):
    #UNUSED, but it was mentioned in outlier analysis
    #the min caps the feature at 1 and when it's subtracted 
    #the weight is 0 meaning High error features will have a 0 weight
    #low error will have a high weight

    return 1 - min(1, _score_regression_model_rmse(truths, preds))

def _score_regression_model_rmse(truths, preds):
    return ((preds - truths) ** 2).mean() ** .5

def _score_regression_model_r2(truths, preds):
    '''
    just explains fit of the model not whether the model generalizes well
    cv woudl be the best best for that
    '''
    from sklearn.metrics import r2_score

    return r2_score(truths, preds)


def _is_binary(vector):
    return len(set(vector)) == 2

def _standardize(df_data):        
    scaler = StandardScaler()
    df_errors_scaled = pd.DataFrame(scaler.fit_transform(df_data), columns=df_data.columns)
    return df_errors_scaled

def _compute_errors(model, X, y):
    predictions = np.array(model.predict(X))
    errors = [truth - predictions[idx] for idx, truth in enumerate(y)]
    return errors

def _vector_varies_in_values(vector): 
    return len(set(vector)) != 1


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

def get_names_of_categorical_variables(df):
    return [feature_name for feature_name in df.columns if str(df[feature_name].dtype) == 'object']
            


    
    

    


