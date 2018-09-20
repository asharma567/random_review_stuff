



'''
Review of manojit's refactoring
for the most part he's done some pretty cool stuff e.g. the generate negative labels. I didn't realize 
the append method on the dataframe creates a new copy which makes it take much more time. He also brought 
down the complexity of it significantly. A clear lesson was to timeprofile every line when refactoring code 
to make it faster.
'''

def generate_negative_labels(df_merged,subject_fields=RELEVANT_BUDGET_FIELDS):
    df_empty = pd.DataFrame() 
 
    #iterate through the entire dataframe per user
    for user_id in df_merged.rocketrip_user_id_expense.unique():
        df_user_level = df_merged[df_merged.rocketrip_user_id_expense == user_id]


        #iterate through each budget ids
        for budget_id in df_user_level.budget_id.dropna().unique():
            df_subject_budget = df_user_level[df_user_level.budget_id == budget_id]
            
            #label field positive
            df_subject_budget['label'] = df_subject_budget.budget_type_budget
            
            #append positive
            df_empty = df_empty.append(df_subject_budget)

            #grab budget-side data
            df_budget_data = df_subject_budget[subject_fields]
            
            #grab every expense not matched to the subject budget ID i.e. negative labels
            df_all_unmatched_lineitems_to_be_filled = \
                df_user_level[df_user_level.budget_id != budget_id]

            #iterates through columns for budget data
            #filling in the budget data for the negative label line items
            for column_to_be_downfilled in df_budget_data.columns:

                #this unpacks a uniquefied value
                subject_budget_value = df_budget_data[column_to_be_downfilled].unique()[0]
                #passes the subject value to all the fields
                df_all_unmatched_lineitems_to_be_filled[column_to_be_downfilled] = subject_budget_value

            #negative label literal
            df_all_unmatched_lineitems_to_be_filled['label'] = 'not_a_match'

            #append negatives
            df_empty = df_empty.append(df_all_unmatched_lineitems_to_be_filled)
        
    return df_empty

Removed the double for loop and the .append method which makes another copy of a 
dataframe under the hood. This became a known thing after i made the function. 

def generate_negative_labels(df, subject_fields = RELEVANT_BUDGET_FIELDS):
	"""
	This method generates negative labels (mismatched expenses) on a user-account level.
	"""

	#Creates a copy of the DataFrame; MAY CHANGE TO LIST TO HOLD FALSE EXAMPLES
	df_overall = df.copy()
	df_overall.loc[:, "label"] = df_overall.loc[:, "budget_type_budget"]
	
	# Retrieve the unique budget ids.
	budgetids_unique = df.id_budget.dropna().unique()
	negative_labels_holder =[df_overall]

	for budget_id in budgetids_unique:
		df_holder = df.loc[df["id_budget"] == budget_id, :]
		user_id = df_holder.loc[:, "rocketrip_user_id_expense"].iloc[0]
		budget_data = df_holder.loc[:, subject_fields]
		#Identify all user expenses not pertaining to this particular budget
		unmatched_line_items = df.loc[(df["rocketrip_user_id_expense"] == user_id) & (df["id_budget"] != budget_id),:]

		num_rows_unmatched = len(budget_data.index)
		num_cols = len(sujbect_fields)

		random_elements = np.random.random
		for index, col in enumerate(subject_fields):
			fake_column_value = budget_data.iloc[random_elements[index]][col]
			unmatched_line_items.loc[:, col] = fake_column_value
		
		unmatched_line_items["label"] = "not_a_match"
		negative_labels_holder.append(unmatched_line_items)

	df_overall = pd.concat(negative_labels_holder)

	return df_overall

'''
This exhibits the new .pipe method for a dataframe which is a .apply equiv for inplace changes. It makes the 
calls in the sequence given.
'''
def preprocess_budget_and_lineitem_data_column_by_column(df):
	"""
	Encapsulation function to perform all feature engineering helper methods on an inputted dataframe.
	Parameters:
		- df_preprocessing: 
	"""
	df = (df.pipe(make_normalized_features)
			.pipe(convert_temporal_features_to_ts)
			.pipe(make_price_related_features)
			.pipe(make_date_related_features)
			.pipe(make_all_string_related_features)
			.pipe(make_stopword_related_features)
			.pipe(make_membership_features)
			.pipe(match_vendor_names)
		)

	return df

def check_word_in_string(normalized_string, target_string, anti_pattern = None):
	"""
	Checks if target_string is contained within the normalized_string.
	PARAMETERS:
		- normalized_string:
		- target_string: The substring the method is checking for.
		- anti_pattern: Any superstring we want to check against.
	Returns a boolean value
	"""
	if anti_pattern:
		return (anti_pattern not in normalized_string) and (target_string in normalized_string)
	else:
		return (target_string in normalized_string)


def make_membership_features(df_processing):
	"""
	Function to create new features relate to set memberships.
	Creates features based on whether a feature contains a particular token
	Parameter:
		- df_preprocessing: The DataFrame containing the line item expenses.
	"""
	expense_check_list = [
		"hotel",
		"flight",
		"car",
		"rail",
		"taxi",
		"fees_and_misc",
		"bus",
		"train",
		"other"
	]

	features_to_check = [
		"expense_type_name_expense",
		"expense_type_name_itemization"
	]

	for expense_feature in expense_check_list:
		for feature in features_to_check:
			df_preprocessing["{0}_{1}".format(feature, expense_feature)] = df_preprocessing[feature].apply(lambda x: float(x == expense_feature))


	return df_preprocessing

def make_stopword_related_features(df_preprocessing):
	"""
	Helper function to make all stopword-related features in-place in the DataFrame.
	"""
	vend_name_exp = "vendor_name_expense_normalized"
	exp_type_name_exp = "expense_type_name_expense"
	exp_type_name_item = "expense_type_name_itemization"


	features_stopwords_pairing = [
		(vend_name_exp, POST_NORMALIZATION_STOP_WORDS_FOR_VENDOR_NAME),
		(vend_name_exp, POST_NORMALIZATION_STOP_WORDS_FOR_CAR_VENDOR_NAME),
		(vend_name_exp, STOP_WORDS_FOR_FLIGHTS_VENDOR_NAME_EXPENSE),
		(exp_type_name_item, POST_NORMALIZATION_STOP_WORDS_FOR_CAR_EXPENSE_TYPE_NAME),
		(exp_type_name_exp, POST_NORMALIZATION_STOP_WORDS_FOR_CAR_EXPENSE_TYPE_NAME),
		(exp_type_name_item, STOP_WORDS_FOR_FLIGHTS_EXPENSE_TYPE_NAME ),
		(exp_type_name_exp, STOP_WORDS_FOR_FLIGHTS_EXPENSE_TYPE_NAME),
		(exp_type_name_item, STOP_WORDS_FOR_RAIL_EXPENSE_TYPE_NAME),
		(exp_type_name_exp, STOP_WORDS_FOR_RAIL_EXPENSE_TYPE_NAME)
		]

	for feature_name, stopword_list in features_stopwords_pairing:
		df_preprocessing["{0}_stopwords".format(feature_name)] = df_preprocessing[feature_name].apply(lambda x: check_membership_if_in_stopwords(x, stoword_list))


	return df_preprocessing
'''
This is a much better way to use .apply using multiple columns.
'''
def match_vendor_names(df):
	"""
	Helper function to create features based on fuzzy string matching.
	TODO: Experiment with fuzzy-wuzzy library.
	Set up experimentation platform to compare fuzzy-wuzzy features
		to stringmatching features.
	"""
	vend_name_exp = "vendor_name_expense_normalized"
	purch_vend_rec = "purchase_vendor_reciepts_normalized"
	travel_vend_rec = "travel_vendor_reciepts_normalized"

	me = sm.MongeElkan(sim_func=sm.JaroWinkler().get_raw_score)
	df.loc[:, "MongeElkanScore"] = df.apply(lambda row: 
		max(
			me.get_raw_score(row[vend_name_exp].split(), row[purch_vend_rec].split()),
			me.get_raw_score(row[travel_vend_rec].split(), row[vend_name_exp].split())
			)
		)
	return df

This is a new feature that's coming together using a new NLP package called spacey (spacy.io) which made to overtake NLTK. Where NLTK lacks is it's productionization usage, it's slow, and takes up a lot of space. However, spacey lacks in documentation.

def make_sentiment_related_features(df_preprocessing):
	"""
	"""
	pass

'''
What I like about the function below is the doc string. It notes the past function it's looking to 
replace and it's high level interpretation which makes it much more sensible.
'''
def make_all_string_related_features(df_processing):
	"""
	Function to create new features related to substring membership.
	For certain strings, check if anti-pattern exists.
	An anti-pattern is a sub-string we don't want to match on.
	For example, if looking for the substring 'tax',
		we don't want to match on 'taxi'.
	"""
	
	'''
	List of strings (and anti-patterns if any) to check for as features
	Functions to check:
		- check_if_tax_in_str: (["tax"], "taxi")
		- check_if_car_in_str: (["car"], None)
		- check_if_hotel_lodging_in_str: (["lodg", "hotel"], None)
		- check_if_airbnb_in_str: (["airbnb", "air bnb"], None)
		- check_if_omega_world_travel_in_str: (["omega world travel"], None)
	'''

	patterns_antipatterns = [
		(["tax"], "taxi", "tax_str"),
		(["car"], None, "car_str"),
		(["lodg", "hotel"], None, "hotel_str")
		(["airbnb", "air bnb"], None, " airbnb"),
		(["omega world travel"], None, "omega_world_travel")
	]

	expense_type_features_to_check = [
		"expense_type_name_expense",
		"expense_type_name_itemization"
	]

	expense_category_features_to_check = [
		"expense_category_itemization",
		"expense_category_expense"
	]

	for pattern, anti_pattern,features in patterns_antipatterns:
		for expense_type in expense_type_features_to_check:
			df_processing["{0}_{1}".format(feature, expense_type)] = df_processing[expense_type].apply(lambda x: check_multiple_substrings(x, pattern, anti_pattern))

		for expense_category in expense_category_features_to_check:
			df_processing["{0}_{1}".format(feature, expense_category)] = df_processing[expense_category].apply(lambda x: check_multiple_substrings(x, pattern, anti_pattern))


	return df_preprocessing

