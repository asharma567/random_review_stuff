'''
Below is my current method for pulling flight data. 

The goal is to model the flight query a user can select from when 
they're creating their budget.

The bottleneck with this query is I have to add a limit of 10000 dpr_ids, 
is there a better way to query perhapa from budget_id instead of dpr_ids.

I'd like to get as much data as possible but flight segment has a 
total of 506082518 rows as such any non-partitioned query I run times out.

'''

import pandas as pd
from sqlalchemy import create_engine
import joblib


def connect_to_market():
    engine_prod = create_engine('mysql+pymysql://production:Kazg8dAmC7dafMd@marketdata-vpc-enc.cvhe9o57xgm1.us-east-1.rds.amazonaws.com:3306/rocketrip_market', echo=False)
    return engine_prod.connect()

def connect_to_prod():
    engine_prod = create_engine('mysql+pymysql://analytician:crouchingtigerhiddenmouse@production-vpc-enc-readonly.cvhe9o57xgm1.us-east-1.rds.amazonaws.com/rocketrip_production', echo=False)
    return engine_prod.connect()

if __name__ == '__main__':
    
    qry = '''
    SELECT 
        DISTINCT (dpr_id)
    FROM 
        rocketrip_market.rocketripmarket_flight
    LIMIT 10000
    '''
    connect_market = connect_to_market()
    df_rocketripmarket_flight_dpr_id = pd.read_sql(qry, connect_market)
    
    #this could be specified to one column to pull faster
    qry = '''
    SELECT 
        * 
    FROM 
        rocketrip_production.rocketripapp_dataproviderresponse_budgets;
    '''

    connect_prod = connect_to_prod()
    rocketripapp_dataproviderresponse_budgets = pd.read_sql(qry, connect_prod)

    some_subset = rocketripapp_dataproviderresponse_budgets[rocketripapp_dataproviderresponse_budgets.dataproviderresponse_id.isin(df_rocketripmarket_flight_dpr_id.dpr_id)]

    data_pull_of_all_flight_options_by_budget_id = {}

    for budget_id in some_subset.budget_id.unique():
        dpr_ids = rocketripapp_dataproviderresponse_budgets[rocketripapp_dataproviderresponse_budgets.budget_id == budget_id].dataproviderresponse_id.tolist()
        #this should be some list of flights
        #this could be specified to one column to pull faster
        qry = '''
        SELECT 
            * 
        FROM 
            rocketrip_market.rocketripmarket_flight as flight
            LEFT JOIN 
                rocketrip_market.rocketripmarket_flightleg as flightleg ON flight.id = flightleg.flight_id
            LEFT JOIN 
                rocketrip_market.rocketripmarket_flightsegment as segment ON segment.leg_id = flightleg.id
        WHERE flight.dpr_id IN ''' + str(dpr_ids).replace(']', ')').replace('[', '(')
        df_rocketripmarket_flight_target_dpr_id = pd.read_sql(qry, connect_market)
        data_pull_of_all_flight_options_by_budget_id[budget_id] = df_rocketripmarket_flight_target_dpr_id

    print ('done!')
    joblib.dump(data_pull_of_all_flight_options_by_budget_id, 'data_pull_of_all_flight_options_by_budget_id.pkl')


