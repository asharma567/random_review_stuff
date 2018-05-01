'''
shiny can be hosted on aws and at that point anyone can hook in to that dashboard. 
dash by comparison of shiny, bokeh, is a python library which enables one to make dashboards. 
This is a great way to automate a non-technical business group with multiple business requests.
'''


import pandas as pd
import numpy as np
try:
    import MySQLdb
except:
    import pymysql
    pymysql.install_as_MySQLdb()
    import MySQLdb
from datetime import datetime, timedelta
from simple_salesforce import Salesforce

import json
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table_experiments as dt

from plotly.offline import download_plotlyjs, plot, iplot
import plotly.graph_objs as go
### requires pip3 install dash-core-components==0.21.0rc1 for Tabs

import urllib.parse

from TS_toolkit.ts_tools import *
from TS_toolkit.db_utils import *

# DBs
prodConn = get_prod_conn()
histConn = get_hist_conn()
sales_force_Conn = get_sf_conn(
	user="mannis@rocketrip.com",
	pwd="Quanson82!",
	token="X8IOyvlmICGpLdgu589Wx21Ar"
)


# get go live date and customer list from Salesforce

soql = '''
SELECT 
	Name,
	Initial_Invites_Sent_date__c 
FROM 
	Account 
WHERE 
	Type = 'Customer'
'''

df_goLiveDate = pd.DataFrame(sales_force_Conn.bulk.Account.query(soql))

# Get list of customers for drop down
custDropDownVals = []
for i in range(0,len(df_goLiveDate)): custDropDownVals.append(dict(label=df_goLiveDate['Name'][i], value=df_goLiveDate['Name'][i]))
customers = list(df_goLiveDate['Name'])

app = dash.Dash()

# configure to run plotly locally
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

# app.config['suppress_callback_exceptions'] = True

# set plotly graph config options
plotConfig = {'showLink':False, 'modeBarButtonsToRemove':['sendDataToCloud']}

# app html layout set up
# can REPL from here, there's no GUI for editing
app.layout = html.Div([

    html.H1(children="Customer Success Metrics Dashboard"),
    html.P(id='header'),

    html.Label('Select Customer'),
    dcc.Dropdown(id='custDropDown', options=custDropDownVals),
    html.Button(id='dl-button', n_clicks=0, children='Get all the data'),
    html.Div(''),

    dcc.Tabs(id='tabs', value='tab-1',
            tabs=[{'label': 'Engagement', 'value': 'tab-1'},
                  {'label': 'Program Review', 'value': 'tab-2'},
                  {'label': 'Historical Data', 'value': 'tab-3'}]),

    # engagement tab
        html.Div([
            dcc.RadioItems(id='travSinceGoLive',
                        options=[{'label': 'Traveled Since Go-live', 'value': 'sinceGoLive'},
                                {'label': 'All Users', 'value': 'allTravs'}],
                        value='sinceGoLive',labelStyle={'display': 'inline-block'}),
            dcc.RadioItems(id='useRTtripCheck', options=[{'label': 'Bucket by RT Trips', 'value': 'rtTrip'}, {'label': 'Bucket by Historical Trips', 'value': 'histTrip'}], value='rtTrip', style={'display': 'none'}),
            dcc.Graph(id='activations-by-bucket', config=plotConfig),
            html.A('Download Graph Data', id='dl-acts', download="rawdata.csv", href="", target="_blank"),
            dcc.RadioItems(id='ctCost-check',
                           options=[{'label': 'Show as count', 'value': 'count'},
                                    {'label': 'Show as $$', 'value': 'cost'}], value='count'),
            dcc.Graph(id='budVbook-graph', config=plotConfig),
            dt.DataTable(id='topTravelers-dt', rows=[{}], filterable=True, sortable=True, selected_row_indices=[])
        ],
        id='tab-1',
        style={'display': 'none'}),

    # program review tab
        html.Div([html.P('program review tab')],
            id='tab-2',
            style={'display': 'none'}
        ),

    # historical data tab
        html.Div([
            dcc.RadioItems(id='histCtCost-check',
                           options=[
                                {'label': 'Show as count', 'value': 'count'},
                                {'label': 'Show as $$', 'value': 'cost'}], 
                                value='count'),
            dcc.Graph(id='histPareto-graph', config=plotConfig)
            ],
            id='tab-3',
            style={'display': 'none'}
        ),

    # hidden dataframes for sharing in callbacks
    html.Div(id='hist-data', style={'display': 'none'}),
    html.Div(id='emp-data', style={'display': 'none'}),
    html.Div(id='budget-data', style={'display': 'none'}),
    html.Div(id='booked-data', style={'display': 'none'}),
    html.Div(id='engagement-data', style={'display': 'none'}),
    html.Div(id='budVbook-data', style={'display': 'none'})

], style={'fontFamily':'Arial'})

# this function returns the tab display when selected
def generate_display_tab(tab_id):
    def display_tab(value):
        if value == tab_id:
            return {'display': 'block'}
        else:
            return {'display': 'none'}
    return display_tab

# callbacks to display each tab as selected
for tab_id in ['tab-1', 'tab-2', 'tab-3']:
    app.callback(
        Output(tab_id, 'style'),
        [Input('tabs', 'value')]
    )(generate_display_tab(tab_id))

# update header
@app.callback(Output('header', 'children'),[Input('custDropDown', 'value')])
def update_header(custName):
    if custName is None:
        return ""
    else:
        goLive = df_goLiveDate.loc[df_goLiveDate['Name'] == custName].Initial_Invites_Sent_date__c.to_string(index=False)
        return(custName + "<br>Go Live Date: " + goLive)

# get historical data and store as hidden json dataframe
@app.callback(Output('hist-data', 'children'),[Input('dl-button', 'n_clicks')],[State('custDropDown', 'value')])
def dl_hist(n_clicks, custName):
    if n_clicks == 0: return

    # query historical data - merge both flight and hotel into one DF
    temp = {}
    for tbl in ["flight", "hotel", "car", "rail"]: 
    	temp[tbl] = get_historical_data(orgName=custName, travelMode=tbl, histConn=histConn)

    flight = pd.DataFrame({'employee': temp['flight'].employee_name, 'employee_id': temp['flight'].employee_id,
                           'class': temp['flight'].fare_class, 'vendor': temp['flight'].vendor, 'hotel': "",
                           'city': "", 'state': "", 'country': "",
                           'route': temp['flight'].route_destinations,
                           'nonstop_or_connecting': temp['flight'].nonstop_or_connecting,
                           'dom_or_int': temp['flight'].dom_or_int,
                           'price': temp['flight'].total_price_usd, 'ap_days': temp['flight'].ap_days,
                           'start': temp['flight'].departure, 'end': temp['flight']['return'],
                           'room_rate': np.nan, 'nights': np.nan, 'ticket_count': temp['flight'].ticket_count,
                           'mode': "flight"})
    hotel = pd.DataFrame(
        {'employee': temp['hotel'].employee_name, 'employee_id': temp['hotel'].employee_id, 'class': "",
         'vendor': temp['hotel'].hotel_brand, 'hotel': temp['hotel'].hotel_property,
         'city': temp['hotel'].hotel_city, 'state': temp['hotel'].hotel_state,
         'country': temp['hotel'].hotel_country, 'route': "", 'nonstop_or_connecting': "",
         'dom_or_int': np.where(temp['hotel'].hotel_country == 'United States', 'Domestic', 'International'),
         'price': temp['hotel'].total_price_usd,
         'ap_days': temp['hotel'].ap_days, 'start': temp['hotel'].checkin, 'end': temp['hotel'].checkout,
         'room_rate': temp['hotel'].room_rate_usd,
         'nights': temp['hotel'].hotel_nights, 'ticket_count': np.nan, 'mode': "hotel"})
    histData = flight.append(hotel)
    histData.start = pd.to_datetime(histData.start)
    histData.end = pd.to_datetime(histData.end)

    return histData.to_json(date_format='iso', orient='split')

# get employee data from prod and store as hidden json dataframe
@app.callback(
    Output('emp-data', 'children'),
    [Input('dl-button', 'n_clicks')],
    [State('custDropDown', 'value')])
def dl_emp(n_clicks, custName):
    if n_clicks == 0: return

    orgId = get_org_id(orgName=custName, prodConn=prodConn)

    # get employees with activated date
    employeeData = pd.read_sql(con=prodConn,
                               sql="""SELECT au.id AS employee_id,au.first_name,au.last_name,au.email,emp.title,dep.department_name,emp.location,emp.status,emp.is_admin,emp.is_copilot,ee.status_date,emp.internal_id
                                                  FROM rocketripapp_employee AS emp
                                                  JOIN rocketripapp_organization AS org ON emp.organization_id = org.id
                                                  JOIN auth_user AS au ON emp.user_id = au.id
                                                  JOIN rocketripapp_department AS dep ON emp.department_id = dep.id
                                                  JOIN (SELECT employee_id,max(created_at) AS status_date
                                                        FROM rocketripapp_employeeevent
                                                        WHERE EVENT = 'Activated' OR EVENT = 'Invited'
                                                        GROUP BY employee_id) AS ee ON ee.employee_id = emp.user_id
                                                  WHERE emp.organization_id = """ + str(orgId))
    return employeeData.to_json(date_format='iso', orient='split')

# get booked item data and store as hidden json dataframe
@app.callback(
    Output('booked-data', 'children'),
    [Input('dl-button', 'n_clicks')],
    [State('custDropDown', 'value')])
def dl_bookedItems(n_clicks, custName):
    if n_clicks == 0: return

    orgId = get_org_id(orgName=custName, prodConn=prodConn)

    bookedItems = pd.read_sql(con=prodConn,
                              sql="""SELECT bi.id,bi.traveler_employee_id as employee_id,bi.cost,bi.created_at,bi.budget_id,bi.travel_mode_type_aggregate as mode
                                                 FROM rocketrip_bookings_bookeditem AS bi
                                                 WHERE bi.deleted_at IS NULL
                                                 AND bi.is_in_use = 1
                                                 AND bi.organization_id = """ + str(orgId))
    return bookedItems.to_json(date_format='iso', orient='split')

# get budget data and store as hidden json dataframe
@app.callback(
    Output('budget-data', 'children'),
    [Input('dl-button', 'n_clicks')],
    [State('custDropDown', 'value')])
def dl_budgets(n_clicks, custName):
    if n_clicks == 0: return

    orgId = get_org_id(orgName=custName, prodConn=prodConn)
    budgets = get_budgets(prodConn=prodConn, orgID=orgId)

    return budgets.to_json(date_format='iso', orient='split')

# format data for engagement graphs
@app.callback(
    Output('engagement-data', 'children'),
    [Input('hist-data', 'children'),
     Input('emp-data','children'),
     Input('budget-data','children'),
     Input('booked-data','children'),
     Input('dl-button', 'n_clicks')])
def agg_engagement_data(d1, d2, d3, d4, n_clicks):
    """
    aggregate data by employee for engagement reporting
    :param histData: historical data from dl_hist (df)
    :param empData: employee data from dl_emp (df)
    :param budgetData: budget data from dl_budgets (df)
    :param bookedData: booked items from dl_bookedItems (df)
    :param sinceGoLive:
    :return: dataframe with prod and historical data by employee
    """
    if n_clicks == 0: return
    histData = pd.read_json(d1, orient='split')
    empData = pd.read_json(d2, orient='split')
    budgetData = pd.read_json(d3, orient='split')
    bookedData = pd.read_json(d4, orient='split')

    empKey = "employee_id" if (all(histData.employee == "") & all(histData.employee_id != "")) else "employee"
    histData.rename(columns={empKey: 'emp'}, inplace=True)
    histData['counter'] = 1

    topFlights = histData[histData['mode'] == 'flight'].groupby('emp', as_index=False)['ticket_count', 'price'].agg('sum')
    topFlights.rename(columns={'ticket_count': 'flightCount', 'price': 'flightSpend'}, inplace=True)

    topHotels = histData[histData['mode'] == 'hotel'].groupby('emp', as_index=False).agg({'counter': 'count', 'price': 'sum'})
    topHotels.rename(columns={'counter': 'hotelCount', 'price': 'hotelSpend'}, inplace=True)

    topTravelers = pd.merge(topFlights, topHotels, 'outer')
    topTravelers['totalSpend'] = topTravelers[['flightSpend', 'hotelSpend']].sum(axis=1)
    topTravelers['histTripCount'] = topTravelers[['flightCount', 'hotelCount']].sum(axis=1)

    # add in prod data
    if empKey == "employee":
        empData['employee'] = empData.first_name.str.lower().replace("[^A-Za-z0-9]",
                                                                               "") + " " + empData.last_name.str.lower().replace(
            "[^A-Za-z0-9]", "")
        if len(topTravelers) > 1:
            if all(topTravelers.emp.str.contains("/")):
                first = topTravelers.emp.str.split("/", expand=True)[1].str.lower()
                last = topTravelers.emp.str.split("/", expand=True)[0].str.lower()
            else:
                first = topTravelers.emp.str.split(" ", expand=True)[0].str.lower()
                last = topTravelers.emp.str.split(" ", expand=True)[
                    round(topTravelers.emp.str.count(" ").mean())].str.lower()
            topTravelers.emp = first.str.lower().replace("[^A-Za-z0-9]", "") + " " + last.str.lower().replace(
                "[^A-Za-z0-9]", "")

            # merge on first/last name if we have historica ldata by employee
            topTravelers = pd.merge(topTravelers, empData, how="left", left_on="emp", right_on="employee")
        else:
            topTravelers = empData
    else:
        topTravelers.emp = topTravelers.emp.astype(int).astype(str)
        topTravelers = pd.merge(topTravelers, empData, how="left", left_on="emp", right_on="internal_id")

    # group booked items by employee, and w/ vs w/o budget
    bookedItemsNoBudByEmp = bookedData[bookedData.budget_id.isnull()].groupby('employee_id').size().reset_index(name='RT_booked_item_count_no_bud')
    bookedItemsWBudByEmp = bookedData[bookedData.budget_id.notnull()].groupby('employee_id').size().reset_index(name='RT_booked_item_count_w_bud')

    # group budgets by employee
    budgetsByEmp = budgetData.groupby('employee_id').size().reset_index(name="RT_budget_count")

    # using employee_id merge in budgets and bookedItems
    topTravelers = pd.merge(topTravelers, budgetsByEmp, on="employee_id", how="left")
    topTravelers = pd.merge(topTravelers, bookedItemsWBudByEmp, on="employee_id", how="left")
    topTravelers = pd.merge(topTravelers, bookedItemsNoBudByEmp, on="employee_id", how="left")

    topTravelers.RT_booked_item_count_w_bud.fillna(0, inplace=True)
    topTravelers.RT_booked_item_count_no_bud.fillna(0, inplace=True)

    # trip counts and check if user has traveled since go live
    topTravelers['RTtripCount'] = topTravelers.RT_booked_item_count_no_bud + topTravelers.RT_booked_item_count_w_bud
    topTravelers['RTtripCount'].loc[np.isnan(topTravelers.RTtripCount)] = 0
    topTravelers['histTripCount'].loc[topTravelers.histTripCount < 0] = 0
    topTravelers['histTripCount'].loc[np.isnan(topTravelers.histTripCount)] = 0
    topTravelers['sinceGoLive'] = topTravelers[['RT_booked_item_count_w_bud', 'RT_booked_item_count_no_bud']].sum(axis=1) > 0

    topTravelers['histTripBucket'] = get_trip_bins(topTravelers.histTripCount)
    topTravelers['RTtripBucket'] = get_trip_bins(topTravelers.RTtripCount)

    return topTravelers.to_json(date_format='iso', orient='split')

# toggle the use Rt tripbucket checkbox
@app.callback(
    Output('useRTtripCheck','style'),
    [Input('travSinceGoLive', 'value')])
def show_rtTrip_checkBox(toggle):
    if toggle == 'sinceGoLive':
        return {'display': 'block'}
    else:
        return {'display': 'none'}

# activations by trip bucket component
@app.callback(
    Output('activations-by-bucket', 'figure'),
    [Input('travSinceGoLive', 'value'),
     Input('useRTtripCheck', 'value'),
     Input('engagement-data', 'children')])
def make_activations_graph(goLiveToggle, tripToggle, data):

    topTravelers = pd.read_json(data, orient='split')

    if tripToggle == 'rtTrip':
        tripVar = 'RTtripBucket' if goLiveToggle == 'sinceGoLive' else 'histTripBucket'
        xTitle = 'Trip Bucket - trips since launch'
    else:
        tripVar = 'histTripBucket'
        xTitle = 'Trip Bucket - trips in historical'

    if goLiveToggle == 'sinceGoLive':
        acts = topTravelers[topTravelers.email.notnull() & topTravelers.status.isin(["Active", "Invited"])][topTravelers.sinceGoLive][['status', tripVar]]
        gTitle = 'Traveler Activations'
    else:
        acts = topTravelers[topTravelers.email.notnull() & topTravelers.status.isin(["Active", "Invited"])][['status', 'histTripBucket']]
        gTitle = 'All Activations'

    graphActs = pd.merge(acts.groupby([tripVar, 'status']).size().reset_index(name='count'),acts.groupby(tripVar).size().reset_index(name='total'), how='left', on=tripVar)
    graphActs['per'] = graphActs['count'] / graphActs['total']
    graphActs[tripVar] = pd.Categorical(graphActs[tripVar], categories=['0','1','2:5','6:10','2:10','11:20','21:40','41:75','76:100','101+'])
    graphActs = graphActs.sort_values(by=tripVar)

    return {'data': [
        go.Bar(x=graphActs[graphActs.status == 'Active'][tripVar], y=graphActs[graphActs.status == 'Active']['per'],
               text=graphActs[graphActs.status == 'Active']['count'], textposition='auto', name='Active',
               marker=dict(color=RT_colors()['orange'][0]), textfont=dict(family='arial', size=14, color='#FFFFFD')),
        go.Bar(x=graphActs[graphActs.status == 'Invited'][tripVar], y=graphActs[graphActs.status == 'Invited']['per'],
               text=graphActs[graphActs.status == 'Invited']['count'], textposition='auto', name='Invited',
               marker=dict(color=RT_colors()['grey'][0]), textfont=dict(family='arial', size=14, color='#FFFFFD'))
    ],
        'layout': go.Layout(
            title=gTitle,
            xaxis={'title': xTitle},
            yaxis={'tickformat': ',.0%'},
            barmode='stack'
        )}

# update download csv for activations by tripbucket
@app.callback(
    Output('dl-acts', 'href'),
    [Input('travSinceGoLive', 'value'),
     Input('engagement-data', 'children')])
def update_act_download_link(toggle, data):

    topTravelers = pd.read_json(data, orient='split')

    if toggle == 'sinceGoLive':
        acts = topTravelers[topTravelers.email.isnull() == False & topTravelers.status.isin(["Active", "Invited"])][
            topTravelers.sinceGoLive][['status', 'tripBucket']]
    else:
        acts = topTravelers[topTravelers.email.isnull() == False & topTravelers.status.isin(["Active", "Invited"])][
            ['status', 'tripBucket']]

    graphActs = pd.merge(acts.groupby(['tripBucket', 'status']).size().reset_index(name='count'),
                         acts.groupby('tripBucket').size().reset_index(name='total'), how='left', on='tripBucket')
    graphActs['per'] = graphActs['count'] / graphActs['total']

    csv_string = graphActs.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
    return csv_string

@app.callback(
    Output('budVbook-data', 'children'),
    [Input('booked-data', 'children'),
     Input('custDropDown', 'value'),
     Input('dl-button', 'n_clicks')])
def agg_budVbook_data(data, custName, n_clicks):

    bookedData = pd.read_json(data, orient='split')

    goLive = df_goLiveDate.loc[df_goLiveDate['Name'] == custName].Initial_Invites_Sent_date__c.to_string(index=False)
    bookedData = bookedData[bookedData.created_at > datetime.strptime(goLive, '%Y-%m-%d')]

    bookedData['date'] = pd.to_datetime(bookedData.created_at, "%Y-%m-%d").dt.date
    bookCt = bookedData[bookedData.budget_id.isnull()].groupby('date', as_index=False).size().reset_index(name='bookNoBudCt')
    bookCost = bookedData[bookedData.budget_id.isnull()].groupby('date', as_index=False).agg({'cost': 'sum'}).rename(columns={'cost': 'bookNoBudCost'})
    books = pd.merge(bookCt, bookCost, how='outer')

    buds = bookedData[bookedData.budget_id.notnull()].groupby('date', as_index=False).size().reset_index(name='bookBud')
    budsCt = bookedData[bookedData.budget_id.notnull()].groupby('date', as_index=False).size().reset_index(name='bookBudCt')
    budsCost = bookedData[bookedData.budget_id.notnull()].groupby('date', as_index=False).agg({'cost': 'sum'}).rename(columns={'cost': 'bookBudCost'})
    buds = pd.merge(budsCt, budsCost, how='outer')

    bookVbud = pd.merge(books, buds, how='outer').fillna(0)

    for col in bookVbud.columns.values[1:5]: bookVbud['cumul_' + col] = bookVbud[col].cumsum()

    return bookVbud.to_json(date_format='iso', orient='split')

@app.callback(
    Output('budVbook-graph', 'figure'),
    [Input('budVbook-data','children'),
     Input('ctCost-check','value')])
def make_budVbook_graph(data, toggle):

    bookVbud = pd.read_json(data, orient='split')

    var = 'Ct' if toggle == 'count' else 'Cost'
    tickForm = ',' if toggle == 'count' else '$,'

    bookTrace = go.Scatter(x=bookVbud.date, y=bookVbud['cumul_bookNoBud' + var], name='Bookings w/o budget', line=dict(color=RT_colors()['grey'][0]))
    budTrace = go.Scatter(x=bookVbud.date, y=bookVbud['cumul_bookBud' + var], name='Bookings w/o budget', line=dict(color=RT_colors()['orange'][0]))

    layout = dict(title='Bookings w/ and w/o budgets overtime',
                  xaxis=dict(rangeselector=dict(buttons=list([
                              dict(count=1, label='1m', step='month', stepmode='backward'),
                              dict(count=6, label='6m', step='month', stepmode='backward'),
                              dict(step='all')])
                            ),rangeslider=dict(thickness=.05),
                      type='date'),
                  yaxis=dict(tickformat=tickForm))

    return dict(data=[bookTrace, budTrace], layout=layout)

# populate datatable with top traveler data
@app.callback(
    Output('topTravelers-dt','rows'),
    [Input('engagement-data', 'children')])
def make_topTraveler_dt(data):
    topTravelers = pd.read_json(data, orient='split')
    return topTravelers.to_dict('records')

@app.callback(
    Output('histPareto-graph','figure'),
    [Input('engagement-data','children'),
     Input('histCtCost-check', 'value')])
def agg_histPareto_data(data, toggle):
    topTravelers = pd.read_json(data, orient='split')

    var = 'cumul_histTripPer' if toggle == 'count' else 'cumul_histSpendPer'
    yTitle = '% of trips' if toggle == 'count' else '% of spend'

    histTravSumm = topTravelers[topTravelers.histTripCount.notnull()][['emp','histTripCount','totalSpend']]

    travSummSpend = histTravSumm.sort_values('totalSpend', ascending=False).reset_index(drop=True)
    travSummSpend['cumul_histSpend'] = travSummSpend.totalSpend.cumsum()
    travSummSpend['cumul_histSpendPer'] = travSummSpend.cumul_histSpend / max(travSummSpend.cumul_histSpend)
    travSummSpend['numEmployees'] = pd.Series(range(1,len(travSummSpend) + 1))

    travSummTrip = histTravSumm.sort_values('histTripCount', ascending=False).reset_index(drop=True)
    travSummTrip['cumul_histTripCount'] = travSummTrip.histTripCount.cumsum()
    travSummTrip['cumul_histTripPer'] = travSummTrip.cumul_histTripCount / max(travSummTrip.cumul_histTripCount)
    travSummTrip['numEmployees'] = pd.Series(range(1, len(travSummTrip) + 1))

    trace = go.Scatter(x=travSummTrip['numEmployees'], y=travSummTrip['cumul_histTripPer']) if toggle == 'count' else go.Scatter(x=travSummSpend['numEmployees'], y=travSummSpend['cumul_histSpendPer'])

    layout = dict(title='Historical Traveler Pareto', yaxis=dict(title=yTitle, tickformat='%'))

    return dict(data=[trace], layout=layout)

def get_trip_bins(counts):

    n = max(counts)

    bins = dict.fromkeys([0], '0')
    bins.update(dict.fromkeys([1], '1'))

    if n > 100:
        bins.update(dict.fromkeys(range(2, 11), '2:10'))
        bins.update(dict.fromkeys(range(11, 21), '11:20'))
        bins.update(dict.fromkeys(range(21, 41), '21:40'))
        bins.update(dict.fromkeys(range(41, 76), '41:75'))
        bins.update(dict.fromkeys(range(76, 101), '76:100'))
        bins.update(dict.fromkeys(range(101, int(n + 1)), '101+'))

    elif n > 50 and n <= 100:
        bins.update(dict.fromkeys(range(2, 6), '2:5'))
        bins.update(dict.fromkeys(range(6, 11), '6:10'))
        bins.update(dict.fromkeys(range(11, 21), '11:20'))
        bins.update(dict.fromkeys(range(21, 41), '21:40'))
        bins.update(dict.fromkeys(range(41, int(n + 1)), '41+'))

    else:
        bins.update(dict.fromkeys(range(2, 6), '2:5'))
        bins.update(dict.fromkeys(range(6, 11), '6:10'))
        bins.update(dict.fromkeys(range(11, int(n + 1)), '11+'))

    x = []
    for i in counts: x.append(bins[int(i)])

    return x

if __name__ == '__main__':
    app.run_server()


# download the data for testing:
# cust = 'Kelloggs'
# histData = pd.read_json(dl_hist(n_clicks=1, custName=cust), orient='split')
# empData = pd.read_json(dl_emp(n_clicks=1, custName=cust), orient='split')
# budgetData = pd.read_json(dl_budgets(n_clicks=1, custName=cust), orient='split')
# bookedData = pd.read_json(dl_bookedItems(n_clicks=1, custName=cust), orient='split')