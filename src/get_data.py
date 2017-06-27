import pandas as pd
import numpy as np
import zipfile

def unzip_file(): # unzip data.zip
    zip = zipfile.ZipFile('data/data.zip')
    zip.extractall('data')

def data_to_df(): # read json data into dataframe
    df = pd.read_json('data/data.json')
    return df

def add_fraud(df): # adding a y column for our data, defining fraud as: ['fraudster', 'fraudster_event', 'fraudster_att']
    df['fraud'] = df['acct_type'].apply(lambda x: 1 if x in [u'fraudster_event', u'fraudster', u'fraudster_att'] else 0)
    df.drop('acct_type', axis=1, inplace=True) # this is effectively our y

def unpack_tix(df): # ticket column is a list of dictionaries of unicode; we want that information pulled out into separate columns for analysis
    cost = [] # create empty lists
    quantity_sold = []
    quantity_total = []

    for i in range(len(df['ticket_types'])): # all rows
        line_cost = [] # reset values for new row
        line_quant = []
        line_total = []
        for j in range(len(df['ticket_types'].values[i])): # each separate listing per row
            line_cost.append(df['ticket_types'].values[i][j]['cost']) # add to row list
            line_quant.append(df['ticket_types'].values[i][j]['quantity_sold'])
            line_total.append(df['ticket_types'].values[i][j]['quantity_total'])
        cost.append(line_cost) # add row list to full list
        quantity_sold.append(line_quant)
        quantity_total.append(line_total)

    df['cost'] = cost # set lists as new df columns
    df['quantity_sold'] = quantity_sold
    df['quantity_total'] = quantity_total

def previous_payouts(df): # unpack 'previous payouts' dictionary info
    n = len(df['previous_payouts'])
    n_previous_transactions = []
    user_id = []
    user_state = []
    previous_amounts = [[] for x in range(n)]

    for i in range(n):
        lis = df['previous_payouts'].iloc[i]
        n_previous_transactions.append(len(lis))
        try:
            user_id.append(lis[0]['uid'])
        except:
            user_id.append(np.nan)
        try:
            user_state.append(lis[0]['state'])
        except:
            user_state.append(np.nan)
        for entry in lis:
            previous_amounts[i].append(entry['amount'])
    df['n_previous_transactions'] = n_previous_transactions
    df['user_id'] = user_id
    df['user_state'] = user_state
    df['previous_amounts'] = previous_amounts

# drop extraneous columns
def trim_df(df):
    trim = ['approx_payout_date', 'org_name', 'payout_type', 'payee_name', 'org_desc', 'num_payouts', 'num_order', 'name', 'object_id', 'user_id', 'venue_latitude', 'venue_longitude', 'venue_state', 'venue_address', 'venue_country', 'venue_name', 'cost', 'listed', 'has_header', 'country', 'email_domain', 'event_created', 'event_end', 'event_published', 'event_start', 'description', 'currency', 'gts', 'previous_payouts', 'ticket_types', 'previous_amounts', 'quantity_sold', 'quantity_total', 'user_created', 'user_state']

    for col in trim:
        df.pop(col)

# add columns based on engineered features
def add_cols(df):
    # listed
    df['listed'] = [1 if x == 'y' else 0 for x in df['listed']]
    # avg_amounts
    avg_amounts = []
    for entry in df['previous_amounts']:
        if len(entry) != 0:
            avg_amounts.append(float(sum(entry))/len(entry))
        else:
            avg_amounts.append(0)
    df['avg_amounts'] = avg_amounts
    # avg_cost
    avg_cost = []
    for entry in df['cost']:
        if len(entry) != 0:
            avg_cost.append(float(sum(entry))/len(entry))
        else:
            avg_cost.append(0)
    df['avg_cost'] = avg_cost
    # jason's model preds
    description_pred = np.load('data/description_proba.npy')
    # description_pred = [1 if x == True else 0 for x in description_pred]
    df['description_pred'] = description_pred
    org_description_pred_r = np.load('data/org_description_proba_r.npy')
    # org_description_pred_r = [1 if x == True else 0 for x in org_description_pred_r]
    df['org_description_pred_r'] = org_description_pred_r
    org_description_pred_p = np.load('data/org_description_proba_p.npy')
    # org_description_pred_p = [1 if x == True else 0 for x in org_description_pred_p]
    df['org_description_pred_p'] = org_description_pred_p
    # peter's states
    state = np.load('data/state_schtuff.npy')
    df['diff_state'] = state
    # 0 means same state or NaN, 1 means CLEARLY different state

# convert unix columns into datetime columns, even though we ultimately did not use them
def convert_unix_to_datetime(new_col_name, old_col_name): # converting unix timestamps into datetime format
    df[new_col_name]= pd.to_datetime(df[old_col_name], infer_datetime_format=True, unit='s')
    return df

if __name__ == '__main__':
    unzip_file()
    df = data_to_df()
    unpack_tix(df)
    previous_payouts(df)
    add_fraud(df)
    convert_unix_to_datetime('user_created', 'user_created')
    convert_unix_to_datetime('event_created', 'event_created')
    convert_unix_to_datetime('approx_payout_date', 'approx_payout_date')
    add_cols(df)
    trim_df(df)
    df = df.dropna() # drop rows with NaN values; only a couple hundred
    df.to_pickle('pickles/df.pkl') # pickle df
