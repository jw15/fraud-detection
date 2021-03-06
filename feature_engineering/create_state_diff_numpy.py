import pandas as pd
import numpy as np

def data_to_df(): # read json data into dataframe
    df = pd.read_json('../data/data.json')
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

def previous_payouts(df): # unpacking 'previous payouts' info
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

# comically long function to get 'venue_state' and 'user_state' on the same page;
# ideally, you'd want some sort of database for autocorrections or even more ideally,
# just use drop-down menus for location
def clean_states(df):
    df['user_state'] = df['user_state'].str.upper() # set all names to uppercase
    df['venue_state'] = df['venue_state'].str.upper() # for comparison purposes
    df['venue_state'].fillna(value=np.nan, inplace=True) # replace None with NaN
    df['user_state'].replace('JAKARTA PUSAT', 'INDONESIA',inplace=True)
    df['venue_state'].replace('DKI JAKARTA', 'INDONESIA',inplace=True)
    df['venue_state'].replace('JAKARTA', 'INDONESIA',inplace=True)
    df['venue_state'].replace('MAHARASHTRA', 'INDIA',inplace=True)
    df['venue_state'].replace('BALI', 'INDONESIA',inplace=True)
    df['venue_state'].replace('CENTRAL JAVA', 'INDONESIA',inplace=True)
    df['venue_state'].replace('WEST JAVA', 'INDONESIA',inplace=True)
    df['venue_state'].replace('CALIFORNIA', 'CA',inplace=True)
    df['user_state'].replace('INNOVIT', 'ICELAND',inplace=True)
    df['venue_state'].replace('GULLBRINGUSYSLA', 'ICELAND',inplace=True)
    df['venue_state'].replace('CUNDINAMARCA', 'COLOMBIA',inplace=True)
    df['user_state'].replace('', np.nan,inplace=True)
    df['user_state'].replace(' ', np.nan,inplace=True)
    df['user_state'].replace('OXFORDSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('OXFORDSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('OXON', 'UK',inplace=True)
    df['venue_state'].replace('ESSEX', 'UK',inplace=True)
    df['venue_state'].replace('WILTSHIRE', 'UK',inplace=True)
    df['user_state'].replace('STAFFORDSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('STAFFORDSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('LIVERPOOL', 'UK',inplace=True)
    df['venue_state'].replace('EAST RIDING OF YORKSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('WEST YORKSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('NORTH YORKSHIRE', 'UK',inplace=True)
    df['user_state'].replace('SOUTH AUSTRALIA', 'AUSTRALIA',inplace=True)
    df['user_state'].replace('BANGKOK', 'THAILAND',inplace=True)
    df['user_state'].replace('LONDON', 'UK',inplace=True)
    df['venue_state'].replace('GREATER LONDON', 'UK',inplace=True)
    df['venue_state'].replace('MERSYD', 'UK',inplace=True)
    df['venue_state'].replace('SOUTHEND-ON-SEA', 'UK',inplace=True)
    df['venue_state'].replace('SOUTHEND ON SEA', 'UK',inplace=True)
    df['venue_state'].replace('GREATER LONDON', 'UK',inplace=True)
    df['venue_state'].replace('BRIGHTON AND HOVE', 'UK',inplace=True)
    df['venue_state'].replace('THE CITY OF BRIGHTON AND HOVE', 'UK',inplace=True)
    df['venue_state'].replace('SUNDERLAND', 'UK',inplace=True)
    df['venue_state'].replace('WOKINGHAM', 'UK',inplace=True)
    df['venue_state'].replace('CAMBS', 'UK',inplace=True)
    df['venue_state'].replace('CNWLL', 'UK',inplace=True)
    df['venue_state'].replace('CORNWALL', 'UK',inplace=True)
    df['venue_state'].replace('NORTHUMBERLAND', 'UK',inplace=True)
    df['venue_state'].replace('DURHAM', 'UK',inplace=True)
    df['venue_state'].replace('BERKSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('KINGSTON UPON HULL', 'UK',inplace=True)
    df['venue_state'].replace('MILTON KEYNES', 'UK',inplace=True)
    df['venue_state'].replace('MIDDLESEX', 'UK',inplace=True)
    df['venue_state'].replace('WEST YORK', 'UK',inplace=True)
    df['venue_state'].replace('BUCKINGHAMSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('HAMPSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('MEDWAY', 'UK',inplace=True)
    df['venue_state'].replace('NORTH YORK', 'UK',inplace=True)
    df['venue_state'].replace('COVENTRY', 'UK',inplace=True)
    df['venue_state'].replace('CHESHIRE', 'UK',inplace=True)
    df['venue_state'].replace('BEDS', 'UK',inplace=True)
    df['venue_state'].replace('SWINTON', 'UK',inplace=True)
    df['venue_state'].replace('WOLVERHAMPTON', 'UK',inplace=True)
    df['venue_state'].replace('LUTON', 'UK',inplace=True)
    df['venue_state'].replace('NOORD HOLLAND', 'NETHERLANDS',inplace=True)
    df['venue_state'].replace('ZUID HOLLAND', 'NETHERLANDS',inplace=True)
    df['venue_state'].replace('ZH', 'NETHERLANDS',inplace=True)
    df['venue_state'].replace('FLEVOLAND', 'NETHERLANDS',inplace=True)
    df['venue_state'].replace('UTRECHT', 'NETHERLANDS',inplace=True)
    df['venue_state'].replace('GALWAY', 'IRELAND',inplace=True)
    df['venue_state'].replace('WICKLOW', 'IRELAND',inplace=True)
    df['venue_state'].replace('CORK', 'IRELAND',inplace=True)
    df['venue_state'].replace('KERRY', 'IRELAND',inplace=True)
    df['venue_state'].replace('CLARE', 'IRELAND',inplace=True)
    df['venue_state'].replace('ST THOMAS', 'VIRGIN ISLANDS',inplace=True)
    df['venue_state'].replace('BA', 'AR',inplace=True)
    df['venue_state'].replace('WIEN', 'AUSTRIA',inplace=True)
    df['venue_state'].replace('VIENNA', 'AUSTRIA',inplace=True)
    df['venue_state'].replace('GRAND CASABLANCA', 'MOROCCO',inplace=True)
    df['venue_state'].replace('SOUSS-MASSA-DRAA', 'MOROCCO',inplace=True)
    df['venue_state'].replace('DOUKKALA-ABDA', 'MOROCCO',inplace=True)
    df['venue_state'].replace('TANGIER-TETOUAN', 'MOROCCO',inplace=True)
    df['venue_state'].replace('MARRAKESH-TENSIFT-AL HAOUZ', 'MOROCCO',inplace=True)
    df['venue_state'].replace('MANAGUA', 'NICARAGUA',inplace=True)
    df['venue_state'].replace('ESTELI', 'NICARAGUA',inplace=True)
    df['venue_state'].replace('MONTEVIDEO', 'URUGUAY',inplace=True)
    df['venue_state'].replace('', np.nan,inplace=True)
    df['venue_state'].replace(' ', np.nan,inplace=True)
    df['venue_state'].replace('None', np.nan,inplace=True)
    df['venue_state'].replace('ONTARIO', 'ON',inplace=True)
    df['venue_state'].replace('ON.', 'ON',inplace=True)
    df['venue_state'].replace('ON K7H 3C6', 'ON',inplace=True)
    df['venue_state'].replace('HAMILTON', 'ON',inplace=True)
    df['venue_state'].replace('BRITISH COLUMBIA', 'BC',inplace=True)
    df['venue_state'].replace('NEWFOUNDLAND AND LABRADOR', 'NL',inplace=True)
    df['venue_state'].replace('SA', 'AUSTRALIA',inplace=True)
    df['venue_state'].replace('SOUTH AUSTRALIA', 'AUSTRALIA',inplace=True)
    df['venue_state'].replace('NT', 'AUSTRALIA',inplace=True)
    df['venue_state'].replace('NORTHERN TERRITORY', 'AUSTRALIA',inplace=True)
    df['venue_state'].replace('SWANSEA', 'WALES',inplace=True)
    df['venue_state'].replace('CARDIFF', 'WALES',inplace=True)
    df['venue_state'].replace('CEREDIGION', 'WALES',inplace=True)
    df['venue_state'].replace('CAERPHILLY', 'WALES',inplace=True)
    df['venue_state'].replace('MERTHYR TYDFIL', 'WALES',inplace=True)
    df['venue_state'].replace('GWYND', 'WALES',inplace=True)
    df['venue_state'].replace('AD DAWHAH', 'QATAR',inplace=True)
    df['venue_state'].replace('FARO', 'PORTUGAL',inplace=True)
    df['venue_state'].replace('LISBON', 'PORTUGAL',inplace=True)
    df['venue_state'].replace('KENT', 'UK',inplace=True)
    df['venue_state'].replace('GLOUCESTERSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('RICHMOND UPON THAMES', 'UK',inplace=True)
    df['venue_state'].replace('LONDON BOROUGH OF ENFIELD', 'UK',inplace=True)
    df['venue_state'].replace('LONDON BOROUGH OF CAMDEN', 'UK',inplace=True)
    df['venue_state'].replace('GT MAN', 'UK',inplace=True)
    df['venue_state'].replace('SALFORD', 'UK',inplace=True)
    df['venue_state'].replace('PORTSMOUTH', 'UK',inplace=True)
    df['venue_state'].replace('GLOS', 'UK',inplace=True)
    df['venue_state'].replace('DERRY', 'N-IRELAND',inplace=True)
    df['venue_state'].replace('BELFAST', 'N-IRELAND',inplace=True)
    df['venue_state'].replace('SOR-TRONDELAG', 'NORWAY',inplace=True)
    df['venue_state'].replace('EDINBURGH, CITY OF', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('CITY OF EDINBURGH', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('SCOTTISH BORDERS', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('SCOTTISH BORDERS, THE', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('ABERDEEN CITY', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('MIDLOTHIAN', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('FIFE', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('EAST RENFREWSHIRE', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('SAINT JAMES', np.nan,inplace=True)
    df['venue_state'].replace('TEL AVIV', 'ISRAEL',inplace=True)
    df['venue_state'].replace('WALLOON REGION', 'BELGIUM',inplace=True)
    df['venue_state'].replace('BRUSSELS', 'BELGIUM',inplace=True)
    df['venue_state'].replace('VLAAMS GEWEST', 'BELGIUM',inplace=True)
    df['venue_state'].replace('ANTWERPEN', 'BELGIUM',inplace=True)
    df['venue_state'].replace('BRUSSELS HOOFDSTEDELIJK GEWEST', 'BELGIUM',inplace=True)
    df['venue_state'].replace('BRUXELLES', 'BELGIUM',inplace=True)
    df['venue_state'].replace('GP', 'AL',inplace=True)
    df['venue_state'].replace('DUBLIN', 'IRELAND',inplace=True)
    df['venue_state'].replace('COUNTY DUBLIN', 'IRELAND',inplace=True)
    df['venue_state'].replace('CO. KILDARE', 'IRELAND',inplace=True)
    df['venue_state'].replace('KILDARE', 'IRELAND',inplace=True)
    df['venue_state'].replace('SALTA', 'ARGENTINA',inplace=True)
    df['venue_state'].replace('TIBURON', 'CA',inplace=True)
    df['venue_state'].replace('SUFFOLK', 'UK',inplace=True)
    df['venue_state'].replace('SOMERSET', 'UK',inplace=True)
    df['venue_state'].replace('NOTTINGHAM', 'UK',inplace=True)
    df['venue_state'].replace('WARRINGTON', 'UK',inplace=True)
    df['venue_state'].replace('MANCHESTER', 'UK',inplace=True)
    df['venue_state'].replace('BIRMINGHAM', 'UK',inplace=True)
    df['venue_state'].replace('SHEFFIELD', 'UK',inplace=True)
    df['venue_state'].replace('BOURNEMOUTH', 'UK',inplace=True)
    df['venue_state'].replace('SOUTH YORK', 'UK',inplace=True)
    df['venue_state'].replace('WEST MID', 'UK',inplace=True)
    df['venue_state'].replace('WEST MIDS', 'UK',inplace=True)
    df['venue_state'].replace('WEST MIDLANDS', 'UK',inplace=True)
    df['venue_state'].replace('SURREY', 'UK',inplace=True)
    df['venue_state'].replace('ENGLAND', 'UK',inplace=True)
    df['venue_state'].replace('NEWCASTLE UPON TYNE', 'UK',inplace=True)
    df['venue_state'].replace('LONDON, CITY OF', 'UK',inplace=True)
    df['venue_state'].replace('CITY OF LONDON', 'UK',inplace=True)
    df['venue_state'].replace('CITY OF LONDON, THE', 'UK',inplace=True)
    df['venue_state'].replace('LONDON BOROUGH OF ISLINGTON', 'UK',inplace=True)
    df['venue_state'].replace('LONDON BOROUGH OF TOWER HAMLETS', 'UK',inplace=True)
    df['venue_state'].replace('LONDON BOROUGH OF LAMBETH', 'UK',inplace=True)
    df['venue_state'].replace('LONDON BOROUGH OF LEWISHAM', 'UK',inplace=True)
    df['venue_state'].replace('KINGSTON UPON THAMES', 'UK',inplace=True)
    df['venue_state'].replace('DEVON', 'UK',inplace=True)
    df['venue_state'].replace('LEICS', 'UK',inplace=True)
    df['venue_state'].replace('NORTHERN IRELAND', 'N-IRELAND',inplace=True)
    df['venue_state'].replace('ANTRIM', 'N-IRELAND',inplace=True)
    df['venue_state'].replace('CO ANTRIM', 'N-IRELAND',inplace=True)
    df['venue_state'].replace(u'STOCKHOLMS L\xc4N', 'SWEDEN',inplace=True)
    df['venue_state'].replace('STOCKHOLM COUNTY', 'SWEDEN',inplace=True)
    df['venue_state'].replace('CAPITAL REGION OF DENMARK', 'DENMARK',inplace=True)
    df['venue_state'].replace(u'ETEL\xc4-SUOMI', 'FINLAND',inplace=True)
    df['venue_state'].replace('MADRID', 'SPAIN',inplace=True)
    df['venue_state'].replace('COMUNIDAD DE MADRID', 'SPAIN',inplace=True)
    df['venue_state'].replace('COMMUNITY OF MADRID', 'SPAIN',inplace=True)
    df['venue_state'].replace('PM', 'SPAIN',inplace=True)
    df['venue_state'].replace('CATALONIA', 'SPAIN',inplace=True)
    df['venue_state'].replace('WARSZAWA', 'POLAND',inplace=True)
    df['venue_state'].replace('MICHIGAN', 'MI',inplace=True)
    df['venue_state'].replace('AUSTRALIAN CAPITAL TERRITORY', 'AUSTRALIA',inplace=True)
    df['venue_state'].replace('WESTERN AUSTRALIA', 'AUSTRALIA',inplace=True)
    df['venue_state'].replace('ACT', 'AUSTRALIA',inplace=True)
    df['venue_state'].replace('TASMANIA', 'AUSTRALIA',inplace=True)
    df['venue_state'].replace('TAS', 'AUSTRALIA',inplace=True)
    df['venue_state'].replace('ALGIERS', 'ALGERIA',inplace=True)
    df['venue_state'].replace('BRISTOL, CITY OF', 'UK',inplace=True)
    df['venue_state'].replace('CITY OF BRISTOL', 'UK',inplace=True)
    df['venue_state'].replace('CITY OF WESTMINSTER', 'UK',inplace=True)
    df['venue_state'].replace('KRUNG THEP', 'THAILAND',inplace=True)
    df['venue_state'].replace('RANONG', 'THAILAND',inplace=True)
    df['venue_state'].replace('PHUKET', 'THAILAND',inplace=True)
    df['venue_state'].replace('ALAKSA', 'AK',inplace=True)
    df['venue_state'].replace('DUBAI', 'UAE',inplace=True)
    df['venue_state'].replace('ABU DHABI', 'UAE',inplace=True)
    df['venue_state'].replace('VIRGINIA', 'VA',inplace=True)
    df['venue_state'].replace('VIRGINA', 'VA',inplace=True)
    df['venue_state'].replace('KENTUCKY', 'KY',inplace=True)
    df['venue_state'].replace('MASSACHUSETTS', 'MA',inplace=True)
    df['venue_state'].replace('MANITOBA', 'MB',inplace=True)
    df['venue_state'].replace('NEW BRUNSWICK', 'NB',inplace=True)
    df['venue_state'].replace('SASKATCHEWAN', 'SK',inplace=True)
    df['venue_state'].replace('PETERBOROUGH', 'UK',inplace=True)
    df['venue_state'].replace('MO.', 'MO',inplace=True)
    df['venue_state'].replace('OHIO', 'OH',inplace=True)
    df['venue_state'].replace('NORTH CAROLINA', 'NC',inplace=True)
    df['venue_state'].replace('SOUTH CAROLINA', 'SC',inplace=True)
    df['venue_state'].replace('DISTRICT OF COLUMBIA', 'DC',inplace=True)
    df['venue_state'].replace('D.C.', 'DC',inplace=True)
    df['venue_state'].replace('FL 3220', 'FL',inplace=True)
    df['venue_state'].replace('FLORIDA (DOWNTOWN', 'FL',inplace=True)
    df['venue_state'].replace('FLORIDA', 'FL',inplace=True)
    df['venue_state'].replace('PENNSYLVANIA 19149', 'PA',inplace=True)
    df['venue_state'].replace('IOW', 'IA',inplace=True)
    df['venue_state'].replace('HAWAII  96815', 'HI',inplace=True)
    df['venue_state'].replace('VERMONT', 'VT',inplace=True)
    df['venue_state'].replace('TEXAS', 'TX',inplace=True)
    df['venue_state'].replace('NOVA SCOTIA', 'NS',inplace=True)
    df['venue_state'].replace('NORFK', 'UK',inplace=True)
    df['venue_state'].replace('NORFOLK', 'UK',inplace=True)
    df['venue_state'].replace('MIDDLESBROUGH', 'UK',inplace=True)
    df['venue_state'].replace('READING', 'UK',inplace=True)
    df['venue_state'].replace('EAST DUNBARTONSHIRE', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('BEDFORDSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('STIRLING', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('CENTRAL BEDFORDSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('PEMBROKESHIRE', 'UK',inplace=True)
    df['venue_state'].replace('SUFFK', 'UK',inplace=True)
    df['venue_state'].replace('SUFFOLK COUNTY', 'UK',inplace=True)
    df['venue_state'].replace('T & W', 'UK',inplace=True)
    df['venue_state'].replace('PROVENCE ALPES COTE D&AMP;AMP;AMP;AMP;AMP;AMP;AMP;', 'FRANCE',inplace=True)
    df['venue_state'].replace('PACA', 'FRANCE',inplace=True)
    df['venue_state'].replace('IDF', 'FRANCE',inplace=True)
    df['venue_state'].replace('LANGUEDOC-ROUSSILLON', 'FRANCE',inplace=True)
    df['venue_state'].replace('PAYS DE LA LOIRE', 'FRANCE',inplace=True)
    df['venue_state'].replace('PCH', np.nan,inplace=True)
    df['venue_state'].replace('PHNOM PENH', 'CAMBODIA',inplace=True)
    df['venue_state'].replace('BUDAPEST', 'HUNGARY',inplace=True)
    df['venue_state'].replace('AUSTRALIAN CAPITAL TERRITORY;', 'AUSTRALIA',inplace=True)
    df['venue_state'].replace('ROYAL BOROUGH OF KENSINGTON AND CHELSEA', 'UK',inplace=True)
    df['venue_state'].replace('PUERTO RICO', 'PR',inplace=True)
    df['venue_state'].replace('SAN JUAN', 'PR',inplace=True)
    df['venue_state'].replace('AGUADA', 'PR',inplace=True)
    df['venue_state'].replace('NEW JERSEY', 'NJ',inplace=True)
    df['venue_state'].replace('NOVA SCOTIA', 'NS',inplace=True)
    df['venue_state'].replace('LAZIO', 'ITALY',inplace=True)
    df['venue_state'].replace('REDCAR AND CLEVELAND', 'UK',inplace=True)
    df['venue_state'].replace('CLEVELAND', 'UK',inplace=True)
    df['venue_state'].replace('LONDON', 'UK',inplace=True)
    df['venue_state'].replace('GT LON', 'UK',inplace=True)
    df['venue_state'].replace('CHESHIRE WEST AND CHESTER', 'UK',inplace=True)
    df['venue_state'].replace('WIRRAL', 'UK',inplace=True)
    df['venue_state'].replace('DERBY', 'UK',inplace=True)
    df['venue_state'].replace('DERBYSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('WEST BERKSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('CUMB', 'UK',inplace=True)
    df['venue_state'].replace('NEW SOUTH WALES', 'AUSTRALIA',inplace=True)
    df['venue_state'].replace('NSW', 'AUSTRALIA',inplace=True)
    df['venue_state'].replace('ALBERTA', 'AB',inplace=True)
    df['venue_state'].replace('ILE DE FRANCE', 'FRANCE',inplace=True)
    df['venue_state'].replace('MONACO', 'FRANCE',inplace=True)
    df['venue_state'].replace('ALSACE', 'FRANCE',inplace=True)
    df['venue_state'].replace('SAINT-PAUL', 'FRANCE',inplace=True)
    df['venue_state'].replace('AUCKLAND', 'NZ',inplace=True)
    df['venue_state'].replace('WHANGAREI', 'NZ',inplace=True)
    df['venue_state'].replace('HOKITIKA', 'NZ',inplace=True)
    df['venue_state'].replace('CHRISTCHURCH', 'NZ',inplace=True)
    df['venue_state'].replace('WELLINGTON', 'NZ',inplace=True)
    df['venue_state'].replace('WAITAKERE', 'NZ',inplace=True)
    df['venue_state'].replace('OTAGO', 'NZ',inplace=True)
    df['venue_state'].replace('TAKAKA', 'NZ',inplace=True)
    df['venue_state'].replace('CANADA', 'BC',inplace=True)
    df['venue_state'].replace('QLD', 'AUSTRALIA',inplace=True)
    df['venue_state'].replace('QUEENSLAND', 'AUSTRALIA',inplace=True)
    df['venue_state'].replace('NJ 07102-4398', 'NJ',inplace=True)
    df['venue_state'].replace('PENNSYLVANIA', 'PA',inplace=True)
    df['venue_state'].replace('QUEBEC', 'QC',inplace=True)
    df['venue_state'].replace('CA 92590', 'CA',inplace=True)
    df['venue_state'].replace('CA 94704', 'CA',inplace=True)
    df['venue_state'].replace('CA 94607', 'CA',inplace=True)
    df['venue_state'].replace('NY 11570', 'NY',inplace=True)
    df['venue_state'].replace('TOKYO', 'JAPAN',inplace=True)
    df['venue_state'].replace('BERLIN', 'GERMANY',inplace=True)
    df['venue_state'].replace('BAYERN', 'GERMANY',inplace=True)
    df['venue_state'].replace('BW', 'GERMANY',inplace=True)
    df['venue_state'].replace('NRW', 'GERMANY',inplace=True)
    df['venue_state'].replace('BY', 'GERMANY',inplace=True)
    df['venue_state'].replace('HE', 'GERMANY',inplace=True)
    df['venue_state'].replace('NDS', 'GERMANY',inplace=True)
    df['venue_state'].replace('HAMBURG', 'GERMANY',inplace=True)
    df['venue_state'].replace('NORDRHEIN WESTFALEN', 'GERMANY',inplace=True)
    df['venue_state'].replace('HESSEN', 'GERMANY',inplace=True)
    df['venue_state'].replace('BEIJING', 'CHINA',inplace=True)
    df['venue_state'].replace('SHANGHAI', 'CHINA',inplace=True)
    df['venue_state'].replace('VICTORIA', 'AUSTRALIA',inplace=True)
    df['venue_state'].replace('VIC', 'AUSTRALIA',inplace=True)
    df['venue_state'].replace('WARKS', 'UK',inplace=True)
    df['venue_state'].replace('CUMBRIA', 'UK',inplace=True)
    df['venue_state'].replace('LEICESTER', 'UK',inplace=True)
    df['venue_state'].replace('SHROPSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('CAMBRIDGESHIRE', 'UK',inplace=True)
    df['venue_state'].replace('BATH AND NORTH EAST SOMERSET', 'UK',inplace=True)
    df['venue_state'].replace('NORTH SOMERSET', 'UK',inplace=True)
    df['venue_state'].replace('CANTERBURY', 'UK',inplace=True)
    df['venue_state'].replace('LONDON BOROUGH OF CROYDON', 'UK',inplace=True)
    df['venue_state'].replace('PERTH AND KINROSS', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('CLACKMANNANSHIRE', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('FALKIRK', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('CO. WATERFORD', 'IRELAND',inplace=True)
    df['venue_state'].replace('COUNTY WATERFORD', 'IRELAND',inplace=True)
    df['venue_state'].replace('WATERFORD', 'IRELAND',inplace=True)
    df['venue_state'].replace('VORONEZHSKAYA OBLAST', 'RUSSIA',inplace=True)
    df['venue_state'].replace('NOTTINGHAMSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('NOTTS', 'UK',inplace=True)
    df['venue_state'].replace('WORCESTERSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('STOKE ON TRENT', 'UK',inplace=True)
    df['venue_state'].replace('STOKE-ON-TRENT', 'UK',inplace=True)
    df['venue_state'].replace('DUR', 'UK',inplace=True)
    df['venue_state'].replace('ISLINGTON', 'UK',inplace=True)
    df['venue_state'].replace('BRACKNELL FOREST', 'UK',inplace=True)
    df['venue_state'].replace('SOUTHAMPTON', 'UK',inplace=True)
    df['venue_state'].replace('NORTHAMPTONSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('THURROCK', 'UK',inplace=True)
    df['venue_state'].replace('SOLIHULL', 'UK',inplace=True)
    df['venue_state'].replace('LINCS', 'UK',inplace=True)
    df['venue_state'].replace('LINCOLN', 'UK',inplace=True)
    df['venue_state'].replace('ISTANBUL', 'TURKEY',inplace=True)
    df['venue_state'].replace('NEW PROVIDENCE', 'BAHAMAS',inplace=True)
    df['venue_state'].replace('OUEST', 'HAITI',inplace=True)
    df['venue_state'].replace('CORDILLERA ADMINISTRATIVE REGION', 'PHILIPPINES',inplace=True)
    df['venue_state'].replace('CAR', 'PHILIPPINES',inplace=True)
    df['venue_state'].replace('NCR', 'PHILIPPINES',inplace=True)
    df['venue_state'].replace('BLACKPOOL', 'UK',inplace=True)
    df['venue_state'].replace('LEEDS', 'UK',inplace=True)
    df['venue_state'].replace('HERTFORDSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('HERTFORD', 'UK',inplace=True)
    df['venue_state'].replace('POOLE', 'UK',inplace=True)
    df['venue_state'].replace('LANCASHIRE', 'UK',inplace=True)
    df['venue_state'].replace('KIRKLEES', 'UK',inplace=True)
    df['venue_state'].replace('BELGRAVIA', 'UK',inplace=True)
    df['venue_state'].replace('YORK', 'UK',inplace=True)
    df['venue_state'].replace('BRADFORD', 'UK',inplace=True)
    df['venue_state'].replace('EAST SUSSEX', 'UK',inplace=True)
    df['venue_state'].replace('WEST SUSSEX', 'UK',inplace=True)
    df['venue_state'].replace('ST HELIER', 'UK',inplace=True)
    df['venue_state'].replace('WARWICKSHIRE', 'UK',inplace=True)
    df['venue_state'].replace('LEICESTERSHIRE COUNTY', 'UK',inplace=True)
    df['venue_state'].replace('DUDLEY', 'UK',inplace=True)
    df['venue_state'].replace('DORSET', 'UK',inplace=True)
    df['venue_state'].replace('PLYMOUTH', 'UK',inplace=True)
    df['venue_state'].replace('ERONGO', 'NAMIBIA',inplace=True)
    df['venue_state'].replace('DISTRITO FEDERAL', 'MEXICO',inplace=True)
    df['venue_state'].replace('D.F.', 'MEXICO',inplace=True)
    df['venue_state'].replace('GUANAJUATO', 'MEXICO',inplace=True)
    df['venue_state'].replace('WESTERN CAPE', 'S-AFRICA',inplace=True)
    df['venue_state'].replace('KZN', 'S-AFRICA',inplace=True)
    df['venue_state'].replace('HO CHI MINH CITY', 'VIETNAM',inplace=True)
    df['venue_state'].replace('KHANH HOA PROVINCE', 'VIETNAM',inplace=True)
    df['venue_state'].replace('QUANG BINH PROVINCE', 'VIETNAM',inplace=True)
    df['venue_state'].replace('TIEN GANG', 'VIETNAM',inplace=True)
    df['venue_state'].replace('TIEN GIANG', 'VIETNAM',inplace=True)
    df['venue_state'].replace('HANOI', 'VIETNAM',inplace=True)
    df['venue_state'].replace('SINDH', 'PAKISTAN',inplace=True)
    df['venue_state'].replace('PUNJAB', 'PAKISTAN',inplace=True)
    df['venue_state'].replace('KARACHI DISTRICT', 'PAKISTAN',inplace=True)
    df['venue_state'].replace('HAUTE NORMANDIE', 'FRANCE',inplace=True)
    df['venue_state'].replace('BURGANDY', 'FRANCE',inplace=True)
    df['venue_state'].replace('RA', 'FRANCE',inplace=True)
    df['venue_state'].replace('AUVERGNE', 'FRANCE',inplace=True)
    df['venue_state'].replace('RHONE ALPES', 'FRANCE',inplace=True)
    df['venue_state'].replace('GLASGOW CITY', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('LANARKSHIRE', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('SOUTH AYRSHIRE', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('NORTH LANARKSHIRE', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('DUNDEE CITY', 'SCOTLAND',inplace=True)
    df['venue_state'].replace('WAIKATO', 'NZ',inplace=True)
    df['venue_state'].replace('MANAWATU WANGANUI', 'NZ',inplace=True)
    df['venue_state'].replace('BAY OF PLENTY', 'NZ',inplace=True)
    df['venue_state'].replace('CITY OF ZAGREB', 'CROATIA',inplace=True)
    df['venue_state'].replace('LAGOS', 'NIGERIA',inplace=True)
    df['venue_state'].replace('KHOMAS', 'NAMIBIA',inplace=True)
    df['venue_state'].replace('NAIROBI', 'KENYA',inplace=True)
    df['venue_state'].replace('NAKURU', 'KENYA',inplace=True)
    df['venue_state'].replace('CENTRE', 'CAMEROON',inplace=True)
    df['venue_state'].replace('MUSCAT', 'OMAN',inplace=True)
    df['venue_state'].replace('FAMAGUSTA', 'CYPRUS',inplace=True)
    df['user_state'] = df['user_state'].astype(str)
    df['venue_state'] = df['venue_state'].astype(str)

def check_location(df): # comparing 'venue_state' vs. 'user_state' to identify observable differences
    schtuff = []
    for i in range(len(df)):
        if df['venue_state'][i] == 'nan' or df['user_state'][i] == 'nan':
            schtuff.append(0)
        elif df['venue_state'][i] == df['user_state'][i]:
            schtuff.append(0)
        elif df['venue_state'][i] != df['user_state'][i]:
            schtuff.append(1)
    return schtuff

if __name__ == '__main__':
    df = data_to_df()
    unpack_tix(df)
    previous_payouts(df)
    clean_states(df)
    schtuff = check_location(df)
    schtuff = np.array(schtuff)
    np.save('../data/state_schtuff.npy', schtuff) # saving results as numpy array for export
