## Identifying Fraud

### Premise

As contract data scientists/consultants hired by a new e-commerce site, our goal was to try to weed out fraudsters.

### Goals:

 1. Build a model that identifies potential fraud cases.
 2. Build a sustainable software project that we could hand off to the company engineers by deploying our model in the cloud.

### Step 1: EDA and Data Cleaning

1. Convert unix timestamp to datetime
2. Expand columns that contain dictionaries:
    * 'ticket_types'
    * 'previous_payouts'
3. Feature engineering
    * Compared 'venue_state' to 'user_state' to create 'diff_state' binary value
        * This helped our model, but the signal was in the opposite direction of what we expected <br />
        (different states = less likely to be fraud)
        * We suspect this may be due to the amount of information provided, with fraudsters being less likely to be forthcoming with location data
    * Added columns with predicted probabilities derived from TF-IDF analysis of event and organization descriptions (see: below)
4. Removed extraneous columns, especially those that might create leakage (looking at you, 'acct_type')

### Step 2: Building Preliminary Models

1. NLP
    * TF-IDF analysis of event descriptions to predict fraud using Naive Bayes.
        * First run: Recall = 37%, Precision = 43%
    * Topic modeling: Maybe one of the topics indicates fraud?
        * Decided not to pursue
    * TF-IDF analysis of organization descriptions using Naive Bayes
        * First run (parsed HTML, did not use stop words): Recall = 85%, Precision = 15%
        * Second run (did not parse HTML, did use stop words): Recall = 14%, Precision = 59%

2. Random Forest
    * Implemented a standard 75/25 train/test split on our data
    * Used a Random Forest Classifier to predict fraud based on our dataframe, which now included the above engineered features
        * Recall: 92%, Precision: 95%

3. Gradient Boost
    * Attempted to use XGBoost to improve upon our Random Forest results
        * Recall: 91%, Precision: 96%
    * Given that we are prioritizing recall (ability to accurately diagnose real fraud) over precision (only diagnosing fraud when fraud is present), XGBoost did not quite improve upon our model

### Step 3: Build Website via Flask

We ran out of time to get the website working on an EC2 instance. In retrospect, we needed more time and manpower dedicated to making the web app than we allocated. There are still some bugs related to the way the pickling was done (unicode error), but it should work once those are sorted out.

### Authors
This work was a collaboration between Jennifer Waller (github.com/jw15), Peter Nygaard (github.com/RetepAdam), Jason Olson, and Doug Schuster (github.com/schustda).
README written by Peter Nygaard and Jen Waller.
