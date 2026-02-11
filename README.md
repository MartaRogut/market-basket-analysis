# Market Basket Analysis (Apriori vs FP-Growth)

This project performs Market Basket Analysis on transaction data to discover frequent itemsets and association rules.
It compares two popular algorithms: **Apriori** and **FP-Growth**, and visualizes the strongest rules using support, confidence, and lift.

## What the script does
The script:
1. Loads transactional basket data from a CSV file (`basket_analysis1.csv`)
2. Cleans and converts the dataset to a boolean transaction matrix (0/1 -> False/True)
3. Mines frequent itemsets using:
   - **Apriori**
   - **FP-Growth**
4. Generates association rules from frequent itemsets using:
   - `metric="lift"` and a minimum lift threshold
5. Prints:
   - basic dataset info
   - top frequent itemsets
   - top association rules (sorted by lift)
   - a comparison between Apriori and FP-Growth (top itemsets + number of rules)
6. Produces visualizations:
   - bar chart of top itemsets by **support**
   - bar chart of top rules by **lift**
   - 3D scatter plot of rules (**support**, **confidence**, **lift**)
7. Interprets the strongest rule in plain language (support/confidence/lift meaning)

## Data
**Input file**
- `basket_analysis1.csv`

**Expected format**
- Columns represent products/items
- Rows represent transactions
- Values are converted to numeric then to boolean (non-zero = True)

The script also automatically drops an `Unnamed: 0` column if it exists.

## Key parameters (thresholds)
Configured at the top of the script:
- `MIN_SUPPORT = 0.05`
- `MIN_LIFT = 1.0`
- `MIN_CONFIDENCE_STRONG = 0.5`

Display settings:
- `TOP_N_ITEMS = 10`
- `TOP_N_RULES = 10`
- `TOP_N_3D = 100`

Saving plots:
- `SAVE_PLOTS = False` (set to `True` to save charts)
- `OUTPUT_DIR = "outputs"`

## Technologies
- Python
- pandas
- matplotlib
- mlxtend (`apriori`, `fpgrowth`, `association_rules`)

## How to run
1. Place the files in the same folder:
   - `market-basket-analysis.py`
   - `basket_analysis1.csv`

2. Install dependencies:
```bash
pip install pandas matplotlib mlxtend
