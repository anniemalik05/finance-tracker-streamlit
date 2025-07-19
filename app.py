import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import zipfile

# Create sample transaction data
data = {
    'Date': pd.date_range(start='2023-09-01', periods=20, freq='D'),
    'Description': [
        'Grocery Store', 'Electricity Bill', 'Movie Tickets', 'Restaurant', 'Gas Station',
        'Online Shopping', 'Water Bill', 'Gym Membership', 'Coffee Shop', 'Pharmacy',
        'Supermarket', 'Internet Bill', 'Concert', 'Fast Food', 'Fuel Refill',
        'Clothing Store', 'Mobile Recharge', 'Bookstore', 'Streaming Service', 'Taxi Ride'
    ],
    'Amount': [
        120, 60, 45, 80, 50,
        150, 40, 55, 15, 30,
        110, 70, 90, 25, 45,
        130, 35, 40, 20, 50
    ]
}
df = pd.DataFrame(data)

# Categorize transactions
def categorize(description):
    categories = {
        'Groceries': ['grocery', 'supermarket'],
        'Utilities': ['electricity', 'water', 'internet', 'mobile'],
        'Entertainment': ['movie', 'concert', 'streaming'],
        'Dining': ['restaurant', 'coffee', 'fast food'],
        'Transport': ['gas', 'fuel', 'taxi'],
        'Shopping': ['shopping', 'clothing', 'bookstore'],
        'Health': ['pharmacy', 'gym']
    }
    description = description.lower()
    for category, keywords in categories.items():
        if any(keyword in description for keyword in keywords):
            return category
    return 'Other'

df['Category'] = df['Description'].apply(categorize)

# Budget tracking
total_spent = df['Amount'].sum()
budget_limit = 1000
budget_status = "✅ Within Budget" if total_spent <= budget_limit else "⚠️ Budget Exceeded"

# Visualization 1: Spending by Category
category_totals = df.groupby('Category')['Amount'].sum()
plt.figure(figsize=(8, 6))
category_totals.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Spending by Category')
plt.ylabel('')
plt.tight_layout()
plt.savefig('spending_by_category.png')
plt.close()

# Visualization 2: Expense Trend and Prediction
df['Day'] = np.arange(len(df))
X = df[['Day']]
y = df['Amount']
model = LinearRegression().fit(X, y)
future_days = np.arange(len(df), len(df) + 4).reshape(-1, 1)
predicted_expenses = model.predict(future_days)

plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Amount'], label='Actual Expenses')
future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=4)
plt.plot(future_dates, predicted_expenses, label='Predicted Expenses', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.title('Expense Trend and Prediction')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('expense_prediction.png')
plt.close()

# Save data to CSV
df.to_csv('transaction_data.csv', index=False)

# Save summary to text file
with open('summary.txt', 'w', encoding='utf-8') as f:
    f.write("AI-powered Personal Finance Tracker Summary\n")
    f.write("==========================================\n")
    f.write(f"Total Spent: ${total_spent:.2f}\n")
    f.write(f"Budget Limit: ${budget_limit}\n")
    f.write(f"Status: {budget_status}\n")

# Create ZIP file
with zipfile.ZipFile('finance_tracker_project.zip', 'w') as zipf:
    zipf.write('transaction_data.csv')
    zipf.write('spending_by_category.png')
    zipf.write('expense_prediction.png')
    zipf.write('summary.txt')
print("Project packaged into 'finance_tracker_project.zip' successfully.")