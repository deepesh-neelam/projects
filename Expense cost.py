import json
from datetime import datetime

expenses = []

def add_expense():
    amount = float(input("Enter amount: "))
    category = input("Enter category (Food, Travel, etc): ")
    note = input("Enter note: ")
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    expense = {
        "amount": amount,
        "category": category,
        "note": note,
        "date": date
    }

    expenses.append(expense)
    print("Expense added successfully ✅")

def view_expenses():
    if not expenses:
        print("No expenses recorded.")
        return

    print("\n--- All Expenses ---")
    for exp in expenses:
        print(f"{exp['date']} | {exp['category']} | ${exp['amount']} | {exp['note']}")

def total_spent():
    total = sum(exp["amount"] for exp in expenses)
    print(f"\nTotal Spent: ${total}")

def category_summary():
    summary = {}
    for exp in expenses:
        summary[exp["category"]] = summary.get(exp["category"], 0) + exp["amount"]

    print("\n--- Category Summary ---")
    for category, amount in summary.items():
        print(f"{category}: ${amount}")

def save_data():
    with open("expenses.json", "w") as file:
        json.dump(expenses, file)
    print("Data saved to expenses.json 💾")

def load_data():
    global expenses
    try:
        with open("expenses.json", "r") as file:
            expenses = json.load(file)
        print("Data loaded successfully 📂")
    except FileNotFoundError:
        print("No saved data found.")

# Load existing data at start
load_data()

while True:
    print("\n==== Expense Tracker ====")
    print("1. Add Expense")
    print("2. View Expenses")
    print("3. Total Spent")
    print("4. Category Summary")
    print("5. Save Data")
    print("6. Exit")

    choice = input("Choose option (1-6): ")

    if choice == '1':
        add_expense()
    elif choice == '2':
        view_expenses()
    elif choice == '3':
        total_spent()
    elif choice == '4':
        category_summary()
    elif choice == '5':
        save_data()
    elif choice == '6':
        save_data()
        print("Goodbye 👋")
        break
    else:
        print("Invalid choice.")