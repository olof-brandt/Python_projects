"""
This script calculates the minimum hourly rate required for a freelancer or a business owner
to cover all costs, taxes, and savings goals. The calculation takes into account private expenses,
desired savings, business expenses, social security contributions (social fees), and municipal tax.
The goal is to determine a rate that ensures all costs are covered while achieving the desired net income.

The process involves:
1. Determining the required net salary (after taxes) based on private expenses and savings.
2. Calculating the gross salary needed before tax to achieve this net salary.
3. Computing social security contributions based on the gross salary.
4. Summing all costs (business expenses, gross salary, social fees) to find the total gross income needed.
5. Dividing the total gross income by the number of billable hours per month to find the hourly rate.
"""

def calculate_hourly_rate(business_expenses, private_expenses, savings, communal_tax, social_fees):
    """
    Calculates the minimum hourly rate needed to cover costs and savings.

    Parameters:
    - business_expenses: Fixed business costs per month (in currency units)
    - private_expenses: Personal expenses per month (in currency units)
    - savings: Desired savings per month (in currency units)
    - communal_tax: Municipal tax percentage (e.g., 33 for 33%)
    - social_fees: Social security contribution percentage (e.g., 28.8 for 28.8%)

    Returns:
    - hourly_rate: The minimum rate per hour needed to cover all costs
    """

    # Number of billable hours per month
    monthly_hours = 160

    # Calculate the required net salary (after tax) to cover private expenses and savings
    required_net_salary = private_expenses + savings

    # Calculate the gross salary before tax needed to achieve the required net salary
    gross_salary_before_tax = required_net_salary / (1 - communal_tax / 100)

    # Calculate social security contributions based on the gross salary
    social_fees_amount = gross_salary_before_tax * (social_fees / 100)

    # Total gross income required per month:
    # Business expenses + gross salary + social fees
    total_required_gross_income = (
        business_expenses +
        gross_salary_before_tax +
        social_fees_amount
    )

    # Calculate the necessary hourly rate
    hourly_rate = total_required_gross_income / monthly_hours

    return hourly_rate

# Example usage:
social_fees = 28.8       # Social security contributions in percentage
communal_tax = 33        # Municipal tax percentage
business_expenses = 3000 # Fixed business expenses in currency units
savings = 8000           # Savings goal in currency units
private_expenses = 12000 # Personal expenses in currency units

# Calculate the required hourly rate
result = calculate_hourly_rate(
    business_expenses,
    private_expenses,
    savings,
    communal_tax,
    social_fees
)

print(f"To cover your expenses, you need to charge at least {result:.2f} currency units per hour.")
