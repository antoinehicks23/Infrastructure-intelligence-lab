from datetime import datetime
from calendar import monthrange


def add_months(d, months):
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    day = min(d.day, monthrange(year, month)[1])
    return d.replace(year=year, month=month, day=day)


def simulate_payoff(principal, apr_percent, payment, extra_payment=0.0, max_months=1200):
    balance = float(principal)
    apr = float(apr_percent) / 100.0
    monthly_rate = apr / 12.0

    months = 0
    total_interest = 0.0
    total_paid = 0.0

    # Guardrails
    if payment + extra_payment <= 0:
        return {
            "paid_off": False,
            "reason": "Monthly payment must be greater than 0.",
            "months": None,
            "total_interest": None,
            "total_paid": None
        }

    # If you can't cover first month interest, you'll never pay it off
    first_month_interest = balance * monthly_rate
    if payment + extra_payment <= first_month_interest:
        return {
            "paid_off": False,
            "reason": "Payment too small to cover monthly interest (loan will grow).",
            "months": None,
            "total_interest": None,
            "total_paid": None
        }

    while balance > 0 and months < max_months:
        months += 1

        # Add interest
        interest = balance * monthly_rate
        total_interest += interest
        balance += interest

        # Apply payment
        pay = payment + extra_payment

        # Final month: don't go negative
        if pay > balance:
            pay = balance

        balance -= pay
        total_paid += pay

        # Clamp for safety (should already be 0 from final month logic)
        if balance < 0:
            balance = 0

    if balance <= 0:
        return {
            "paid_off": True,
            "months": months,
            "total_interest": total_interest,
            "total_paid": total_paid
        }

    return {
        "paid_off": False,
        "reason": "Hit max_months limit (check inputs).",
        "months": months,
        "total_interest": total_interest,
        "total_paid": total_paid
    }


def run_comparison():
    principal = float(input("Loan balance (principal): "))
    apr_percent = float(input("APR % (e.g., 14.54): "))
    payment = float(input("Monthly payment: "))

    start_str = input("Start date (YYYY-MM-DD): ")
    start_date = datetime.strptime(start_str, "%Y-%m-%d").date()

    print("\nScenario A: no extra payment")
    result_a = simulate_payoff(principal, apr_percent, payment, extra_payment=0.0)

    extra = float(input("\nScenario B extra payment (enter 0 if none): "))
    print("\nScenario B: with extra payment")
    result_b = simulate_payoff(principal, apr_percent, payment, extra_payment=extra)

    print("\n================ RESULTS ================")

    if result_a["paid_off"]:
        payoff_date_a = add_months(start_date, result_a["months"])
        print("A) Months:", result_a["months"])
        print("A) Payoff date:", payoff_date_a)
        print("A) Total interest: ${:.2f}".format(result_a["total_interest"]))
        print("A) Total paid:     ${:.2f}".format(result_a["total_paid"]))
    else:
        print("A) Not paid off:", result_a.get("reason", "Unknown"))

    print("----------------------------------------")

    if result_b["paid_off"]:
        payoff_date_b = add_months(start_date, result_b["months"])
        print("B) Months:", result_b["months"])
        print("B) Payoff date:", payoff_date_b)
        print("B) Total interest: ${:.2f}".format(result_b["total_interest"]))
        print("B) Total paid:     ${:.2f}".format(result_b["total_paid"]))
    else:
        print("B) Not paid off:", result_b.get("reason", "Unknown"))

    if result_a["paid_off"] and result_b["paid_off"]:
        months_saved = result_a["months"] - result_b["months"]
        interest_saved = result_a["total_interest"] - result_b["total_interest"]
        print("\n>>> Months saved: {}".format(months_saved))
        print(">>> Interest saved: ${:.2f}".format(interest_saved))


run_comparison()
