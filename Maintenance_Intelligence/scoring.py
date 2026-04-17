def risk_to_score(risk_level):
    if risk_level == "High risk":
        return 3
    elif risk_level == "Medium risk":
        return 2
    else:
        return 1


def total_risk_score(temp_risk, pressure_risk):
    return risk_to_score(temp_risk) + risk_to_score(pressure_risk)