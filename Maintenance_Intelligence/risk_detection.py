def detect_temp_risk(temp_list):
    trend = temp_list[-1] - temp_list[0]

    if trend >= 10:
        return "High risk"
    elif trend >= 5:
        return "Medium risk"
    else:
        return "Normal"