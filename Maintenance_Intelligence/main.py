from trend import calculate_trend
from risk_detection import detect_temp_risk


temps = [230, 205, 215, 220]

trend = calculate_trend(temps)
print("Trend:", trend)

result = detect_temp_risk(temps)
print("Risk Level:", result)


# Test cases
print(detect_temp_risk([200, 205, 210]))  # rising
print(detect_temp_risk([200, 200, 200]))  # flat
print(detect_temp_risk([220, 210, 200]))  # falling