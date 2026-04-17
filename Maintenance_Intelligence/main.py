from pressure import detect_pressure_risk
from scoring import total_risk_score


def calculate_trend(values):
		return values[-1] - values[0]
		
temps = [230, 205, 215, 220]

trend = calculate_trend(temps)
print(trend)

def detect_temp_risk(temp_list):
		trend = temp_list[-1] - temp_list[0]
		
		if trend > 10:
			return "High risk"
			
		elif trend > 5:
			return "Medium risk"
			
		else:
			return "Normal"
			
temps = [230, 205, 215, 220]
pressure = [100, 95, 85]

temp_risk = detect_temp_risk(temps)
pressure_risk = detect_pressure_risk(pressure)

print("Temp risk:", temp_risk)
print("Pressure risk:", pressure_risk)

score = total_risk_score(temp_risk, pressure_risk)
print("Total risk score", score)
