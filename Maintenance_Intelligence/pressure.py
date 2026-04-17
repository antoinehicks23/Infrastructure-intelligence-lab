def detect_pressure_risk(pressure_list):
		trend = pressure_list[-1] - pressure_list[0]
		
		if trend <= -10: 
			return "High risk"
			
		elif trend <= -5: 
			return "Medium risk"
			
		else: 
			return "Normal"

if __name__ == "__main__":
		pressure = [100, 95, 85]
		print(detect_pressure_risk(pressure))
			
