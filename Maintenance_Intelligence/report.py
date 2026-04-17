def print_risk_report(
	equipment_name, temp_risk, pressure_risk, total_score
	
	):
	
			print("-----RISK REPORT-----")
			print("Equipment:", equipment_name)
			print("Temperature risk", temp_risk)
			print("Pressure risk", pressure_risk)
			print("Total risk score", total_score)
	
			if total_score >= 5: 
				print("Overall status: CRITICAL")
			elif total_score >= 4:
				print("Overall status: HIGH ATTENTION")
			elif total_score >= 3:
				print("Overall status: WATCH")
			else:
				print("Overall status: STABLE")