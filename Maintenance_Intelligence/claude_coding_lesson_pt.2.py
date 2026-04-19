temps = [300, 275, 325, 350]

def temp_rise(temps): 
	trend = temps[-1] - temps[0]
	
	if trend >= 20:
		print("High Risk! Conduct immediate actions ⚠️")
	elif trend >= 0:
		print("Moderate risk. Conduct controlling actions")
	else: 
		print("Temps are stable ✅.")
		
temp_rise(temps)

#print(f"temperatures: {temps}")
#print(f"first temp:  {temps[0]}")
#print(f"last temp: {temps[-1]}")

