def climbStair(n, stor={1:1,2:2}):
	'''
	I: something
	O: another thing
	'''
	
	#base
	if n in stor: 
		return n
	else:
		if n - 1 not in stor:
			stor[n-1] = climbStair(n-1)
		if n - 2 not in stor:
			stor[n-2] = climbStair(n-2)
	
	stor[n] = stor[n-1] + stor[n-2]
	return stor[n]

#fib

def fib(n, base={0:1, 1:2}):
	return base[n-1] + base[n-2]
#climbing stars
#coin change 
#find out what's the coin change problem


#