import cvxpy as cvx

x = cvx.Variable()
y = cvx.Variable()

constraints = [x + y == 1,
			   x - y >= 1]

obj = cvx.Minimize((x-y)**2)

prob = cvx.Problem(obj, constraints)

prob.solve()

print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)

# Works! Alright!