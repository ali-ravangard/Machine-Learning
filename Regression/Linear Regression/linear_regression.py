from read_file import read_data
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = read_data("data.csv","csv") # reading the data
df = (df-df.mean())/df.std() # normalizing the data without using sklearn
n,m = df.shape 
x0 = np.ones((n,1)) 
df = np.hstack((x0,df)) # add bias to feature data 
X = df[:,[0,1]]
y = df[:,[2]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def closed_form(x_data , y_data):
    theta_mat = (np.linalg.inv(x_data.T @ x_data)) @ ( x_data.T @ y_data)

    return theta_mat

theta_train = closed_form(X_train,y_train) #theta on train data 
theta_test = closed_form(X_test,y_test) #theta on test data 
print("theta matrix for closed form is = ",theta_train)
# print(y_test)
y_pre_test = X_test @ theta_train
# print(y_pre_test)

x = np.linspace(-2,3)
# plt.scatter(X[:,[1]], y )
# plt.scatter(X_test[:,[1]], y_test , color = "blue" )
# plt.scatter(X_test[:,[1]], y_pre_test ,color = "green")
# plt.plot(x, theta_train[0]+ x* theta_train[1],'-r')
# plt.show()
#--------------------------------------------------------------------------------------
# gradient descent 

def cost (X,Y,theta):
    return np.sum((X.dot(theta)-Y)**2)/2/len(Y)

def gradient_descent(X,Y,theta,alpha,iterations,threshold):
    cost_values=[]
    for i in range(iterations) : 
        gradient = X.T @ (X.dot(theta)-Y)
        theta = theta - alpha*gradient
        cost_value = cost(X,Y,theta)
        cost_values.append(cost_value)
        if i > 1 and np.abs(cost_values[-2]-cost_values[-1]) < threshold:
            print("stop iteration is =", i)
            break
    return theta , cost_values

alpha = 0.0001
iterations = 1000
threshold = 0.0001
theta_init = np.array([-1,-1]).reshape([2,1])
(theta_train_gd , cost_values) = gradient_descent(X_train,y_train,theta_init, alpha ,iterations, threshold)
(theta_test_gd , cost_values) = gradient_descent(X_test,y_test,theta_init, alpha ,iterations, threshold)

print("theta matrix for gradient descent is = ",theta_train_gd)
# print(b)

# plt.scatter(X[:,[1]], y )

y_pre_test_gd = X_test @ theta_train_gd

# plt.scatter(X_test[:,[1]], y_test , color = "blue" )
# plt.scatter(X_test[:,[1]], y_pre_test_gd ,color = "green")
# plt.plot(x, theta_train_gd[0]+ x* theta_train_gd[1],'-r')
# plt.show()


# plt.scatter(range(len(cost_values)), cost_values , color = "blue" )
# plt.show()

# print("cost on train data is : " , cost(X_train,y_train,theta_train))
# print("cost on test data is : " , cost(X_test,y_test,theta_test))
# print("cost on train data is : " , cost(X_train,y_train,theta_train_gd))
# print("cost on test data is : " , cost(X_test,y_test,theta_test_gd))