#install.packages("plot3D")

#####################
## 1- PREPARE DATA ##
#####################
obs <- read.table("~/Documents/GitHub/IJCAI-2022/LunarLander-v2/save_x2.txt", quote="\"")
names(obs) = c("x", "y", "v_x", "v_y", "theta", "v_theta", "c_L", "c_R")

actions <- read.table("~/Documents/GitHub/IJCAI-2022/LunarLander-v2/save_y2.txt", quote="\"") - + 1
names(actions) = "a"


data_o = cbind(obs, actions)
data = data_o[1:2000,]

#n0 = c(-0.09,  0.12, -0.12,  0.27,  0.22,  0.18,  0.22,  0.1)
#obs_w = data.frame(mapply('*', obs, n0))
#data_w = cbind(obs_w, actions)
#data = data_w[1:2000,]


plot(data$y, data$v_y, col= ifelse(data$a == 2, "red", "black"))
abline(b=-0.5, a=-0.02)
abline(b=-0.5, a=-0.00)
abline(v=0.01)



# PLOTS 
# X vs ~
plot(data$x, data$v_x, col= ifelse(data$a == 2, "red", "black")) 
plot(data$x, data$y, col= ifelse(data$a == 2, "red", "black")) 
plot(data$x, data$v_y, col= ifelse(data$a == 2, "red", "black"))
plot(data$x, data$theta, col= ifelse(data$a == 2, "red", "black")) 
plot(data$x, data$v_theta, col= ifelse(data$a == 2, "red", "black")) 
# V_X vs ~
plot(data$v_x, data$y, col= ifelse(data$a == 2, "red", "black")) 
plot(data$v_x, data$v_y, col= ifelse(data$a == 2, "red", "black")) 
plot(data$v_x, data$theta, col= ifelse(data$a == 2, "red", "black")) 
plot(data$v_x, data$v_theta, col= ifelse(data$a == 2, "red", "black")) 
# Y vs ~.
plot(data$y, data$v_y, col= ifelse(data$a == 2, "red", "black"))
plot(data$y, data$theta, col= ifelse(data$a == 2, "red", "black"))
plot(data$y, data$v_theta, col= ifelse(data$a == 2, "red", "black"))
# V_Y vs ~.
plot(data$v_y, data$theta, col= ifelse(data$a == 2, "red", "black"))
plot(data$v_y, data$v_theta, col= ifelse(data$a == 2, "red", "black"))
# THETA 
plot(data$theta, data$v_theta, col= ifelse(data$a == 2, "red", "black"))
# CONTACT
plot(data$c_L, data$c_R, col= ifelse(data$a == 2, "red", "black"))

# individual
plot(data$x, col= ifelse(data$a == 2, "red", "black"))
plot(data$y, col= ifelse(data$a == 2, "red", "black"))
plot(data$v_x, col= ifelse(data$a == 2, "red", "black"))
plot(data$v_y, col= ifelse(data$a == 2, "red", "black"))
plot(data$theta, col= ifelse(data$a == 2, "red", "black"))
plot(data$v_theta, col= ifelse(data$a == 2, "red", "black"))
plot(data$c_L, col= ifelse(data$a == 2, "red", "black"))
plot(data$c_R, col= ifelse(data$a == 2, "red", "black"))
plot(data$c_L + data$c_R, col= ifelse(data$a == 2, "red", "black"))




# SVM
library(e1071)
dat = data.frame(data[,-9], a = as.factor(data$a))
svmfit = svm(a ~ ., data = dat, kernel = "linear", cost = 1000, scale = FALSE)#, cost = rep(2, 8))
print(svmfit)

# COEFF 
beta = drop(t(svmfit$coefs)%*%as.matrix(dat[svmfit$index,-9]))
beta0 = svmfit$rho

print(beta)
print(beta0)

# Plot 
plot(svmfit, dat, v_y~y)
plot(svmfit, dat, c_L~c_R)
