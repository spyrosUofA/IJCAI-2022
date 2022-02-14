library(e1071)

# Load data
obs <- read.table("~/Documents/GitHub/IJCAI-2022/LunarLander-v2/save_x2.txt", quote="\"")
actions <- read.table("~/Documents/GitHub/IJCAI-2022/LunarLander-v2/save_y2.txt", quote="\"")
names(obs) = c("x", "y", "v_x", "v_y", "theta", "v_theta", "c_L", "c_R")
names(actions) = "a"

# Train SVM 
dat = cbind(obs, actions)[0:10000,]
svmfit = svm(a ~ ., data = dat, kernel = "linear", cost = 10, scale = FALSE)#, cost = rep(2, 8))
print(svmfit)

# COEFF 
beta = drop(t(svmfit$coefs)%*%as.matrix(dat[svmfit$index,-9]))
beta0 = svmfit$rho

print(beta)
print(beta0)

# Plot 
plot(svmfit, dat, v_y~y)
plot(svmfit, dat, c_L~c_R)
plot(svmfit, dat, v_theta~theta)




#
install.packages('elasticnet')
library(elasticnet)
enet(x = as.matrix(obs[,c(-7,-8)]), y = actions$a, lambda = 0)


# SVM with LASSO (alpha=1)
install.packages('sparseSVM')
library(sparseSVM)

fit = sparseSVM(as.matrix(obs), y = actions$a, alpha=1) #,  lambda = .6)
fit$weights

coef(fit, 0.05)
y_hat = predict(fit, as.matrix(obs)) #, lambda = c(0.2, 0.5))
coef(fit)

lam_x = as.numeric(colnames(y_hat))
score_y = colMeans(y_hat == actions$a)
plot(lam_x, score_y)




cv.fit <- cv.sparseSVM(as.matrix(obs), y = actions$a, alpha=1, seed = 1234)
predict(cv.fit, as.matrix(obs))
coef(cv.fit)
plot(cv.fit)



p = predict(fit, as.matrix(obs)) #, lambda = c(0.51, 0.6))


### SPARSE LDA -- not working... 
#install.packages("sparseLDA")
library(sparseLDA)

xx = as.matrix(obs)
yy = as.matrix(as.numeric(actions == 0))

out <- sda(xx, yy,
           lambda = 1e-6,
           #stop = -1,
           #maxIte = 25,
           scale = FALSE,
           trace = TRUE)

## predict training samples
train <- predict(out, Xn)
## testing
Xtst<-X[Iout,]
Xtst<-normalizetest(Xtst,Xc)
8
smda
test <- predict(out, Xtst)
print(test$class)



# Perfect classifier using only C_R and C_L...
p = rep(3, 11354)
for (i in 1:11354){
  if (obs$c_L[i] + obs$c_R[i] > 0.1){
    p[i] = 0
  }
}
mean(p == actions$a)

