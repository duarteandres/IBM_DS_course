# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:52:25 2020

@author: zaratejo
"""

"""
Created on Wed Jul  8 19:17:46 2020

@author: Juanma Alvarez
"""

import numpy as np
import scipy.stats as si
from scipy.stats import multivariate_normal as mvn

#%% Parametros

S = 100 #spot price
K = 100  #strike price
T =0.25#.5 #time to maturity
r = 0.08 #interest rate
sigma = 0.20 #volatility of underlying asset
b = -.04  #b<r


#%%

beta = (0.5-(b/sigma**2)) + np.sqrt( ((b/sigma**2)-0.5)**2+ 2*r/sigma**2)

B0=max(K,(r/(r-b))*K)
B8=(beta/(beta-1))*K
h_T= -(b*T+2*sigma*np.sqrt(T))*((K**2)/((B8-B0)*B0))
#h_T= -(b*T+2*sigma*np.sqrt(T))*(B0/(B8-B0))
X=B0+(B8-B0)*(1-np.exp(h_T))
#X=XT*np.exp(-r*T)
alfa_X=(X-K)*(X**(-beta))

def expec(S,T,sigma,b,Y,H,X):
    lambdas= -r + Y*b+0.5*Y*(Y-1)*(sigma)**2
    k = ((2*b)/(sigma**2)) + (2*Y-1)
    

    d1= - (np.log(S / H) + ( b + (Y- 0.5 )* sigma**2) * T) / (sigma * np.sqrt(T))
    Nd1 =si.norm.cdf(d1, 0.0, 1.0)


    d2= - (np.log(X**2 /(S* H)) + ( b + (Y- 0.5 )* sigma**2) * T) / (sigma * np.sqrt(T))
    Nd2 =si.norm.cdf(d2, 0.0, 1.0)

    
    
    expec=(np.exp(lambdas*T)*(S**Y))*(Nd1 - ((X/S)**k)*Nd2)
    
    return expec
   

Call =alfa_X*(S**beta) -alfa_X*expec(S,T,sigma,b,beta,X,X)+expec(S,T,sigma,b,1,X,X) \
        -expec(S,T,sigma,b,1,K,X)-K*expec(S,T,sigma,b,0,X,X) +K*expec(S,T,sigma,b,0,K,X)

#print(alfa_X*(S**beta),-alfa_X*expec(S,T,sigma,b,beta,X,X),expec(S,T,sigma,b,1,X,X),
 #     -expec(S,T,sigma,b,1,K,X),-K*expec(S,T,sigma,b,0,X,X),K*expec(S,T,sigma,b,0,K,X))
#%% 
        

t=0.5*(np.sqrt(5)-1)*T

h_t= -(b*(T-t)+2*sigma*np.sqrt(T-t))*((K**2)/((B8-B0)*B0))
#h_t= -(b*(T-t)+2*sigma*np.sqrt(T-t))*(B0/(B8-B0))

x=B0+(B8-B0)*(1-np.exp(h_t)) 
alfa_x=(x-K)*(x**(-beta))       
def expec2(S,T,sigma,b,Y,H,X,x,t):        

       
    d1= - (np.log(S / x) + ( b + (Y- 0.5 )* sigma**2) * t) / (sigma * np.sqrt(t))
    d2= - (np.log(X**2 / (S*x)) + ( b + (Y- 0.5 )* sigma**2) * t) / (sigma * np.sqrt(t))
    d3= - (np.log(S / x) - ( b + (Y- 0.5 )* sigma**2) * t) / (sigma * np.sqrt(t))
    d4= - (np.log(X**2 / (S*x)) - ( b + (Y- 0.5 )* sigma**2) * t) / (sigma * np.sqrt(t))
    D1 =  - (np.log(S / H) + ( b + (Y- 0.5 )* sigma**2) * T) / (sigma * np.sqrt(T))
    D2 = - (np.log(X**2 /(S* H)) + ( b + (Y- 0.5 )* sigma**2) * T) / (sigma * np.sqrt(T))
    D3= - (np.log(x**2 /(S* H)) + ( b + (Y- 0.5 )* sigma**2) * T) / (sigma * np.sqrt(T))
    D4= - (np.log(S*x**2 /(H*(X**2))) + ( b + (Y- 0.5 )* sigma**2) * T) / (sigma * np.sqrt(T))
    
    Corr= np.sqrt(t/T)    
    dist = mvn(mean=np.array([0,0]), cov=np.array([[1, Corr],[Corr, 1]]))
    dist2 = mvn(mean=np.array([0,0]), cov=np.array([[1, -Corr],[-Corr, 1]]))
    
    lambdas= -r + Y*b+0.5*Y*(Y-1)*sigma**2
    k= 2*b/sigma**2 + (2*Y-1)
    
    expec2 = np.exp(lambdas*T)*(S**Y)* \
               ((dist.cdf(np.array([d1,D1])) -((X/S)**k)*dist.cdf(np.array([d2,D2])) - \
                ((x/S)**k)*dist2.cdf(np.array([d3,D3])) +  ((x/X)**k)*dist2.cdf(np.array([d4,D4]))))#dist.cdf(np.array([D1,E1]))

    return expec2

Call_c= alfa_X*(S**beta)-alfa_X*expec(S,t,sigma,b,beta,X,X) +expec(S,t,sigma,b,1,X,X)-expec(S,t,sigma,b,1,x,X) \
       -K*expec(S,t,sigma,b,0,X,X) +K*expec(S,t,sigma,b,0,x,X) \
        + alfa_x*expec(S,t,sigma,b,beta,x,X) -alfa_x*expec2(S,T,sigma,b,beta,x,X,x,t)\
        +expec2(S,T,sigma,b,1,x,X,x,t)-expec2(S,T,sigma,b,1,K,X,x,t)    \
        -K*expec2(S,T,sigma,b,0,x,X,x,t)  +K*expec2(S,T,sigma,b,0,K,X,x,t)


print(Call)


print(Call_c)

print(2*Call_c-Call)