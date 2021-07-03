#
# $Id: modular.py 740 2021-06-20 13:54:15Z hat $
#
from statistics import mean
import math
import random
import numpy as np
LENGTH_SIGNAL = 128
WIGHT_THR  =0.25

def signfunction(x,ispositive=True):
    if ispositive:
        return max(0,x)
    else:
        return -signfunction(-x,True)
        
vsignfunction = np.vectorize(signfunction)

def signfunction_w(x,rho, ispositive=True):
    if ispositive:
        result = max(0.0, x*rho)/(rho+0.000000000001)
        return result 
    else:
        return -signfunction_w(x, -rho, True)

vsignfunction_w = np.vectorize(signfunction_w)
def Gaussian(x):
    return math.exp(-x**2)

vGaussian = np.vectorize(Gaussian)

def diff_Gaussian(x):
    return -2*x*math.exp(-x**2)
vdiff_Gaussian = np.vectorize(diff_Gaussian)
def quant(x, length):
    #print 'output of encoder', x 
    result = round(x*length)
    if result>length:
        return length
    else:
        return result 
vquant = np.vectorize(quant)
def tanh(x):
    
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

def diff_quant(x, length, kappa):
    tan = 1-tanh(x-(round(x)+0.5)/length)**2

    return tan/(2*length*tanh(0.5*kappa/length))

vdiff_quant = np.vectorize(diff_quant)

def sigmoid(x):

    return 1/(1 + math.exp(-x))

vsigmoid = np.vectorize(sigmoid)
def sigmoid_shift(x):

    return 1/(1 + math.exp(-x))-0.5

def diff_sigmoid(x):

    return sigmoid(x)*(1-sigmoid(x))

vdiff_sigmoid = np.vectorize(diff_sigmoid)

class Always():
    def __init__(self, weight,  tau1,tau2):
 
        self.tau1 = tau1
        self.tau2 = tau2
        self.type ='always'
        self.w =weight 
        self.rho = None 

    def set_rho(self,rho):
        self.rho  = rho         
    def set_weight(self, weight):

        self.w = weight

    def robustness(self, weight, rho):
        wrho = np.multiply(weight, rho)

        ewrho = wrho[self.tau1:self.tau2+1]
        if min(ewrho)>=0:
            result = wrho[self.tau1:self.tau2+1]+1
            result = np.product(result)
            result = result**(1.0/(self.tau2-self.tau1+1)) -1.0
        else:
            sigrho = vsignfunction(ewrho, False)
            result = sum(sigrho)
            result = result/(self.tau2-self.tau1+1)

        return result


    def set_interval(self,tau1,tau2):
        self.tau1 = tau1 
        self.tau2 = tau2 

    def gradient_w(self):
        wrho = np.multiply(self.w, self.rho)
        M = self.tau2-self.tau1+1
        ewrho = wrho[self.tau1:self.tau2+1]
        if min(ewrho)>=0:
            shift = ewrho+1
            result = np.product(shfit)
            temp = result/shift 
            result = result**(1.0/M-1) 
            result = result*np.multiply(self.rho[self.tau1:self.tau2+1],temp)/M
        else:
            result = vsignfunction_w(self.rho[self.tau1:self.tau2+1],self.w[self.tau1:self.tau2+1], False)/M


        grad = np.zeros(self.w.size)
        grad[self.tau1:self.tau2+1] = result

        return grad 


    def gradient_r(self):

        wrho = np.multiply(self.w, self.rho)
        ewrho=wrho[self.tau1:self.tau2+1]
        result = 1.0
        M = self.tau2-self.tau1+1
        if min(ewrho)>=0:
            shift = ewrho +1
            result = np.product(result)
            temp = result/shift
            result = result**(1.0/M-1) 
            result = result*np.multiply(self.w[self.tau1:self.tau2+1], temp)/M 
        else:
            result = vsignfunction_w(self.w[self.tau1:self.tau2+1], self.rho[self.tau1:self.tau2+1], False)/M

        grad = np.zeros(self.w.size)
        grad[self.tau1:self.tau2+1] = result

        return grad 

class Eventually():
    def __init__(self, weight,  tau1,tau2):
        self.tau1 = tau1
        self.tau2 = tau2
        self.type = 'eventually'        
        self.w =weight 
        self.rho = None 
    def set_weight(self, weight):
        self.w = weight

    def set_rho(self,rho):
        self.rho  = rho 

    def robustness(self, weight, rho):
 
        #self.rho = rho 
        wrho = np.multiply(weight,rho)
        ewrho = wrho[self.tau1:self.tau2+1]
        #print wrho, weight, rho 
         
        if max(ewrho)<0:
 
            result = 1-ewrho
            result = np.product(result)
            result = -result**(1.0/(self.tau2-self.tau1+1)) +1.0

        else:
            sigrho = vsignfunction(ewrho, True)
            result = sum(sigrho)
            result = result/(self.tau2-self.tau1+1)

        return result


    def set_interval(self,tau1,tau2):
        self.tau1 = tau1 
        self.tau2 = tau2 

    def gradient_w(self):

        wrho = np.multiply(self.w, self.rho)
        ewrho = wrho[self.tau1:self.tau2+1]

 
        M = self.tau2-self.tau1+1

        if max(ewrho)<0:
            shift = 1 - ewrho
            result = np.product(shift)
            temp = result/shift
            result = result**(1.0/M-1)
            result = result*np.multiply(self.rho[self.tau1:self.tau2+1],temp)/M
       

        else:
            result =  vsignfunction_w(self.rho[self.tau1:self.tau2+1], self.w[self.tau1:self.tau2+1],  True)/M




        grad = np.zeros(self.w.size)
        grad[self.tau1:self.tau2+1] = result

        return grad 


    def gradient_r(self):


        wrho = np.multiply(self.w, self.rho)
        ewrho = wrho[self.tau1:self.tau2+1]

        result = 1.0
        M = self.tau2-self.tau1+1

        if max(ewrho)<0:
 
            shift  = 1- ewrho
            result = np.product(shift)
            temp = result/shift
            result = result**(1.0/M-1)
            result = result*np.multiply(self.w[self.tau1:self.tau2+1], temp)/M


        else:
            result = vsignfunction_w(self.w[self.tau1:self.tau2+1],self.rho[self.tau1:self.tau2+1], True)/M

        grad = np.zeros(self.w.size)
        grad[self.tau1:self.tau2+1] = result

        return grad 

class AND():
    def __init__(self, w, rho=np.array([])):
        self.rho = rho
        self.w = w
        self.type = 'and'



    def set_weight(self, weight):
        self.w = weight
        
    def set_rho(self,rho):
        self.rho = rho 

    def output(self, weight, rho):
 
        new_weight =[]
        for weig in weight:
            if Gaussian(weig) <0.2:
                new_weight.append(Gaussian(1))
            else:
                new_weight.append(Gaussian(weig))

        alw = Always(np.array(new_weight), 0,rho.size-1)
        alw.set_rho(rho)

        return alw.robustness(np.array(new_weight),rho)

    def gradient_w(self):

        Gw = vGaussian(self.w)
        wrho  = np.multiply(Gw,self.rho)
        M = self.rho.size
        if min(wrho)>=0:
            shift = 1+wrho
            result = np.product(shift)
            temp = result/shift
            result = result**(1.0/M-1)
            result  = result*np.multiply(temp,self.rho)
            result = np.multiply(result, vdiff_Gaussian(self.w))/M
 

        else:

            result = np.multiply(vsignfunction_w(self.rho,self.w, False),vdiff_Gaussian(self.w))/M
        return result 

    def gradient_r(self):
        
        Gw = vGaussian(self.w)
        wrho  = np.multiply(Gw,self.rho)
        
        M = len(self.rho)
        if min(wrho)>=0:
 
            shfit = 1+wrho
            result = np.product(shfit)

            temp = result/shfit
            result = result**(1.0/M-1)

            result = result*np.multiply(vGaussian(self.w),temp)/M

 

        else:
            result =  vsignfunction_w(vGaussian(self.w), self.rho, False)/M

            #print 'not satisfied', result
        return result

class OR():

    def __init__(self,  w, rho=np.array([])):
        self.rho = rho
        self.w = w
        self.type = 'or'

    def output(self, weight, rho):
        new_weight =[]
        for weig in weight:
            if Gaussian(weig) <0.2:
                new_weight.append(Gaussian(1))
            else:
                new_weight.append(Gaussian(weig))

        evn = Eventually(np.array(new_weight), 0, rho.size-1)
        return evn.robustness(np.array(new_weight),rho)


    def set_weight(self, weight):
        self.w = weight

    def set_rho(self,rho):
        self.rho = rho 

    def gradient_w(self):

        Gw = vGaussian(self.w)
        wrho  = np.multiply(Gw,self.rho)
        M = self.rho.size
        if max(wrho)<0:
            shift = 1-wrho
            result = np.product(shift)
            temp = result/shift
            result = result**(1/M-1)
            result = result*np.multiply(self.rho,vdiff_Gaussian(self.w))
            result = np.multiply(temp,result)

            
        else:
      
            result  = np.multiply(vsignfunction_w(self.rho, self.w, True),vdiff_Gaussian(self.w))/M

        return result 

    def gradient_r(self):

        Gw = vGaussian(self.w)
        wrho  = np.multiply(Gw,self.rho)
        M = self.rho.size
        if max(wrho)<0:
            shift = 1-wrho
            result = np.product(shift)
            temp = result/shift
            result = result**(1/M-1)
            result = result*np.multiply(vGaussian(self.w),temp)/M
            
        else:
            result = vsignfunction_w(vGaussian(self.w), self.rho, True)/M

        return result   


class EventualAlways():

    def __init__(self, weight, tau0, tau1,tau2):
        self.tau0 = tau0
        self.tau1 = tau1
        self.tau2 = tau2

        self.type = 'eventualalways'

        self.w =weight 
        self.rho = None 


    def robustness(self,weight,rho):
        erho =[]
        for i in range(0,self.tau0+1):
            new_tau1 = min(self.tau1+i,len(weight)-1)
            new_tau2 = min(self.tau2+i, len(weight)-1)
            alw = Always(weight, new_tau1, new_tau2)
            rob = alw.robustness(weight,rho)
            erho.append(rob)

        weights = np.array([1]*len(erho))
        Or = OR(weights, np.array(erho))

        return Or.output(weights, np.array(erho))

    def set_interval(self, tau1, tau2):
        self.tau1 = tau1
        self.tau2 = tau2

    def set_weight(self, weight):
        self.w =weight
    def set_rho(self,rho):
        self.rho = rho 

    def set_shift(self,tau0):
        self.tau0 = tau0

    def set_interval(self,tau1,tau2):
        self.tau1 = tau1 
        self.tau2 = tau2 

    def gradient_w(self):
        result =0.0
        rhos =[]
 
        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,self.w.size-1)
            new_tau2 = min(self.tau2+k, self.w.size-1)           
            alw = Always(self.w,new_tau1, new_tau2)
            alw.set_rho(self.rho)
            rhos.append(alw.robustness(self.w,self.rho))

        weights = np.array([1]*len(rhos))

        Or = OR(weights, np.array(rhos))
        grad_or =  Or.gradient_r()
        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,self.w.size-1)
            new_tau2 = min(self.tau2+k, self.w.size-1)                
            alw = Always(self.w, new_tau1, new_tau2)
            alw.set_rho(self.rho)
            result = result +grad_or[k]*alw.gradient_w()

        return result





    def gradient_r(self):
        result =0.0
        rhos =[]
 
        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,self.w.size-1)
            new_tau2 = min(self.tau2+k, self.w.size-1)                
            alw = Always(self.w, new_tau1,new_tau2)
            alw.set_rho(self.rho)
            rhos.append(alw.robustness(self.w,self.rho))

        weights = np.array([1]*len(rhos))
        Or = OR(weights,np.array(rhos))
        grad_or =Or.gradient_r()

        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,self.w.size-1)
            new_tau2 = min(self.tau2+k, self.w.size-1)    
            alw = Always(self.w, new_tau1,new_tau2)
            alw.set_rho(self.rho)
            result = result + grad_or[k]*alw.gradient_r()

        return result

class AlwaysEventual():
    def __init__(self, weight, tau0, tau1,tau2):
        self.tau0 = tau0
        self.tau1 = tau1
        self.tau2 = tau2
        self.type ='alwayseventual'
        self.w =weight 
        self.rho = None 

    def set_weight(self, weight):
        self.w = weight

    def set_rho(self, rho):
        self.rho = rho


    def robustness(self, weight, rho):

        erho =[]
        for i in range(0,self.tau0+1):
            new_tau1 = min(self.tau1+i,self.w.size-1)
            new_tau2 = min(self.tau2+i, self.w.size-1)
            even = Eventually(weight, new_tau1,new_tau2)
            even.set_rho(self.rho)
            rob = even.robustness(weight,rho)
            erho.append(rob)

        weights =np.array([1]*len(erho))

        And = AND(weights, np.array(erho))

        return And.output(weights, np.array(erho))



    def set_interval(self,tau1,tau2):
        self.tau1 = tau1 
        self.tau2 = tau2 

    
    def set_shift(self,tau0):
        self.tau0 = tau0

    def gradient_w(self):
        result =0.0
        rhos =[]
 
        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,self.w.size-1)
            new_tau2 = min(self.tau2+k, self.w.size-1)    
            even = Eventually(self.w,new_tau1,new_tau2)
            even.set_rho(self.rho)
            rhos.append(even.robustness(self.w, self.rho))

        weights = np.array([1.0]*len(rhos))

        And = AND(weights, np.array(rhos))
        grad_and = And.gradient_r()

        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,self.w.size-1)
            new_tau2 = min(self.tau2+k, self.w.size-1)                
            even = Eventually(self.w,new_tau1,new_tau2)
            even.set_rho(self.rho)
            result = result + grad_and[k]*even.gradient_w()

        return result
    def gradient_r(self):
        result =0.0
        rhos =[]
        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,self.w.size-1)
            new_tau2 = min(self.tau2+k, self.w.size-1)    
            even = Eventually(self.w,new_tau1,new_tau2)
            even.set_rho(self.rho)
            rhos.append(even.robustness(self.w,self.rho))

        weights = np.array([1.0]*len(rhos))
        And = AND(weights,np.array(rhos))
        grad_and = And.gradient_r()

        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,self.w.size-1)
            new_tau2 = min(self.tau2+k, self.w.size-1)                
            even = Eventually(self.w,new_tau1,new_tau2)
            even.set_rho(self.rho)
            result = result +grad_and[k]*even.gradient_r()
        return result

class Predicate():
    def __init__(self, islarge, constant):
        self.islarge = islarge
        self.constant = constant
        self.type = 'predicate'
        self.mw = 1
        self.vw = 1

    def set_weight(self,constant):
        self.constant = constant
    def set_moment(self,mw,vw):
        self.mw = mw 
        self.vw  = vw 

    def output(self,rho):

        if self.islarge:
            return rho - self.constant
        else:
            return self.constant - rho  


    def gradient_w(self):
        if self.islarge:
            return -1.0
        else:
            return 1.0

    def gradient_r(self):

        if self.islarge:
            return 1.0
        else:
            return -1.0


class Decoder():
    def __init__(self, W_o, W_h):
        self.W_o = W_o
        self.W_h = W_h
        self.h_out =None 
    def set_weight(self, W_o, W_h):
        self.W_o = W_o
        self.W_h = W_h        
    def output(self,  tau1,tau2):
        weight =[]
        h_out =[]
        for w in self.W_h:
            z = (tau1*w[0]+tau2*w[1])/LENGTH_SIGNAL
            h_out.append(sigmoid(z))

        self.h_out = np.array(h_out)

        for w in self.W_o:

            z = np.multiply(self.h_out,w)

            weight.append(sigmoid(mean(z)))

        return  np.array(weight)


    def gradient_o(self, i,  h_out):
        # i the ith neuron in output layer

        z = np.multiply(self.h_out,self.W_o[i])
        result= h_out*diff_sigmoid(mean(z))
        return result

    def gradient_h(self, j,  tau, h_out, Grad_o):
        # jth neuron in hidden

        result = 0.0
        w = np.array(self.W_o)[:,j]
        grad_o = Grad_o[:,j]/h_out[j]
        result =np.multiply(w,grad_o)
        zh = np.multiply(tau,self.W_h[j])/LENGTH_SIGNAL
        result = sum(result)*diff_sigmoid(mean(zh))*tau

        return result

    def gradient_i(self,i, tau,  Grad_h):
        #ith neuron in input
        w = np.array(self.W_h)[:,i]
        grah_h = np.array(Grad_h)[:,i]
        result = np.multiply(w,grah_h)/tau[i]
        return sum(result)
        
class Encoder():
    def __init__(self, W_o, W_h, qunti_length, kappa):
        self.W_o = W_o
        self.W_h = W_h
        self.qunti_length = qunti_length
        self.kappa = kappa
        self.h_out =None 
        self.out = None 


    def set_weight(self, W_o, W_h):
        self.W_o = W_o
        self.W_h = W_h        


    def output(self, rho):
        h_out =[]
        for w in self.W_h:
            result = np.multiply(rho,w)
            h_out.append(tanh(mean(result)))
        self.h_out = np.array(h_out)
        out = []
        for w in self.W_o:
            result = np.multiply(w,self.h_out)
            out.append(sigmoid(mean(result)))
        self.out = np.array(out) 

        tau1 = min(quant(out[0],self.qunti_length),self.qunti_length-2)
        tau2 = min(tau1+quant(out[1],self.qunti_length),self.qunti_length)-1
        return np.array([tau1,tau2])


    def gradient_o(self, i,  h_out):
        # ith output neuron
        if i ==0:
            diff_q1 = diff_quant(self.out[0],self.qunti_length,self.kappa)
            diff_q2 = diff_quant(self.out[1],self.qunti_length,self.kappa)
            z = np.multiply(self.h_out,self.W_o[i])
            result= h_out*diff_sigmoid(mean(z))*(diff_q1+diff_q2)

        else:
            diff_q2 = diff_quant(self.out[1],self.qunti_length,self.kappa)
            z = np.multiply(self.h_out,self.W_o[i])
            result= h_out*diff_sigmoid(mean(z))*diff_q2
        return result 


    def gradient_h(self, j,  inputs,h_out, Grad_o):
        w = np.array(self.W_o)[:,j]
        grad_o = np.array(Grad_o[:,j])/h_out[j]
        result =np.multiply(w,grad_o)
        zh = np.multiply(inputs,self.W_h[j])/LENGTH_SIGNAL
        result = sum(result)*diff_sigmoid(mean(zh))*inputs
        return result


class atomic_formula():
    def __init__(self, encoder, decoder,  operator):
        self.encoder = encoder
        self.decoder = decoder
        self.operator = operator

    def set_init(self, encoder, decoder, operator):
        self.encoder = encoder
        self.decoder = decoder
        self.operator 

    def output(self, rho):
        tau = self.encoder.output(self.rho)
        weight  = self.decoder.output(tau[0],tau[1])

        return operator.robustness(weigth,rho)

            

                




                    

            






   








        





