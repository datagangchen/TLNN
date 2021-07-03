#
# $Id: modular.py 740 2021-06-20 13:54:15Z hat $
#
from statistics import mean
import math
import random

WIGHT_THR  =0.25
def signfunction(x,ispositive=True):
    if ispositive:
        return max(0,x)
    else:
        return -signfunction(-x,True)

def quant(x, length):
    result = round(x*length)
    if result>length:
        return length
    else:
        return result 

def tanh(x):
    
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

def diff_quant(x, length, kappa):
    tan = 1-tanh(x-(round(x)+0.5)/length)**2

    return tan/(2*length*tanh(0.5*kappa/length))

def sigmoid(x):

    return 1/(1 + math.exp(-x))

def diff_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


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
        new_weight =[]

        for weig in weight:
            if weig <WIGHT_THR:
                new_weight.append(0)
            else:
                new_weight.append(weig)
        
        self.w = new_weight
    def robustness(self, weight, rho):
 
        self.rho = rho 

        wrho = [w*r for w, r in zip(weight,rho)]

        result = 1.0
        if min(wrho[self.tau1:self.tau2+1])>0:
            for i in range(self.tau1,self.tau2+1):
                result = result*(1+wrho[i])

            result = result**(1.0/(self.tau2-self.tau1+1)) -1.0

        else:
            sigrho = [signfunction(rho,False) for rho in wrho]

            result = 0.0
           
            
            for i in range(self.tau1, self.tau2+1):
                
                result = result+sigrho[i]

            result = result/(self.tau2-self.tau1+1)

        return result


    def set_interval(self,tau1,tau2):
        self.tau1 = tau1 
        self.tau2 = tau2 

    def gradient_w(self, j):
        if j<self.tau1  or j> self.tau2:
            return 0.0

  

        wrho = [w*r for w, r in zip(self.w, self.rho)]

        result = 1.0
        M = self.tau2-self.tau1+1

        if min(wrho[self.tau1:self.tau2+1])>0:
            for i in range(self.tau1,self.tau2+1):
                result = result*(1+wrho[i])

            temp = result/(1+wrho[j])

            result = result**(1.0/M-1) 
            result = 1.0/M*result*temp*self.rho[j]

        else:


            result = 1.0/M*signfunction(self.rho[j],False)

        return result


    def gradient_r(self, j):
 
        if j<self.tau1  or j> self.tau2:
            return 0.0

        wrho = [w*r for w, r in zip(self.w, self.rho)]

        result = 1.0
        M = self.tau2-self.tau1+1

        if min(wrho[self.tau1:self.tau2+1])>0:
            for i in range(self.tau1,self.tau2+1):
                result = result*(1+wrho[i])

            temp = result/(1+wrho[j])

            result = result**(1.0/M-1) 
            result = 1.0/M*result*temp*self.w[j]

        else:


            result = 1.0/M*signfunction(self.w[j],False)

        return result

class Eventually():
    def __init__(self, weight,  tau1,tau2):
        self.tau1 = tau1
        self.tau2 = tau2
        self.type = 'eventually'        
        self.w =weight 
        self.rho = None 
    def set_weight(self, weight):
        new_weight =[]

        for weig in weight:
            if weig <WIGHT_THR:
                new_weight.append(0)
            else:
                new_weight.append(weig)
        
        self.w = new_weight

    def set_rho(self,rho):
        self.rho  = rho 

    def robustness(self, weight, rho):
 
        self.rho = rho 
        wrho = [w*r for w, r in zip(weight,rho)]
         

        result = 1.0
        

        if max(wrho[self.tau1:self.tau2+1])<=0:
            for i in range(self.tau1,self.tau2+1):
                result = result*(1-wrho[i])

            result = -result**(1.0/(self.tau2-self.tau1+1)) +1.0

        else:
            sigrho = [signfunction(srho,True) for srho in wrho]

            result = 0.0
            for i in range(self.tau1, self.tau2+1):
                result = result+sigrho[i]

            result = result/(self.tau2-self.tau1+1)

        return result


    def set_interval(self,tau1,tau2):
        self.tau1 = tau1 
        self.tau2 = tau2 

    def gradient_w(self, j):
        if j<self.tau1  or j> self.tau2:
            return 0.0
        
        wrho = [w*r for w, r in zip(self.w, self.rho)]

        result = 1.0
        M = self.tau2-self.tau1+1

        if max(wrho[self.tau1:self.tau2+1])<=0:
            for i in range(self.tau1,self.tau2+1):
                result = result*(1-wrho[i])

            temp = result/(1-wrho[j])

            result = result**(1.0/M-1)
            result = (1.0/M)*temp*result*self.rho[j]

        else:


            result = (1.0/M)*signfunction(self.rho[j], True)


        return result        
        


    def gradient_r(self, j):

        if j<self.tau1  or j> self.tau2:
            return 0.0
        
        wrho = [w*r for w, r in zip(self.w, self.rho)]

        result = 1.0
        M = self.tau2-self.tau1+1

        if max(wrho[self.tau1:self.tau2+1])<=0:
            for i in range(self.tau1,self.tau2+1):
                result = result*(1-wrho[i])

            temp = result/(1-wrho[j])

            result = result**(1.0/M-1)
            
            result = (1.0/M)*temp*result*self.w[j]

        else:


            result = (1.0/M)*signfunction(self.w[j], True)

        if self.w[j]<=0:
            result = (1.0/M)*temp*result*0.01


        return result        

class AND():
    def __init__(self, w, rho=[]):
        self.rho = rho
        self.w = w
        self.type = 'and'



    def set_weight(self, weight):
        new_weight =[]

        for weig in weight:
            if weig <WIGHT_THR:
                new_weight.append(0.01)
            else:
                new_weight.append(weig)
        
        self.w = new_weight
        
    def set_rho(self,rho):
        self.rho = rho 

    def output(self, weight, rho):
        self.rho = rho 
        new_weight =[]

        for weig in weight:
            if weig <0.2:
                new_weight.append(0.0)
            else:
                new_weight.append(weig)

        alw = Always(new_weight, 0,len(rho)-1)
        return alw.robustness(weight,self.rho)

    def gradient_w(self, j):
        wrho = [w*r for w, r in zip(self.w, self.rho)]
        M = len(self.rho)
        if min(self.rho)>0:
            
            result =1.0
            for i in range(M):
                result = result*(1+wrho[i])

            temp = result/(1+wrho[j])

            result = result**(1.0/M-1)
            result = self.rho[j]*result*temp/M

            return result 
        else:
            return signfunction(self.rho[j]/M, False)





    def gradient_r(self, j):
        wrho = [w*r for w, r in zip(self.w, self.rho)]
        M = len(self.rho)
        if min(self.rho)>0:
            
            result =1
            for i in range(M):
                result = result*(1+wrho[i])

            temp = result/(1+wrho[j])

            result = result**(1.0/M-1)
            result = self.w[j]*result*temp/M

 
        else:
            result =  signfunction(self.w[j]/M, False)


        if self.w[j]<=0:
            result = (1.0/M)*temp*result*0.01

        return result

class OR():

    def __init__(self,  w, rho=[]):
        self.rho = rho
        self.w = w
        self.type = 'or'


    def output(self, weight, rho):
        self.rho = rho 
        new_weight =[]

        for weig in weight:
            if weig <0.2:
                new_weight.append(0.0)
            else:
                new_weight.append(weig)


        evn = Eventually(new_weight, 0,len(self.rho)-1)
        return evn.robustness(weight,self.rho)




    def set_weight(self, weight):
        new_weight =[]

        for weig in weight:
            if weig <0.2:
                new_weight.append(0.01)
            else:
                new_weight.append(weig)
        
        self.w = new_weight

    def set_rho(self,rho):
        self.rho = rho 

    def gradient_w(self, j):
        wrho = [w*r for w, r in zip(self.w, self.rho)]
        M = len(self.rho)
        if max(self.rho)<=0:
            
            result =1.0
            for i in range(M):
                result = result*(1-wrho[i])

            temp = result/(1-wrho[j])

            result = result**(1/M-1)
            result = self.rho[j]*result*temp/M

            return result 
        else:
            return signfunction(self.rho[j]/M, True)


    def gradient_r(self, j):
        wrho = [w*r for w, r in zip(self.w, self.rho)]
        M = len(self.rho)
        if max(self.rho)<=0:
            
            result =1.0
            for i in range(M):
                result = result*(1-wrho[i])

            temp = result/(1-wrho[j])

            result = result**(1/M-1)
            result = self.w[j]*result*temp/M

            return result 
        else:
            return signfunction(self.w[j]/M, True)

class EventualAlways():

    def __init__(self, weight, tau0, tau1,tau2):
        self.tau0 = tau0
        self.tau1 = tau1
        self.tau2 = tau2

        self.type = 'eventualalways'

        self.w =weight 
        self.rho = None 


    def robustness(self,weight,rho):

        self.rho = rho 
        erho =[]
        for i in range(0,self.tau0+1):
            new_tau1 = min(self.tau1+i,len(weight)-1)
            new_tau2 = min(self.tau2+i, len(weight)-1)
            alw = Always(weight, new_tau1, new_tau2)
            weights = [1]*len(rho)
            rob = alw.robustness(self.w,rho)
            erho.append(rob)

        Or = OR( weight, erho)

        return Or.output(weight, erho)

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

    def gradient_w(self, j):
        result =0.0
        rhos =[]
 
        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,len(self.w)-1)
            new_tau2 = min(self.tau2+k, len(self.w)-1)           
            alw = Always(self.w,new_tau1, new_tau2)
            alw.set_rho(self.rho)
            rhos.append(alw.robustness(self.w,self.rho))

        weights = [1]*len(rhos)

        Or = OR(rhos,weights)

        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,len(self.w)-1)
            new_tau2 = min(self.tau2+k, len(self.w)-1)                
            alw = Always(self.w, new_tau1, new_tau2)
            alw.set_rho(self.rho)
            result = result + Or.gradient_w(k)*alw.gradient_w(j)


        return result





    def gradient_r(self, j):
        result =0.0
        rhos =[]
 
        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,len(self.w)-1)
            new_tau2 = min(self.tau2+k, len(self.w)-1)                
            alw = Always(self.w, new_tau1,new_tau2)
            alw.set_rho(self.rho)
            rhos.append(alw.robustness(self.w,self.rho))

        weights = [1]*len(rhos)
        Or = OR(rhos,weights)

        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,len(self.w)-1)
            new_tau2 = min(self.tau2+k, len(self.w)-1)    
            alw = Always(self.w, new_tau1,new_tau2)
            alw.set_rho(self.rho)

            result = result + Or.gradient_r(k)*alw.gradient_r(j)


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
        self.rho = rho 
        erho =[]
        for i in range(0,self.tau0+1):
            new_tau1 = min(self.tau1+i,len(self.w)-1)
            new_tau2 = min(self.tau2+i, len(self.w)-1)
            even = Eventually(weight, new_tau1,new_tau2)
            even.set_rho(self.rho)
            weights = [1]*len(rho)
            rob = even.robustness(weight,rho)
            erho.append(rob)

        And = AND(weight, erho)

        return And.output(weight, erho)



    def set_interval(self,tau1,tau2):
        self.tau1 = tau1 
        self.tau2 = tau2 

    
    def set_shift(self,tau0):
        self.tau0 = tau0

    def gradient_w(self, j):
        result =0.0
        rhos =[]
 
        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,len(self.w)-1)
            new_tau2 = min(self.tau2+k, len(self.w)-1)    
            even = Eventually(self.w,new_tau1,new_tau2)
            even.set_rho(self.rho)
            rhos.append(even.robustness(self.w, self.rho))

        weights = [1.0]*len(rhos)

        And = AND(rhos,weights)

        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,len(self.w)-1)
            new_tau2 = min(self.tau2+k, len(self.w)-1)                
            even = Eventually(self.w,new_tau1,new_tau2)
            even.set_rho(self.rho)
            result = result + And.gradient_w(k)*even.gradient_w(j)

        return result
    def gradient_r(self, j):
        result =0.0
        rhos =[]
 
        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,len(self.w)-1)
            new_tau2 = min(self.tau2+k, len(self.w)-1)    
            even = Eventually(self.w,new_tau1,new_tau2)
            even.set_rho(self.rho)
            rhos.append(even.robustness(self.w,self.rho))

        weights = [1.0]*len(rhos)

        And = AND(rhos,weights)

        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,len(self.w)-1)
            new_tau2 = min(self.tau2+k, len(self.w)-1)                
            even = Eventually(self.w,new_tau1,new_tau2)
            even.set_rho(self.rho)

            result = result + And.gradient_r(k)*even.gradient_r(j)

        return result

class Predicate():
    def __init__(self, islarge, constant):
        self.islarge = islarge
        self.constant = constant
        self.type = 'predicate'

    def set_weight(self,constant):
        self.constant = constant

    def output(self,rho):

        if self.islarge:
            return [r-self.constant for r in rho]  
        else:
            return [self.constant-r for r in rho]


    def gradient_w(self, j):
        if self.islarge:
            return -1.0
        else:
            return 1.0

    def gradient_r(self, j):

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
            z = tau1*w[0]+tau2*w[1]
            h_out.append(sigmoid(z))
        
        self.h_out = h_out 

        for w in self.W_o:
            z =  [h*we for h,we in zip(h_out,w)]
            weight.append(sigmoid(sum(z)))

        return  weight


    def gradient_o(self, i, j, h_out):
        # i the ith neuron in output layer and jth neuron in hidden layer
        self.h_out = h_out
        z = [h*we for h,we in zip(self.h_out,self.W_o[i])]

        return self.h_out[j]*diff_sigmoid(sum(z))

    def gradient_h(self,i, j, k, tau, h_out):
        #ith neuron in input, jth neuron in hidden, kth neuron in output
 
        w = self.W_o[k]
        self.h_out = h_out
        zo =  [h*we for h,we in zip(self.h_out,w)]

        zh = [h*we for h,we in zip(tau,self.W_h[j])]

        return diff_sigmoid(sum(zo))*diff_sigmoid(sum(zh))*tau[i]*w[j]

    def gradient_i(self,i,j, k, tau, h_out):
        #ith neuron in input, jth neuron in hidden, kth neuron in output
        self.h_out = h_out
        w = self.W_o[k]
        zo =  [h*we for h,we in zip(self.h_out,w)]

        zh = [h*we for h,we in zip(tau,self.W_h[j])]



        return diff_sigmoid(sum(zo))*diff_sigmoid(sum(zh))*self.W_h[j][i]*w[j]
        
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
            result =[x*y for x, y in zip(rho,w)]
            h_out.append(sigmoid(sum(result)))

        self.h_out = h_out

        out = []

        for w in self.W_o:
            result = [x*y for x, y in zip(w,self.h_out)]
            out.append(sigmoid(sum(result)))

        self.out = out 


        return [min(quant(out[0],self.qunti_length),self.qunti_length-1), min(quant(out[0],self.qunti_length)+quant(out[1],self.qunti_length),self.qunti_length)-1]

    def gradient_o(self, i, j, h_out):
        # ith output neuron, jth hidden neuron
        self.h_out = h_out
      
        diff_q = diff_quant(self.out[i],self.qunti_length,self.kappa)

        z =[x*y for x, y in zip(self.h_out,self.W_o[i])]

        diff_h = diff_sigmoid(sum(z))

        return self.h_out[j]*diff_q*diff_h



    def gradient_h(self, i, j, k, inputs):

        #ith input neuron, jth hidden neuron, kth output neuron 
        if k==0:
            diff_q = diff_quant(self.out[k],self.qunti_length,self.kappa)

            zo =[x*y for x, y in zip(self.h_out,self.W_o[k])]

            diff_h = diff_sigmoid(sum(zo))*self.W_o[k][j]   

            zh =[x*y for x, y in zip(inputs,self.W_h[j])]

            diff_w = diff_sigmoid(sum(zh))*inputs[i]

            return diff_w*diff_h*diff_q
        else:
            
            diff_q1 = diff_quant(self.out[0],self.qunti_length,self.kappa)
            diff_q2 = diff_quant(self.out[1],self.qunti_length,self.kappa)

            zo1 =[x*y for x, y in zip(self.h_out,self.W_o[0])]
            zo2 =[x*y for x, y in zip(self.h_out,self.W_o[1])]

            diff_h1 = diff_q1*diff_sigmoid(sum(zo1))*self.W_o[0][j]   
            diff_h2 = diff_q2*diff_sigmoid(sum(zo1))*self.W_o[1][j]  

            zh =[x*y for x, y in zip(inputs,self.W_h[j])]

            diff_w = diff_sigmoid(sum(zh))*inputs[i]

            return diff_w*diff_h1*diff_h2

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

class TLNN():
    def __init__(self, predicates, atomics, ANDOR, And):
        self.predicates = predicates
        self.atomics= atomics 
        self.ANDOR =ANDOR
        self.And = And 

    def output(self,x):
        rho =[]
        for i in range(len(self.predicates)):
            pre = self.predicates[i]
            atom = self.atomics[i]
            opt = atom.operator
            encoder = atom.encoder
            decoder = atom.decoder
 
            rhop = pre.output(x)
            tau = encoder.output(rho)
            weight = decoder.output(tau[0],tau[1])
            opt.set_weight(weight)
            opt.set_interval(int(tau[0]),int(tau[1]))
      
            rhoa = opt.robustness(opt.w, rhop)
            rho.append(rhoa)

        andorrho=[]
        for i in range(len(self.ANDOR)):
            out = self.ANDOR[i].output(self.ANDOR[i].w, rho)
            andorrho.append(out)
   

        return self.And.output(self.And.w, andorrho)


    def add_atomic(self,predicate, atomic):
        self.predicate.append(predicate)
        self.atomics.append(atomic)

        weight = random.uniform(0, 1)

        for operator in  self.ANDOR:
            operator.w.append(weight)
             

    def gradient_descent(self, eta, x, yd):

        ploss = self.output(x)-yd 
    
        topand = self.And
        weight = topand.w 

        gradient = []
        for i in range(len(weight)):
            gradient.append(ploss*topand.gradient_w(i))

        new_topand_w = [max(0.01,xr-eta*yr) for xr, yr in zip(weight, gradient)]  
        #topand.set_weight(update_w)
        print new_topand_w

        and2 = self.ANDOR[0]
        or2 = self.ANDOR[1]
        weight2and = and2.w
        weight2or = or2.w 
        gradient_and =[]
        gradient_or  = []

        grad_atom=[]
        for i in range(len(weight2and)):
            grad_and = ploss*topand.gradient_r(0)*and2.gradient_w(i)
            gradient_and.append(grad_and)
            grad_or = ploss*topand.gradient_r(1)*or2.gradient_w(i)
            gradient_or.append(grad_or)
            grad_atom.append(grad_and+grad_or)

    


        new_and_w = [max(0.11,xr-eta*yr) for xr, yr in zip(weight2and,gradient_and)] 
        new_or_w  =[max(0.11,xr-eta*yr) for xr,yr in zip(weight2or,gradient_or)]  

        weight_atomic =[]
        weigth_atomic_j=[]


 
        predicates = self.predicates

        atomics = self.atomics



        new_p =[]
        for ia in range(len(atomics)):
            gradient_p=[]
            predicate = predicates[ia]
            atom = atomics[ia]
            encoder = atom.encoder
            decoder = atom.decoder
            opt = atom.operator
            grad_r_atom =[]

            for i in range(len(opt.w)):
                grad_r_atom.append(opt.gradient_w(i)*grad_atom[ia])




            rhop = predicate.output(x)
            tau = encoder.output(rhop)
            e_h_out = encoder.h_out
            wig = decoder.output(tau[0],tau[1])
            d_h_out = decoder.h_out


            for j in range(len(opt.w)):
                gradient_p.append(opt.gradient_r(j)*predicate.gradient_w(j)*grad_atom[ia])
            new_p.append(predicate.constant-eta*mean(gradient_p))

            print 'changed predicate', eta*mean(gradient_p)

            D_W_o = decoder.W_o
            D_W_h = decoder.W_h
            new_D_W_o =[]
            new_D_W_h =[]

            for i in range(len(D_W_o)):
                temp =[]
                for j in range(len(D_W_o[i])):
                    grad = grad_r_atom[i]*decoder.gradient_o(i,j,decoder.h_out)
                    temp.append(D_W_o[i][j]-eta*grad)
                new_D_W_o.append(temp)


            for j in range(len(D_W_h)):
                temp =[]
                for i in range(len(D_W_h[j])):
                    grad=[]
                    for k in range(len(D_W_o)):
                        tau = [opt.tau1, opt.tau2]
                        grado = grad_r_atom[k]*decoder.gradient_h(i,j,k,tau,decoder.h_out)
                        grad.append(grado)
                    temp.append(D_W_h[j][i]-eta*sum(grad))
                new_D_W_h.append(temp)

            
            E_W_o = encoder.W_o
            E_W_h = encoder.W_h
            new_E_W_o =[]
            new_E_W_h =[]

            de_in =[]
            for j in range(len(D_W_h[0])):
                temp =[]
                for m in range(len(D_W_h)):
                    gradd =[]
                    for k in range(len(D_W_o)):
                        tau = [opt.tau1, opt.tau2]
                        grads = grad_r_atom[k]*decoder.gradient_i(j,m,k,tau, decoder.h_out)
                        gradd.append(grads)
                    temp.append(sum(gradd))
                de_in.append(sum(temp))
                    
            for i in range(len(E_W_o)):
                temp =[]
                for j in range(len(E_W_o[i])):
                    grads = de_in[i]*encoder.gradient_o(i,j,encoder.h_out)
                    temp.append(E_W_o[i][j]-eta*grads)
                new_E_W_o.append(temp)

            for j in range(len(E_W_h)):
                temp =[]
                for i in range(len(E_W_h[j])):
                    grad =[]
                    
                    for k in range(len(de_in)):
                        grads = de_in[k]*encoder.gradient_h(i,j,k,x)
                        grad.append(grads)
                    temp.append(E_W_h[j][i]-eta*sum(grad))
                new_E_W_h.append(temp)

        return new_topand_w, new_p, new_and_w, new_or_w, new_D_W_o, new_D_W_h, new_E_W_o, new_E_W_h


    def update_NN(self,x,new_topand_w, new_p, new_and_w, new_or_w, new_D_W_o, new_D_W_h, new_E_W_o, new_E_W_h):

        predicates = self.predicates
        for pre, w in zip(predicates,new_p):
            pre.set_weight(w)
            print 'predicate weight', w

        And = self.And
        And.set_weight(new_topand_w)

        andor = self.ANDOR
        for i in range(len(andor)):
            if i==0:
                andor[i].set_weight(new_and_w)
                #andor[i].reset_weight(0.1)
            else:
                andor[i].set_weight(new_or_w)
                #andor[i].reset_weight(0.1)

        atomics = self.atomics 
        for atom, pre in zip(atomics,predicates):
            atom.decoder.set_weight(new_D_W_o,new_D_W_h)
            atom.encoder.set_weight(new_E_W_o,new_E_W_h)
            rho = pre.output(x)
            tau = atom.encoder.output(rho)

            print  'interval ', tau
 
            atom.operator.set_interval(int(tau[0]),int(tau[1]))
            atom.operator.set_rho(rho)
            weight = atom.decoder.output(tau[0],tau[1])
            atom.operator.set_weight(weight)

        self.predicates = predicates 
        self.ANDOR = andor 
        self.AND = And 
        self.atomics = atomics


    def train(self,x,yd, eta, H):
        for i in range(H):
            out = self.output(x)
            
           
            new_topand_w, new_p, new_and_w, new_or_w, new_D_W_o, new_D_W_h, new_E_W_o, new_E_W_h = self.gradient_descent(eta, x, yd)
            self.update_NN(x,new_topand_w, new_p, new_and_w, new_or_w, new_D_W_o, new_D_W_h, new_E_W_o, new_E_W_h)
            
            #txt = '---current output---', out*yd, '---current loss---', 0.5*(out-yd)**2

            #print txt




            

            

            

                




                    

            






   








        





