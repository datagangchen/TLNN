

from statistics import mean
import math
import random
import numpy as np
from network import *


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

            opt.set_rho(rhop)
            tau = encoder.output(rhop)
       
            weight = decoder.output(tau[0],tau[1])
            opt.set_weight(weight)
            opt.set_interval(int(tau[0]),int(tau[1]))
      
            rhoa = opt.robustness(opt.w, rhop)
        
            rho.append(rhoa)

        andorrho=[]
        for i in range(len(self.ANDOR)):
            self.ANDOR[i].set_rho(np.array(rho))            
            out = self.ANDOR[i].output(self.ANDOR[i].w, np.array(rho))
            andorrho.append(out)


        self.And.set_rho(np.array(andorrho))

        
        return self.And.output(self.And.w, np.array(andorrho))


    def add_atomic(self,predicate, atomic):
        self.predicate.append(predicate)
        self.atomics.append(atomic)

        weight = random.uniform(0, 1)

        for operator in  self.ANDOR:
            w = list(operator.w)
            w.append(weight)
            operator.set_weight(np.array(w))
       
             

    def gradient_descent(self,  x, yd):

        ploss = self.output(x)-yd 
    
        topand = self.And
        weight = topand.w 

    


        d_topand_w =ploss*topand.gradient_w()
        d_topand_r = ploss*topand.gradient_r()
        and2 = self.ANDOR[0]
        or2 = self.ANDOR[1]
        weight2and = list(and2.w)
        weight2or = list(or2.w)
 

        d_and_w = d_topand_r[0]*and2.gradient_w()
        d_or_w  =d_topand_r[1]*or2.gradient_w()

        grad_atom = d_topand_r[0]*and2.gradient_r()+d_topand_r[1]*or2.gradient_r()
        predicates = self.predicates
        atomics = self.atomics

        d_D_W_o =[]
        d_D_W_h =[]
        d_E_W_o =[]
        d_E_W_h =[]
        d_p =[]

        for ia in range(len(atomics)):
     
            predicate = predicates[ia]
            atom = atomics[ia]
            encoder = atom.encoder
            decoder = atom.decoder
            opt = atom.operator

            gradient_p = opt.gradient_r()*predicate.gradient_w()*grad_atom[ia]
            d_p.append(mean(gradient_p)) 

            grad_r_atom = grad_atom[ia]*opt.gradient_w()
            rhop = predicate.output(x)
            tau = encoder.output(rhop)
            wig = decoder.output(tau[0],tau[1])


            D_W_o = list(decoder.W_o)
            D_W_h = list(decoder.W_h)
  
            temp =[]
            for i in range(len(D_W_o)):
                grad = grad_r_atom[i]*decoder.gradient_o(i,decoder.h_out)
                temp.append(grad)

            d_D_W_o.append(np.array(temp))
            Grad_o_d = np.array(temp) 
        
            temp =[]

            for j in range(len(D_W_h)):

                grad = decoder.gradient_h(j, tau, decoder.h_out, Grad_o_d)
                temp.append(grad)
       
            d_D_W_h.append(np.array(temp))
            Grad_h_d = np.array(temp)  

            E_W_o = list(encoder.W_o)
            E_W_h = list(encoder.W_h)


            de_in =[]
            for j in range(len(D_W_h[0])):
                grad = decoder.gradient_i(j, tau,  Grad_h_d)
                de_in.append(grad)

         
            temp =[]        
            for i in range(len(E_W_o)):
                grads = de_in[i]*encoder.gradient_o(i,encoder.h_out)
                temp.append(grads)

            d_E_W_o.append(np.array(temp))
            Grad_o_e = np.array(temp) 

            temp =[]
            for j in range(len(E_W_h)):
                grad = encoder.gradient_h(j,  x, encoder.h_out, Grad_o_e)
                temp.append(grad)
            d_E_W_h.append(np.array(temp))           


 

        return d_topand_w, d_p, d_and_w, d_or_w, d_D_W_o, d_D_W_h, d_E_W_o, d_E_W_h


    def update_NN(self,sig,eta, d_topand_w, d_p, d_and_w, d_or_w, d_D_W_o, d_D_W_h, d_E_W_o, d_E_W_h):

        #predicates = self.predicates
        for pre, w in zip(self.predicates,d_p):
            mw = 0.8*pre.mw  + 0.2*w
            vw =0.9*pre.vw +0.1*w**2
            pre.set_moment(mw,vw)
            h_mw = mw/0.2
            h_vw = vw/0.1
            h_vw = h_vw**(0.5)
            pre.set_weight(pre.constant - eta*w)

        #print  'chenages ', d_p
        atomics = self.atomics 
        rhoandor =[]
        for atom, pre, dDo, dDh, dEo, dEh in zip(self.atomics,self.predicates, d_D_W_o, d_D_W_h, d_E_W_o,d_E_W_h):
            new_D_W_o = atom.decoder.W_o - eta*dDo
            new_D_W_h = atom.decoder.W_h - eta*dDh
            new_E_W_o = atom.encoder.W_o- eta*dEo
            new_E_W_h = atom.encoder.W_h - eta*dEh

            atom.decoder.set_weight(new_D_W_o,new_D_W_h)
            atom.encoder.set_weight(new_E_W_o,new_E_W_h)
 
            rho = pre.output(sig)
            tau = atom.encoder.output(rho)

 
            atom.operator.set_interval(int(tau[0]),int(tau[1]))
            atom.operator.set_rho(rho)
            weight = atom.decoder.output(tau[0],tau[1])

            atom.operator.set_weight(weight)
             
            rhoandor.append(atom.operator.robustness(weight,rho))



        toprho =[]


        for i in range(len(self.ANDOR)):
            if i==0:
 
                new_and_w = self.ANDOR[i].w-eta*d_and_w
                self.ANDOR[i].set_weight(new_and_w)
                self.ANDOR[i].set_rho(np.array(rhoandor))
                toprho.append(self.ANDOR[i].output(self.ANDOR[i].w, self.ANDOR[i].rho))
                #print 'new_and_w', new_and_w

            else:
                new_or_w = self.ANDOR[i].w-eta*d_or_w
                #print 'new_or_w', new_or_w
                self.ANDOR[i].set_weight(new_or_w)
                self.ANDOR[i].set_rho(np.array(rhoandor))
                toprho.append(self.ANDOR[i].output(self.ANDOR[i].w, self.ANDOR[i].rho))

        #And = self.And
 

        new_And_w = self.And.w -eta*d_topand_w


        self.And.set_weight(new_And_w)
        self.And.set_rho(np.array(toprho))




    def train(self,sig,yd, eta, H):
        for i in range(H):
            out = self.output(sig)
            d_topand_w, d_p, d_and_w, d_or_w, d_D_W_o, d_D_W_h, d_E_W_o, d_E_W_h = self.gradient_descent(sig, yd)
            self.update_NN(sig, eta, d_topand_w, d_p, d_and_w, d_or_w, d_D_W_o, d_D_W_h, d_E_W_o, d_E_W_h)
            




            

            
