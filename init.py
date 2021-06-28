from modular import *
import random
import numpy as np 

LENGTH_SIGNAL = 128
NUM_D_HIDDEN = 5
NUM_E_HIDDEN =5


def random_weight(lenght):
    weight = []
    for i in range(lenght):
        weight.append(random.uniform(0.0,1.0))
    return weight

def generate_operator():
    opt =[4]
    tau0 =10
    chose = random.choice(opt)
    weight = random_weight(LENGTH_SIGNAL)
    tau1 = random.randint(LENGTH_SIGNAL/4,LENGTH_SIGNAL/2)
    tau2 = tau1+ random.randint(LENGTH_SIGNAL/4,LENGTH_SIGNAL/2)
    if chose ==1:
        return Always(weight,tau1,tau2)
    elif chose==2:
        return Eventually(weight,tau1,tau2)

    elif chose==3:
        return EventualAlways(weight, 10, tau1,tau2)

    else:
        return AlwaysEventual(weight,tau0,tau1,tau2)
        


def generate_predicate():
    cho = [True, False]
    return Predicate(random.choice(cho),random.uniform(0.0,1.0))


def generate_D_W_o():
    D_W_o =[]
    for i in range(LENGTH_SIGNAL):
        D_W_o.append(random_weight(NUM_D_HIDDEN))

    return D_W_o

def generate_D_W_h():
    D_W_h=[]
    for i in range(NUM_D_HIDDEN):
        D_W_h.append(random_weight(2))

    return D_W_h

def generate_E_W_o():
    E_W_o=[]
    for i in range(2):
        E_W_o.append(random_weight(NUM_E_HIDDEN))
    return E_W_o

def generate_E_W_h():
    E_W_h =[]
    for i in range(NUM_E_HIDDEN):
        E_W_h.append(random_weight(LENGTH_SIGNAL))

    return E_W_h


def generate_atom():
    E_W_o = generate_E_W_o()
    E_W_h = generate_E_W_h()
    D_W_o = generate_D_W_o()
    D_W_h = generate_D_W_h()
    decoder = Decoder(D_W_o,D_W_h)
    encoder = Encoder(E_W_o, E_W_h,LENGTH_SIGNAL,0.1)
    operator = generate_operator()

    return atomic_formula(encoder, decoder,operator)
