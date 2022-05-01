import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

def pdf_crosses(a, pdfx, pdfy):
    
    crosses = []
    if a < pdfy[0]:
        crosses.append(pdfx[0])
    for i in range(len(pdfx)-1):
        if (pdfy[i] < a and pdfy[i+1] > a):
            crosses.append(np.interp(a, pdfy[i:i+2], pdfx[i:i+2]))
        elif (pdfy[i] > a and pdfy[i+1] < a):
            crosses.append(np.interp(a, pdfy[i:i+2][::-1], pdfx[i:i+2][::-1]))
    if a < pdfy[-1]:
        crosses.append(pdfx[-1])
        
    if len(crosses) % 2 != 0:
        print(a,pdfx,pdfy, crosses)
        raise Exception('No idea what to do here!')
    
    return crosses
    
def integrate_posterior(a, pdfx, pdfy):
    if a > np.max(pdfy)*0.98 or a <= 1e-3:
        return np.inf
      
    CHOOSE = pdfy > 1
    crosses = pdf_crosses(a, pdfx[CHOOSE], pdfy[CHOOSE])
    total = 0
    for i in range(0, len(crosses), 2):
        total += quad(lambda x: np.interp(x, pdfx,pdfy), crosses[i], crosses[i+1])[0]
        
    return total
  
def bayes_interval(S,P, prob = 0.682689):
    bayes_a = minimize(lambda a: np.abs(integrate_posterior(a,S,P) - credible_probability), x0 = np.max(P)/2.,method = 'Nelder-Mead')
    return pdf_crosses(bayes_a.x[0], S, P)
