import numpy as np
import xclimf

ymk = 4.0/5

ymi = 2.0/5

um = np.array([ 0.00993818,  0.00977227,  0.00490103,  0.00735685,  0.00020597,
        0.00176924,  0.00628428,  0.00180322,  0.00579694,  0.00968024])
        
vi = np.array([ 0.00485023,  0.0068724 ,  0.00514634,  0.00873745,  0.00459788,
        0.00505054,  0.00158364,  0.00947761,  0.00305369,  0.00633921])
        
vk = np.array([ 0.00812807,  0.00018544,  0.00326339,  0.00613061,  0.00909165,
        0.00288403,  0.0081203 ,  0.00619695,  0.00792532,  0.00632894])

# ==============================================================================

fmi = np.dot(um, vi)
fmk = np.dot(um, vk)

print("fmi", fmi)
print("fmk", fmk)

fmk_fmi = fmk - fmi
fmi_fmk = fmi - fmk

print("fmk_fmi", fmk_fmi)
print("fmi_fmk", fmi_fmk)

g_fmk_fmi = xclimf.g(fmk_fmi)
g_fmi_fmk = xclimf.g(fmi_fmk)
dg_fmk_fmi = xclimf.dg(fmk_fmi)
dg_fmi_fmk = xclimf.dg(fmi_fmk)

print("g(fmk-fmi)", g_fmk_fmi)
print("g(fmi-fmk)", g_fmi_fmk)
print("dg(fmk-fmi)", dg_fmk_fmi)
print("dg(fmi-fmk)", dg_fmi_fmk)

div1 = 1/(1 - (ymk * g_fmk_fmi))
div2 = 1/(1 - (ymi * g_fmi_fmk))
div1_div2 = div1 - div2

print("div1", div1)
print("div2", div2)
print("div1_div2", div1_div2)

reg = 0.001 * vi

print("reg", reg)

pdi1 = ymk * dg_fmi_fmk * div1_div2
pdi2 = ymi * pdi1
pdi3 = pdi2 * um
pdi4_reg = pdi3 - reg
pdi5_gamma = 0.001 * pdi4_reg

print("pdi1", pdi1)
print("pdi2", pdi2)
print("pdi3", pdi3)
print("pdi4_reg", pdi4_reg)
print("pdi4_gamma", pdi5_gamma)




