# Teubner-Strey-Fits
# This script reads a .csv file containing x-ray scattering profiles and performs Teubner Strey-Fits given a temperature and sample name. 
# Adriana A. Rojas
# this code will use a robust linear-squares algorithm to fit a T-S fxn of the form y=1/(e*x**2+g) + 1/(a+b*x**2+c*x**4)to data
#-------------------------------------------------------------------
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import numpy as np
import csv
from matplotlib import rcParams
# ------------------------------------------------
def callfile(SampleName,temp):
# I had all my files named by sample containing scattering profiles for many temperatures, in one directory
# you can edit your directory to indicate where the file containing your data is located
    directory='C:\\Users\\Balsara\\'
    file=directory + SampleName + '_heat.csv'  
    #read the file
    infile=open(file,'r')
    table=[]
    table=[row for row in csv.reader(infile)]
    infile.close()
    #now transform string objects into float
    for r in range(2,len(table)):       # my first two elements were non-float values, so I started where appropriate
        for c in range(0,len(table[0])):
            table[r][c]=float(table[r][c])
    #------------------------------------------
    #now assign the table values to a particular name
    q=[]
    r130C=[]
    r110C=[]
    r90C=[]
    r80C=[]
    r70C=[]
    r60C=[]
    r55C=[]
    r50C=[]
    r45C=[]
    r40C=[]
    r35C=[]
    r30C=[]
    r25C=[]
    # the following identifies the location of each scattering profile in my sheet
    for r in range(2,len(table)):
        q.append(table[r][0])
        r25C.append(table[r][1])
        r30C.append(table[r][2])
        r35C.append(table[r][3])
        r40C.append(table[r][4])
        r45C.append(table[r][5])
        r50C.append(table[r][6])
        r55C.append(table[r][7])
        r60C.append(table[r][8])
        r70C.append(table[r][9])
        r80C.append(table[r][10])
        r90C.append(table[r][11])
        # following makes an exception for sheets containing data above 90C
        if len(table[0])>12:
            r110C.append(table[r][12])
            r130C.append(table[r][13])
    #-----------------------------------------------------------
    #now specify which temperatures to output to graph
    Isaxs=[]
    if temp=='90':
        Isaxs=r90C
    elif temp=='80':
        Isaxs=r80C
    elif temp=='70':
        Isaxs=r70C
    elif temp=='60':
        Isaxs=r60C
    elif temp=='55':
        Isaxs=r55C
    elif temp=='50':
        Isaxs=r50C
    elif temp=='45':
        Isaxs=r45C
    elif temp=='40':
        Isaxs=r40C        
    elif temp=='35':
        Isaxs=r35C
    elif temp=='30':
        Isaxs=r30C
    elif temp=='25':
        Isaxs=r25C
    elif temp=='110':
        Isaxs=r110C
    elif temp=='130':
        Isaxs=r130C
    elif temp=='':
        print('enter a valid temperature')
    return q, Isaxs
# ---------------------------------------------------------------------------------
newtemp='30'      # newtemp is the temperature I wish to analyze
muestra='AAR3Mg'  # muestra is my sample name
#
# identify labels for graphing purposes
if muestra=='AAR1Mg':
    polymer='(9.5-3.6)'
    clr='gold'
elif muestra=='AAR2Mg':
    polymer='(9.5-5.0)'
    clr='mediumorchid'
elif muestra=='AAR3Mg':
    polymer='(9.5-7.7)'
    clr='mediumblue'
elif muestra=='AAR4Mg':
    polymer='(9.5-8.5)'
    clr='forestgreen'
# --------------------------------------------
rcParams['font.family']='sans-serif'
rcParams['font.sans-serif']=['Arial']
size=10
# Call the files and read data--------------------------
qdata, Idata=callfile(muestra,newtemp)
qdata=np.array(qdata[100:300]) # here I can specify which part of the scattering profile on which to perform the fit
Idata=np.array(Idata[100:300])  # here I can specify which part of the scattering profile on which to perform the fit
#print(len(qdata))
# now plot the data----------------------------
plt.figure(figsize=(3,3),dpi=300)
plt.text(.1,40,newtemp+'$^\circ$C',fontsize=size)
plt.loglog(qdata,Idata,'o',color=clr,markersize=3,label=polymer)
plt.xlabel('q (nm $^{-1}$)', fontsize=size)
plt.ylabel('Intensity (cm$^{-1}$)', fontsize=size)
plt.tick_params(labelsize=size)
#plt.tick_params(right='on',direction='in',width=1)
plt.axis([.09,1.5,.01,10**2])
plt.tick_params(which='major',right='on',direction='in',top='on',length=6)
plt.tick_params(which='minor',right='on',direction='in',top='on',length=3)
plt.tight_layout()
# ----------------------------------------------------------------------------
# define the function computing residuals for least-squares minimization
# in the T-S fit
def func(x,q,I):
    # x is an array of coefficient fits where, 
    # x[0] = a, x[1] = b, x[2] = c, x[3] = e, x[4] = g
    Ibgd=1/(x[3]*q**2+x[4])
    Imod=1/(x[0]+x[1]*q**2+x[2]*q**4)+Ibgd
    return Imod-I
# -----------------------------------------------------------------
# now define a function to contain the T-S model 
def tsmod(q,a,b,c,e,g):
    Ibgd=1/(e*q**2+g)
    Imod=1/(a+b*q**2+c*q**4)
    Itot=Ibgd+Imod
    return Itot
# ------------------------------------------------------------------------
# define a function for just the background
def bgd(q,e,g):
    Ibgd=1/(e*q**2+g)
    return Ibgd
# -------------------------------------------------------------------------
# define a function to calculate xi (correlation length), d (domain spacing), and f (amphiphilicity) using the T-S results
def calc(a,b,c):
    xi=(0.5*(a/c)**0.5+b/(4*c))**(-0.5)
    d=2*np.pi*(0.5*(a/c)**0.5-b/(4*c))**(-0.5)
    f=b/(2*(a*c)**(0.5))
    return xi, d, f
#-----------------------------------------------------------------------
#x0=np.ones(5) #use all ones as the initial estimate
x0=[2.6,-88,829,14,1.2] # I can adjust what the starting guesses are 
# run robust least squares with loss='soft_l1', or loss='arctan',set f_scale to 0.1 which means
# that inlier residuals are approximately lower than 0.1
res_robust=least_squares(func,x0,loss='soft_l1', f_scale=0.1,args=(qdata,Idata))
#------------------------------------------------------------------------------
print('The results from the fit to the T-S model')
print('a=', res_robust.x[0],', b=',res_robust.x[1], 'c=',res_robust.x[2])
print('e=',res_robust.x[3],'g=',res_robust.x[4])
# the xi, d, and f values
xi, d, f=calc(res_robust.x[0], res_robust.x[1],res_robust.x[2])
print('xi=', xi, 'd=', d, 'f=', f)
plt.plot(qdata,tsmod(qdata,*res_robust.x),color='k',label='T-S Model')
plt.plot(qdata,bgd(qdata,res_robust.x[3],res_robust.x[4]),'c--',label='Ibgd')
plt.legend(fontsize=size)
plt.savefig(muestra+'_'+newtemp+'C.png')
