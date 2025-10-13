import math
import numpy as np
import matplotlib.pyplot as plt

#u,t = -V u,x + k u,xx  -lamda u + (src?) src est donne pour forcer u=uex
#uex,t + V uex,x - k uex,xx  + lamda uex = src = F[j]*np.sin(freq*t)+Tex[j]*np.cos(freq*t)*freq


iplot=1

# PHYSICAL PARAMETERS
K = 0.01     #Diffusion coefficient
xmin = 0.0
xmax = 1.0    
Time = 2.  #Integration time

V=1.
lamda=1
freq=2*math.pi*3

#mesh adaptation param

niter_refinement=10      #niter different calculations
hmin=0.01
hmax=0.5
err=0.01

# NUMERICAL PARAMETERS
NX = 10    #Number of grid points : initialization
NT = 1000   #Number of time steps max
ifre=1000000  #plot every ifre time iterations
eps=0.001     #relative convergence ratio

for metric_insta in [False, True]:
    
    errorL2=[]
    errorH1=[]
    itertab=[]
    hloc = np.ones((NX))*hmax*0.5
    
    itera=0
    NX0=0
    while( np.abs(NX0-NX) > -10 and itera<niter_refinement):
    
        itertab.append((xmax-xmin)/NX)
        itera+=1
        
        x = np.linspace(xmin,xmax,NX)
        T = np.zeros((NX))
    
    #mesh adaptation using local metric
        if(itera>0):
            xnew=[]
            Tnew=[]        
            nnew=1
            xnew.append(xmin)
            Tnew.append(T[0])        
            while(xnew[nnew-1] < xmax-hmin):
                for i in range(0,NX-1):
                    if(xnew[nnew-1] >= x[i] and xnew[nnew-1] <= x[i+1] and xnew[nnew-1]<xmax-hmin):
                        hll=(hloc[i]*(x[i+1]-xnew[nnew-1])+hloc[i+1]*(xnew[nnew-1]-x[i]))/(x[i+1]-x[i])
                        hll=min(max(hmin,hll),hmax)
                        nnew+=1
    #                    print(nnew,hll,min(xmax,xnew[nnew-2]+hll))
                        xnew.append(min(xmax,xnew[nnew-2]+hll))                
    #solution interpolation for initialization (attention initial solution on first mesh in the row)
                        un=(T[i]*(x[i+1]-xnew[nnew-1])+T[i+1]*(xnew[nnew-1]-x[i]))/(x[i+1]-x[i])
                        Tnew.append(un)
                        
            NX0=NX
            NX=nnew
            x = np.linspace(xmin,xmax,NX)
            x[0:NX]=xnew[0:NX]
            #print(x)
            T = np.zeros((NX))
            T[0:NX]=Tnew[0:NX]
    #        T[NX-1]=0
        
        rest = []
        F = np.zeros((NX))
        RHS = np.zeros((NX))
        hloc = np.ones((NX))*hmax*0.5
        metric = np.zeros((NX))
    
        Tex = np.zeros((NX))
        for j in range (1,NX-1):
            Tex[j] = np.exp(-20*(x[j]-(xmax+xmin)*0.5)**2)
            
        dt=1.e30
        for j in range (1,NX-1):
            Tx=(Tex[j+1]-Tex[j-1])/(x[j+1]-x[j-1])
            Txip1=(Tex[j+1]-Tex[j])/(x[j+1]-x[j])
            Txim1=(Tex[j]-Tex[j-1])/(x[j]-x[j-1])
            Txx=(Txip1-Txim1)/(0.5*(x[j+1]+x[j])-0.5*(x[j]+x[j-1]))
            F[j]=V*Tx-K*Txx+lamda*Tex[j]
            dt=min(dt,0.25*(x[j+1]-x[j-1])**2/(V*np.abs(x[j+1]-x[j-1])+4*K+np.abs(F[j])*(x[j+1]-x[j-1])**2))
    
        print('NX=',NX,'Dt=',dt)        
    
        #time step loop
        n=0
        res=1
        res0=1
        t=0
        while(n<NT and t<Time):
            n+=1
            dt=min(dt,Time-t)
            t+=dt
              
        #discretization of the advection/diffusion/reaction/source equation
            res=0
            for j in range (1, NX-1):
    #viscosite numerique : decentrage pour stabilite de derivee premiere/advection 12.17
                visnum=0.25*(0.5*(x[j+1]+x[j])-0.5*(x[j]+x[j-1]))*np.abs(V) #0.5 h |V|
                xnu=K+visnum            
                Tx=(T[j+1]-T[j-1])/(x[j+1]-x[j-1])
                Txip1=(T[j+1]-T[j])/(x[j+1]-x[j])
                Txim1=(T[j]-T[j-1])/(x[j]-x[j-1])
                Txx=(Txip1-Txim1)/(0.5*(x[j+1]+x[j])-0.5*(x[j]+x[j-1]))
                src=F[j]*np.sin(freq*t)+Tex[j]*np.cos(freq*t)*freq
                RHS[j] = dt*(-V*Tx+xnu*Txx-lamda*T[j]+src)
                if(metric_insta):
                    metric[j]+=min(1./hmin**2,max(1./hmax**2,abs(Txx)/err))
                elif(metric_insta==False and (n==NT or t>=Time)):
                    metric[j]=min(1./hmin**2,max(1./hmax**2,abs(Txx)/err))            
                res+=abs(RHS[j])
    
            metric[0]=metric[1]
            metric[NX-1]=metric[NX-2]
    
            for j in range (1, NX-1):
                T[j] += RHS[j]   #Tn+1 = Tn + dt*(-V*Tx+xnu*Txx-lamda*T[j]+src)
                RHS[j]=0
                
        
            T[0]=0
            T[NX-1]=2*T[NX-2]-T[NX-3]  #Txx=0 second derivative
    
            if (n == 1 ):
                res0=res
    
            rest.append(res)
        #Plot every ifre time steps
    #         if (n%ifre == 0 or t>=Time):
    #             plt.figure()
    #             #print('iter=',n,'residual=',res)
    #             plotlabel = "iter adapt = %1.0f" %itera
    # #            plotlabel = "t = %1.2f" %t
    #             plt.plot(x[0:NX],T[0:NX], label=plotlabel,linestyle='--', marker='o')
          
        if(metric_insta):
            metric[0:NX]/=n  #average (intersect) over n iterations
           
        hloc[0:NX]=np.sqrt(1./metric[0:NX])
        
        # print('iter=',n,'time=',t,'residual=',res)
        # plt.xlabel(u'$x$', fontsize=26)
        # plt.ylabel(u'$T$', fontsize=26, rotation=0)
        # plt.title(u'ADRS insta 1D')
        #plt.legend()
    
        # plt.figure()
        # plt.plot(np.log10(rest/rest[0]))
        # plt.show()
    
    # #    errL2=np.sqrt(np.dot(T-Tex,T-Tex))
        errH1h=0
        errL2h=0
        for j in range (1, NX-1):
            Texx=(Tex[j+1]-Tex[j-1])/(x[j+1]-x[j-1])
            Tx=(T[j+1]-T[j-1])/(x[j+1]-x[j-1])
            errL2h+=(0.5*(x[j+1]+x[j])-0.5*(x[j]+x[j-1]))*(T[j]-Tex[j])**2
            errH1h+=(0.5*(x[j+1]+x[j])-0.5*(x[j]+x[j-1]))*(Tx-Texx)**2
    
        errorL2.append(errL2h)
        errorH1.append(errL2h+errH1h)
        
        
        print(metric_insta,itera,'norm error L2, H1=',errL2h,errH1h)
    
    #    plt.figure(3)
    plt.plot(errorL2,label=str(metric_insta))
    #plt.plot(np.log10(errorH1))    
plt.legend()
plt.show()