import numpy as np
import pandas as pd
from numpy import linalg as LA

#%%
# A be the matrix consists of patterns of class +1
# B be the matrix consists of patterns of class -1
# y belongs to {+1,-1}
def fit(A,B,y,lambda2):
    m=len(B) #  No. of patterns of class -1  
    m2=len(A)
    n=len(A[0]) #  No. of features or dimension of patterns
    e=np.ones(len(A))

    lambda1,lambda3,lambda4 =1,1,1
    delta=0.0001
    t0=1
    p=[]
    b0=0
    w0=np.zeros(n)  
    bcap=b0
    wcap=w0
    F0=m*(1-(delta/2))
        
    L_f1=(1/(delta))*sum([(1+np.inner(B[i],B[i])) for i in range(len(B))])
    L_f2=lambda1*sum([(1+np.inner(A[i],A[i])) for i in range(len(A))])  
    L_f=L_f1+L_f2
    L0=(2*L_f)/n
    
    distance11=np.zeros((m,1))
    distance22=np.zeros((m2,1)) # empty w^T X_i +b for m2 patterns of class +1
    cn=0
    k=0
    
    while(cn<3):
        k=k+1
        f1gradb=np.zeros(m)
        f1gradw=np.zeros((m,n))
        
        f2gradb=np.zeros(m2)
        f2gradw=np.zeros((m2,n))
    
    
        t=(1/2)*(np.sqrt(1+4*(t0)**2))
        
        for i in range(0,m):
            distance11[i]=y*(bcap+np.matmul(B[i].T,wcap))
            if distance11[i]>=1:
                f1gradb[i]=0
                f1gradw[i,:]=0
            if (1-delta) <= distance11[i] <=1:
                f1gradb[i]=-y*(1-distance11[i])/delta
                f1gradw[i,:]=(-y*B[i]*(1-distance11[i]))/delta
            if distance11[i]<(1-delta) :
                f1gradb[i]=-y
                f1gradw[i,:]=-y*B[i]
                
        for i in range(0,m2):
            distance22[i]=(bcap+np.matmul(A[i].T,wcap))
            f2gradb[i]=2*distance22[i]
            f2gradw[i,:]=2*A[i]*distance22[i]
            
        L=1.2*L0
        
        f1gradbsum=(sum(f1gradb))
        f2gradbsum=(sum(f2gradb))
        fgradbsum=f1gradbsum+f2gradbsum
        b=((L*bcap)-fgradbsum)/(L+lambda4)  # updated value of b in every iteration
        
        f1gradwsum=(f1gradw.sum(axis=0))
        f2gradwsum=(f2gradw.sum(axis=0))
        fgradwsum=f1gradwsum+f2gradwsum
        s1=(L*wcap)-fgradwsum  # inner part of S_{lambda}
        
        s2=abs(s1)-lambda2  # max(|t|-\lamda)
        
        s3=np.c_[s2,np.zeros(len(s2))]
        
        s=np.sign(s1)*s3.max(1)
        
        w= s/(L+lambda3) # updated value of w in every iteration
        
        distance1=np.zeros(m)
        distance12=np.zeros(m2)
        phi=np.zeros(m)
        
        for j in range(0,m):
            distance1[j]=y*(b+np.matmul(B[j].T,w))
            
            if distance1[j]>=1:
                phi[j]=0
            if (1-delta) <= distance1[j] <=1:
                phi[j]= ((1-distance1[j])**2)/(2*delta)
            if distance1[j] < (1-delta):
                phi[j]=1-distance1[j]-(delta/2)
                
        for j in range(0,m2):
            distance12[j]=(b+np.matmul(A[j].T,w))**2
            
                
        f1=sum(phi)
        f2=(lambda1/2)*sum(distance12)
        f=f1+f2
        
        g= (lambda2*LA.norm(w,1))+(lambda3/2)*(LA.norm(w)**2)+(lambda4/2)*(b**2)
        
        F=f+g
        
        omega=min((t0)/t,np.sqrt(L0/L))
        
        if F> F0:
            bcap=b
            wcap=w
        else:
            bcap=b+omega*(b-b0)
            wcap=w+omega*(w-w0)
         
        u0=np.r_[b0,w0]
        u=np.r_[b,w]
        
        if ((F0-F)/(1+F0)<=10**(-6) and LA.norm(u0-u)/(1+LA.norm(u0)) <=10**(-6)):  
          p.append(k)
          nrp=len(p)
          if(nrp>=3 and p[nrp-1]-p[nrp-2]==1 and p[nrp-2]-p[nrp-3]==1):
              cn=3
            
        t0=t
        F0=F
        L0=L
        b0=b
        w0=w
    return w,b
#%%
def predict(x,w,b):
    pred=np.zeros(len(x))
    for i in range(len(x)):
        d=np.zeros(2)
        for r in range(2):
            d[r]=abs(np.matmul(x[i].T,w[:,r])+b[:,r])/LA.norm(w[:,r])
            label=np.argmin(d)
        if label==1:
            pred[i]= -1
        else:
            pred[i]= +1
    return pred