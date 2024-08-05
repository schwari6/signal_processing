import math, cmath
import numpy as np
import matplotlib.pyplot as plt

##python scripts that I made
from message_GUI import ButtonMessageBox, MessageBox
from Plots import plot

def main():
    
    #parameters
    pi = math.pi
    j = complex(0,1)
    exp = cmath.exp

    '''
    Calculates a signal fourier series coefficients
    Input parameters: x - signal
                      T1,T2 - time vectors
                      w0 - signal's frequency
    Output: Discrete Fourier Series (dfs) coefficients as nd-array
    '''
    def DFS(x,T1,T2,w0):
        dfs = np.zeros(T1,dtype=complex)
        for k in range(-T1//2, T1//2):
            for n in range(-T2//2, T2//2):
                dfs[T1//2+k] += (1/T1) * x[T2//2+n] * exp(-j*k*w0*n)
        return dfs
    
    '''
    regenerating a signal from its fourier series coefficients
    Input parameters: x - fourier series coefficients
                      T1,T2 - time vectors
                      w0 - signal's frequency
    Output: The original signal
    '''
    def iDFS(x,T1,T2,w0):
        signal = np.zeros(T1,dtype = complex)
        for n in range(-T1//2, T1//2):
            for k in range(-T2//2, T2//2):
                signal[T1//2+n] += x[T2//2+k] * exp(j*k*w0*n)
        return signal
   
    '''
    Calculates a signal fourier transform
    Input parameters: x - signal
                      T1,T2 - time vectors
                      w0 - signal's frequency
    Output: Discrete Time Fourier Transform (dtft) as nd-array
    '''
    def DTFT(x,freq,T):
        dtft = np.zeros(T,dtype=complex)
        for i, w1 in enumerate(freq):
            for n in range(-T//2,T//2):
                dtft[i] += x[T//2+n] * exp(-j * w1 * n)
        return dtft
    
    '''
    Calculates a signal from its fourier transform
    Input parameters: x - signal
                      T1,T2 - time vectors
                      w0 - signal's frequency
    Output: time domain signal
    '''
    def iDTFT(x,freq,T):
        signal = np.zeros(T,dtype=complex)
        for n in range(-T//2,T//2):
            for i,w1 in enumerate(freq):
                signal[T//2+n] += x[i] * exp(j*w1*n)
        return signal/(2*pi)
    
    '''
    Makes decimation to a signal
    Input parameters: x - signal
                      factor - number of zeros between two samples
                      T - time vector
    Output: decimated signal
    '''
    def decimation(x,factor,T):
        sample = np.zeros(T,dtype = complex)
        for i in range(-T//2, T//2,factor):
            sample[T//2+i] = x[T//2+i]
        return sample
        
    '''
    Makes interpolation of a signal
    Input parameters: x - signal
                      factor - number of zeros between two samples
                      T - time vector
    Output: interpolated signal
    '''
    def interpolation(x,factor,T):
        trimmed = x[x != 0]
        temp = np.zeros(factor*len(trimmed),dtype = complex)
        temp[::factor] = trimmed
        sample = np.zeros(T,dtype=complex)
        for i in range(len(temp)):
            sample[900-100*(factor-1)+i] = temp[i]
        return sample
    
    '''
    Makes ZOH interpolation of a signal
    Input parameters: x - signal
                      factor - number of zeros between two samples
                      T - time vector
    Output: ZOH interpolated signal
    '''
    def ZOH(t, x, factor):
        zoh = np.zeros(len(t))
        zoh[0] = x[0]
        temp = 0
        for i in range(1,len(t)):
            if temp == 0:
                val = x[i]
            if temp == factor-1:
                temp = 0
            else:
                temp += 1
            zoh[i] = val
        return zoh

    '''
    Makes FOH interpolation of a signal
    Input parameters: x - signal
                      factor - number of zeros between two samples
                      T - time vector
    Output: FOH interpolated signal
    '''
    def FOH(t, x, factor):
        foh = np.zeros_like(t)
        foh[0] = x[0]
        temp = 0
        for i in range(1,len(t)):
            if temp == 0:
                foh[i] = x[i]
            else:
                t0, t1 = t[i], t[i + 1]
                x0, x1 = x[i], x[i + 1]
                # Linear interpolation
                foh[i] = x0 + (x1 - x0) / (t1 - t0)
            if temp == factor-1:
                temp = 0
            else:
                temp += 1
        return foh

    N=2000 #signal period
    omega = 2*pi/N #signal frequency
    n_vector = np.arange(-N//2, N//2, dtype = complex) #time vector
    
    #frequency vector
    w = np.zeros(N,dtype=complex) 
    for i in range(N):
        w[i] = omega*n_vector[i]

    #1 - signal
    a_n = np.zeros(N,dtype=complex)
    a_n[np.abs(n_vector) < 100] = 1 #window function rect[n]<100
    
    plot('stem',n_vector,a_n,'plot of a[n]','n','Amplitude')
    
    #2 - furier series coefficients
    a_k = DFS(a_n,N,N,omega) #calculates a[n] DFS
    plot('stem',w,a_k,'a[n] coefficients - Dirichlet kernel','Frequency','Magnitude')
    
    #3 - shift in time domain <-> multiplication with exponent in frequency domain
    b_k = np.zeros(N,dtype=complex)
    shift = 100 #shift time of signal
    for k in range(-N//2, N//2):
        b_k[N//2 + k] = a_k[N//2+k] * exp(-j*k*omega*shift)

    b_n = iDFS(b_k,N,N,omega) #restore signal from its DFS
    plot('stem',n_vector,b_n,'Plot of b[n] = a[n-100]','n','Amplitude')
    
    #4 - time domain derivate <-> multiplication with 'k' in frequency domain
    c_k = np.zeros(N,dtype=complex)
    
    #multiply c_k by (1-exp(-jkwn))
    for k in range(-N//2, N//2):
        c_k[N//2 + k] = a_k[N//2+k] * (1-exp(-j*k*omega))
    
    c_n = iDFS(c_k,N,N,omega) #restore signal from its DFS 
    plot('stem',n_vector,c_n,'Plot of c[n] = derivate of a[n]','n','Amplitude')

    #5 - time domain convolution <-> frequency domain multiplication
    d_k = np.zeros(N,dtype=complex)
    
    #d_k = a_k^2 * N
    for k in range(-N//2, N//2):
        d_k[N//2 + k] = a_k[N//2+k]**2 * N
    
    d_n = iDFS(d_k,N,N,omega) #restore signal from its DFS
    plot('stem',n_vector,d_n,'Plot of d[n] = a[n] convolution','n','Amplitude')
    
    #6 - parseval theorm
    parseval_A = 0
    parseval_B = 0
    for i in range(-N//2, N//2):
        parseval_A += np.abs(d_n[N//2+i])**2
        parseval_B += np.abs(d_k[N//2+i])**2
    parseval_A = parseval_A / N
    result = np.isclose(parseval_A, parseval_B)
    MessageBox("Parseval Theorm", str(result))

    #7 - time domain multiplication <-> frequency domain convolution
    # DFS of time multiplication
    e_n = a_n * b_n
    e_k = DFS(e_n,N,N,omega)
    plot('stem',w,e_k,'e[n] coefficients - straight forward','Frequency','Magnitude')
    
    #DFS of frequency domain cyclic convolution
    e_k_hat = np.zeros(N,dtype=complex)
    for k in range(-N//2,N//2):
        b_k_temp = list(b_k[:N//2+k+1][::-1]) + list(b_k[N//2+k+1:][::-1])       
        for l in range(-N//2,N//2): 
            e_k_hat[N//2 + k] += a_k[N//2 + l] * b_k_temp[N//2+l]
    
    e_k_hat = list(e_k_hat[N//2:]) + list(e_k_hat[:N//2]) #shift signal
    plot('stem',w,e_k_hat,'e[n] coefficients - cyclic convolution','Frequency','Magnitude')
    
    #8 - cosine multiplication
    cosine = np.cos(omega * 500 * n_vector)
    g_n = a_n * cosine
    g_k = DFS(g_n,N,N,omega) #calculates g[n] DFS
    plot('stem',w,np.real(g_k),'g[n] coefficients - cosine multiplication','Frequency','Magnitude')
    
    #9 - sinus multiplication
    sinus = np.sin(omega * 500 * n_vector)
    h_n = a_n * sinus
    #plot('stem',n_vector,np.abs(h_n),'plot of h[n]','n','Amplitude')
    
    h_k = DFS(h_n,N,N,omega) #calculates h[n] DFS
    plot('stem',w,-np.imag(h_k),'h[n] coefficients - sinus multiplication','Frequency','Magnitude')
    
    #Hilbert filter
    H_k = np.zeros(N,dtype=complex)
    for k in range(-(N-1)//2,(N-1)//2):
        H_k[N//2+k] = -j * np.sign(k)
    h_k_hat = H_k * g_k #apply hilbert on g_k
    plot('stem',w,-np.imag(h_k_hat),'h[n] coefficiens - Hilbert effect','Frequency','Magnitude')

    #plt h_n_hat - hilberts time domain signal result
    h_n_hat = iDFS(h_k_hat,N,N,omega)
    plot('stem',n_vector,np.abs(h_n_hat),'Plot of h[n] - Hilbert effect','n','Amplitude')
    
    #10 - frequency domain stretch <-> time domain contraction
    alpha = 5 # factor of stretching
    f_k = np.zeros(5*N,dtype=complex)
    #pad with 0
    for i in range(-N//2, N//2):
      f_k[(N//2+i)*5]  = (1/alpha) * a_k[N//2+i]
      
    #plot f_n
    f_n = iDFS(f_k,N,5*N,omega) # regenerate f[n] from its DFS
    plot('stem',n_vector,f_n,'plot of f[n]','n','Amplitude')
    
    #11 - Gibbs theorm
    #plot of M 900:1000
    y = 1 
    for M in range(900,1001,20):   
        a_m = np.zeros(N,dtype=complex)
        # calulate finite numbers (M) of coefficients
        for m in range(-N//2, N//2):
          for i in range(-M, M):
            a_m[N//2+m] += (1/N) * a_k[N//2+i] * cmath.exp(j*m*omega*i)
            
        plt.subplot(2,3,y)
        plot('stem',n_vector,a_m,f"Plot of A{M}[n]",'n','Amplitude',show = False)
        y+=1
        
        # find Gibbs error -> lim{S(x+) - f(x+)} = a * k
        if M==900:
            gibbs = 1 - N * np.real(a_m[903]-a_m[900])   
            MessageBox("Gibbs error value:", gibbs)
            
    plt.tight_layout()
    plt.show()
    
    #12 - decimation
    
    ft = DTFT(a_n,w,N) # original signal fourier transform
    plot('stem',w,ft,"Dtft of a[n]",'Frequency','Magnitude')
    
    
    #Decimation
    decimated_signals = [None,None,None,None,None]
    dtft_of_decimated_signals = [None,None,None,None,None]
    
    for M in range(2,5):
        a_n_decimated = decimation(a_n, M, N) #decimatee signal by factor of M
        dtft_of_decimated = DTFT(a_n_decimated,w,N) #fourier transform
        
        plt.subplot(2,1,1)
        plot('stem',n_vector,a_n_decimated,f"a[n] decimated by M={M}",'n','Amplitude',show=False)
        plt.subplot(2,1,2)
        plot('stem',w,np.abs(dtft_of_decimated),"Fourier Transform",'Frequency','Magnitude')
        
        decimated_signals[M] = a_n_decimated
        dtft_of_decimated_signals[M] = dtft_of_decimated
    
    #13
    #interpolation
    interpolated_signals = [None,None,None,None,None]
    dtft_of_interpolated_signals = [None,None,None,None,None]
    
    for M in range(2,5):
        a_n_interpolated = interpolation(a_n, M, N) #interpolate by factor of M
        dtft_of_interpolated = DTFT(a_n_interpolated,w,N) #fourier transform
        interpolated_signals[M] = a_n_interpolated
        dtft_of_interpolated_signals[M] = dtft_of_interpolated
        
        plt.subplot(2,1,1)
        plot('stem',n_vector,a_n_interpolated,f"a[n] interpolated by M={M}",'n','Amplitude',show=False)
        
        plt.subplot(2,1,2)
        plot('stem',w,np.abs(dtft_of_interpolated),"Fourier Transform",'Frequency','Magnitude')
        
    #14 - Filtering - LPF in frequency domain
    filters = [None,None,None,None,None]
    
    for M in range(2,5):
        LPF = np.zeros(N,dtype=complex)
        LPF[abs(w) < (pi/M)] = pi*M*2/N
        dtft_interpolated = dtft_of_interpolated_signals[M]
        restored = LPF * dtft_interpolated # filter the signal
        time_restored = iDTFT(restored,w,N) #inverse transform
        filters[M] = LPF #save LPF in list
        
        plt.subplot(2,1,1)
        plot('plot',w,LPF,f"low pass filter for M={M}",'Frequency','Magnitude',show=False)
        
        plt.subplot(2,1,2)
        plot('stem',n_vector,time_restored,f"restored signal for M={M}",'n','Amplitude')
        
    #15 - filtering in time domain

    for M in range(2,5):
        LPF = filters[M]
        a_n_interpolated = interpolated_signals[M]
        timed_LPF = iDTFT(LPF,w,N) #show impulse response of the filter
        
        # filter signal in time domain
        filtered_a_n = np.zeros(N,dtype=complex)
        for i in range(-N//2,N//2):
            for n in range(-N//2,N//2):
                filtered_a_n[N//2+i] += a_n_interpolated[N//2+n] * timed_LPF[i-n]
        
        filtered_a_n = list(filtered_a_n[N//2:]) + list(filtered_a_n[:N//2])
        
        plt.subplot(2,1,1)
        plot('stem',n_vector,np.abs(timed_LPF),f"impulse response for M={M}",'n','Amplitude',show=False)
        
        plt.subplot(2,1,2)
        plot('stem',n_vector,filtered_a_n,f"time filtered signal for M={M}",'n','Amplitude')  
        
    
    #16 - ZOH and FOH interpolation

    for M in range(2,5):
        a_n_interpolated = interpolated_signals[M]
        Zoh = ZOH(n_vector,a_n_interpolated,M)
        Foh = FOH(n_vector,a_n_interpolated,M)
        
        plt.subplot(2,1,1)
        plot('stem',n_vector,Zoh,f"ZOH for M={M}",'n','Amplitude',show=False)
        
        plt.subplot(2,1,2)
        plot('stem',n_vector,Foh,f"FOH for M={M}",'n','Amplitude')
    
    #17 - un uniform decimation 

    a_n_unUniform_decimation = np.zeros(N,dtype=complex)
    k = 1
    i = 0
    while i < N:
        a_n_unUniform_decimation[i] = a_n[i]
        i+=k
        k = 2 if k==1 else 3 if k==2 else 1

    plt.subplot(2,1,1)
    plot('stem',n_vector,a_n_unUniform_decimation,"Un uniformed decimation",'n','Amplitude',show=False)
    
    unUniform_fourier = DTFT(a_n_unUniform_decimation,w,N)
    plt.subplot(2,1,2)
    plot('stem',w,np.abs(unUniform_fourier),"fourier transform",'Frequency','Magnitude')
    
    LPF = np.zeros(N,dtype=complex)
    LPF[abs(w) < (pi/4)] = pi*4/N
    filtered_unUniformed = LPF * unUniform_fourier
    timed_filtered_unUniformed = iDTFT(filtered_unUniformed,w,N)
    
    plt.subplot(2,1,1)
    plot('stem',w,filtered_unUniformed,"filtered",'Frequency','Magnitude',show=False)

    plt.subplot(2,1,2)
    plot('stem',n_vector,timed_filtered_unUniformed,"timed filtered",'n','Amplitude')
    
    
if __name__ == '__main__':
    main()