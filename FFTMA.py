import numpy as np
import math
import skfuzzy as fuzzy


def fftma_l3c(m,n, range_h, range_v,err,model):
    range_vertical = range_v
    range_horizontal = range_h
    white_noises = np.random.sample(m*n)
    noises = white_noises.reshape(m,n)
    correlation_function = construct_correlation_function(range_vertical, range_horizontal, noises, model, 0)
    output = FFT_MA_3D( correlation_function, noises )
    output = output.reshape(m*n)
    output = make_it_gaussian(output)
    output = output.reshape(m,n)   
    return output
    
def FFT_MA_3D( correlation_function, noise ):
    S = noise.shape
    c2 = np.fft.ifftn(np.sqrt(np.abs(np.fft.fftn(correlation_function,S))) * np.fft.fftn(noise,S))
    simulation = np.real(c2);
    simulation = simulation[0:S[0], 0:S[1]]         
    return simulation

def construct_correlation_function(Lv, Lh, signal, Type, angle):
        ordem = 2
        I = signal.shape[0]
        size_output = '1d'
        if signal.ndim > 1:
            size_output = '2d'
            J = signal.shape[1]
            if signal.ndim > 2:
                size_output = '3d'
                K = signal.shape[2]
            else:
                K = 1
        else:
            J = 1
            K = 1
            correlation_function = np.zeros(I);
 
        desvio = 1/4
        
        if size_output == '1d':
            correlation_function = np.zeros(I);
        elif size_output == '2d':
            correlation_function = np.zeros((I,J));
        else:
            correlation_function = np.zeros((I,J,K));

        for i in np.arange(0,I):
            for j in np.arange(0,J):                
                x = round(i+1-I/2)
                y = round(j+1-J/2)
                if x == 0:
                    x=0.00001
                    
                theta = math.degrees(math.atan(y/x))
                a = 1/np.sqrt((math.sin(np.radians(angle-theta))**2)*(Lh**-2)+(math.cos(np.radians(angle-theta))**2)*(Lv**-2))
                h = np.sqrt( ( x )**2 + ( y )**2 )
                h = h/a
                
                if Type==1:
                    h = h*3
                    value = math.exp( -h**2 )
                if Type==2:
                    h = h*3
                    value = math.exp( -h );
                if Type==3:
                    if h<1:
                        value = 1 - 1.5 * h + 0.5 * h**3;
                    else:
                        value=0
                value_window = math.exp(-(np.abs((i+1-round(I/2))/(desvio*I))**ordem +  np.abs((j+1-round(J/2))/(desvio*J))**ordem))
                correlation_function[i,j] = value*value_window
                               
        
        return correlation_function
def make_it_gaussian(noise):

    
    noise = noise - np.min(noise)
    noise = noise / np.max(noise)
    
    logs_cumhist = np.unique(np.sort(noise))
    logs_cumhist_x = np.arange(1,noise.shape[0]+1) 
    logs_cumhist_x = logs_cumhist_x / np.max(logs_cumhist_x)
    noise2 = np.interp(noise,logs_cumhist, logs_cumhist_x)
    
    x = np.arange(-4,4,0.001)
    gaussiaN = fuzzy.membership.gaussmf(x, 0, 1)
    gaussiaN = np.unique(np.cumsum(gaussiaN))
    gaussiaN = gaussiaN - np.min(gaussiaN)
    gaussiaN = gaussiaN / np.max(gaussiaN)
    
    logs_cumhist = gaussiaN;
    logs_cumhist_x = np.arange(1,logs_cumhist.shape[0]+1) 
    logs_cumhist_x = logs_cumhist_x / np.max(logs_cumhist_x)
    noise3 = np.interp(noise2,logs_cumhist, logs_cumhist_x)
    
    noise3 = noise3 - np.min(noise3)
    noise3 = noise3/np.max(noise3)

    return noise3