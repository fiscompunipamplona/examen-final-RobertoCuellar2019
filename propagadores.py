import numpy as np


class propagador2d_cuadrados:
    def propagadorFuncionTransferencia(self,entrada, l_onda, z):#propagador para campo cercano-medio
        N  = 768
        M = N
        dx = 19e-6
        LX = N*dx
        LY = LX
        pi = np.pi
        m = np.linspace(-N/2,(N/2)-1,N)
        Mx,My=np.meshgrid(m,m)
        H = np.exp(-1j*pi*l_onda*z*(Mx**2+My**2)/(LX**2))
        H = np.fft.fftshift(H)
        U1 = np.fft.fft2(np.fft.fftshift(entrada))
        U2 = (H*U1)
        fase = np.exp(1j*2*np.pi*z/l_onda)/(1j*l_onda*z) # se normaliza el factor de fase (IMPORTANTE)
        fase = fase/np.amax(fase)
        salida =np.fft.fftshift(np.fft.ifft2(U2))*fase
        return (salida)
    def propagadorRespuestaImpulso(self,entrada, l_onda,z): # propagador para campo medio-lejano
        k = 2*np.pi/l_onda
        N = 768
        dx = 19e-6
        LX = N*dx
        LX_ = l_onda*z*N/LX
        m = np.linspace(-N/2,N/2,N)
        Mx,My = np.meshgrid(m , m)
        h = np.exp(1j*np.pi*(LX**2)*(1/(l_onda*z*N**2))*(Mx**2+My**2))
        U1 = entrada*h
        fase = np.exp(1j*2*np.pi*z/l_onda)/(1j*l_onda*z) # se normaliza el factor de fase
        fase = fase/np.amax(fase)
        
        salida = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(U1)))*np.exp((1j*np.pi*l_onda*z/(LX**2))*(Mx**2+My**2))*fase
        return (salida)
    def propagadorFraccionario(self,entrada,p): #propagador fraccionario, abarca todo el campo
        N = 768 
        m = np.linspace(-N/2,(N/2)-1,N)
        Mx,My=np.meshgrid(m,m)
        u1 = entrada*np.exp(-1j*(np.pi/N)*(np.tan(p*np.pi/4))*(Mx**2+My**2))
        U = np.fft.fft2(np.fft.fftshift(u1))*np.fft.fftshift(np.exp(-1j*(np.pi/N)*(np.sin(p*np.pi/2))*(Mx**2+My**2)))
        u = np.fft.ifftshift(np.fft.ifft2(U))
        U2 = u*np.exp(-1j*(np.pi/N)*np.tan(p*np.pi/4)*(Mx**2+My**2))
        U3  = U2*np.exp(1j*(np.pi/N)*np.tan(p*np.pi/2)*(Mx**2+My**2))
        U4 = U3*np.exp(1j*(-np.pi*np.sign(p*np.pi/2)+2*p*np.pi/2+np.pi)/4)
        return (U4)


