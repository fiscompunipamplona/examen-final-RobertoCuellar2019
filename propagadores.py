import numpy as np


class propagador2d_cuadrados:
    def propagadorFuncionTransferencia(self,entrada, l_onda, z):
        N  = 768
        dx = 19e-6
        LX = N*dx
        pi = np.pi
        m = np.linspace(-N/2,N/2,N)
        Mx,My=np.meshgrid(m,m)
        H = np.exp(-1j*pi*l_onda*z*(Mx**2+My**2)/(LX**2))
        H = np.fft.fftshift(H)
        U1 = np.fft.fft2(np.fft.fftshift(entrada))
        U2 = (H*U1)
        salida =np.fft.fftshift(np.fft.ifft2(U2))*np.exp(1j*2*np.pi*z/l_onda)/(1j*l_onda*z)
        return (salida)
    def propagadorRespuestaImpulso(self,entrada, l_onda,z):
        k = 2*np.pi/l_onda
        N = 768
        dx = 19e-6
        LX = N*dx
        LX_ = l_onda*z*N/LX
        m = np.linspace(-N/2,N/2,N)
        Mx,My = np.meshgrid(m , m)
        h = np.exp(1j*np.pi*(LX**2)*(1/(l_onda*z*N**2))*(Mx**2+My**2))
        U1 = entrada*h
        salida = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(U1)))
        return (salida)


