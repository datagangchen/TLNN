function [SK,M4,M2,f] = SK_W(x,Nfft,Noverlap,Window)
% [SK,M4,M2,f] = SK_W(x,Nfft,Noverlap,Window) 
% Welch's estimate of the spectral kurtosis       
%       SK(f) = M4(f)/M2(f)^2 - 2 
% where M4(f) = E{|X(f)|^4} and M2(f) = E{|X(f)|^2} are the fourth and
% second order moment spectra of signal x, respectively.
% Signal x is divided into overlapping blocks (Noverlap taps), each of which is
% detrended, windowed and zero-padded to length Nfft. Input arguments nfft, Noverlap, and Window
% are as in function 'PSD' or 'PWELCH' of Matlab. Denoting by Nwind the window length, it is recommended to use 
% nfft = 2*NWind and Noverlap = 3/4*Nwind with a hanning window.
% (note that, in the definition of the spectral kurtosis, 2 is subtracted instead of 3 because Fourier coefficients
% are complex circular)
%
% --------------------------
% References: 
% J. Antoni, The spectral kurtosis: a useful tool for characterising nonstationary signals, Mechanical Systems and Signal Processing, Volume 20, Issue 2, 2006, pp.282-307.
% J. Antoni, R. B. Randall, The spectral kurtosis: application to the vibratory surveillance and diagnostics of rotating machines, Mechanical Systems and Signal Processing, Volume 20, Issue 2, 2006, pp.308-331.
% --------------------------
% Author: J. Antoni
% Last Revision: 12-2014
% --------------------------
if length(Window) == 1
    Window = hanning(Window);
end
Window = Window(:)/norm(Window);		% window normalization
n = length(x);							% number of data points
nwind = length(Window); 				% length of window
% check inputs
if nwind <= Noverlap,error('Window length must be > Noverlap');end
if Nfft < nwind,error('Window length must be <= Nfft');end
x = x(:);		
k = fix((n-Noverlap)/(nwind-Noverlap));	% number of windows
% Moment-based spectra
index = 1:nwind;
f = (0:Nfft-1)/Nfft;
M4 = 0;
M2 = 0;
for i=1:k
    xw = Window.*x(index);
    Xw = fft(xw,Nfft);		        
    M4 = abs(Xw).^4 + M4;   
    M2 = abs(Xw).^2 + M2;  
    index = index + (nwind - Noverlap);
end
% normalize
M4 = M4/k;   
M2 = M2/k; 
% spectral kurtosis 
SK = M4./M2.^2 - 2;
% reduce bias near f = 0 mod(1/2)
W = abs(fft(Window.^2,Nfft)).^2;
Wb = zeros(Nfft,1);
for i = 0:Nfft-1,
   Wb(1+i) = W(1+mod(2*i,Nfft))/W(1);
end;
SK = SK - Wb;
if (nargout == 0),
   figure,newplot;
   subplot(211),plot(f(1:Nfft/2),M2(1:Nfft/2)),grid on,
   xlabel('Normalized frequency'),xlim([f(1) f(Nfft/2)]),title('Power spectrum')
   subplot(212),plot(f(1:Nfft/2),SK(1:Nfft/2)),grid on
   xlabel('Normalized frequency'),xlim([f(1) f(Nfft/2)]),title('Spectral Kurtosis')
end
