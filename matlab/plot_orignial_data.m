clear all
close all 
outer = load('TLNN_train_outer.mat');
inner = load('TLNN_train_inner.mat');
ball =  load('TLNN_train_ball.mat');
norm =  load('TLNN_train_norm.mat');
 
xInner = load('xInner.mat');
xinner = xInner.xInner;

L = 2048;
Fs = 12000;
sig = xinner(1:L);
level =3;
[~, ~, ~, fc, ~, BW] = kurtogram(sig, Fs, level);
 
bpf = designfilt('bandpassfir', 'FilterOrder', 200, 'CutoffFrequency1', fc-BW/2, ...
    'CutoffFrequency2', fc+BW/2, 'SampleRate', Fs);
xOuterBpf = filter(bpf, sig);
[pEnvOuterBpf, fEnvOuterBpf, xEnvOuterBpf, tEnvBpfOuter] = envspectrum(sig, Fs, ...
    'FilterOrder', 200, 'Band', [fc-BW/2 fc+BW/2]);

figure
subplot(2,1,1)
t = 0:1/Fs:(L-1)/Fs;
sig = xinner(1:L);
plot(t,sig)
xlabel('Time (s)')
ylabel('Amp')
xlim([0,max(t)])

set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [3 3 5 5]);
set(gcf, 'Alphamap',0.01);
set(gcf, 'Colormap', cool);
set(gcf,'Units', 'inches');
set(gcf,'Position',[4, 4, 5, 5]);
set(gcf,'OuterPosition',[3.5,3.5,5,5])
set(gcf,'Color','white')
set(gca,'LineWidth',1)
% set(gca,'FontUnits','points')
set(gca,'FontSize',12);
set(gca,'Color','none');
set(gca,'Box','on');

subplot(2,1,2)
xEnvOuterBpf=rescale(xEnvOuterBpf,0,1);
plot(t, xEnvOuterBpf,'r','linewidth',1)
xlim([0,max(t)])
xlabel('Time (s)')
ylabel('Amp')

set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [3 3 5 5]);
set(gcf, 'Alphamap',0.01);
set(gcf, 'Colormap', cool);
set(gcf,'Units', 'inches');
set(gcf,'Position',[4, 4, 5, 5]);
set(gcf,'OuterPosition',[3.5,3.5,5,5])
set(gcf,'Color','white')
set(gca,'LineWidth',1)
% set(gca,'FontUnits','points')
set(gca,'FontSize',12);
set(gca,'Color','none');
set(gca,'Box','on');