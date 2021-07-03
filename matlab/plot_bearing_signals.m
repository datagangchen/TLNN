dataInner=  load('209.mat');
xInner = dataInner.X209_DE_time;
Fs = 12000;
L = 8196;
time = 0:1/Fs:(L-1)/Fs;
noise = load('noise.mat');
noise = noise.noise;
Inner = xInner(1:L,1)+noise;
Inner = rescale(Inner,-1,1);

figure

set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [2 2 5 5]);
set(gcf, 'Alphamap',0.01);
set(gcf, 'Colormap', cool);
set(gcf,'Units', 'inches');
set(gcf,'Position',[2, 2, 5, 5]);
set(gcf,'OuterPosition',[1,1,5,5])
set(gcf,'Color','white')
set(gca,'LineWidth',1)
set(gca,'FontUnits','points')
set(gca,'FontSize',16);
set(gca,'fontname','times')
set(gca,'Color','none');
set(gca,'Box','on');
subplot(2,1,1)
plot(time, Inner,'-r')
set(gca,'FontUnits','points')
set(gca,'FontSize',16);
set(gca,'fontname','times')
set(gca,'Color','none');
set(gca,'Box','on');
xlabel('Time (Second)')
ylabel('Amplitude')
title('Faulty Signal')
xlim([0,0.67])

dataNormal=  load('100.mat');
xNormal = dataNormal.X100_DE_time;
fsNormal = dataNormal.X100RPM/60;
Normal = xNormal(1:L,1);
Normal  = rescale(Normal,-1,1);
subplot(2,1,2)
plot(time,Normal,'-b')
set(gca,'FontUnits','points')
set(gca,'FontSize',16);
set(gca,'fontname','times')
set(gca,'Color','none');
set(gca,'Box','on');
xlabel('Time (Second)')
ylabel('Amplitude')
xlim([0,0.67])
title('Normal Signal')