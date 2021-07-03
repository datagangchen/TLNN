
%% generate training data
close all
dataInner=  load('209.mat');
xInner = dataInner.X209_DE_time;
Fs = 12000;
L = 8196;
idx =7;
% noise = 0.6*randn(L,1);
% save noise.mat noise
noise = load('noise.mat');
noise = noise.noise;
noise = 0;
time = 0:1/Fs:(L-1)/Fs;
Inner = xInner(1:L,1)+ noise;
Inner = rescale(Inner,-1,1);

dataNormal=  load('100.mat');

xNormal = dataNormal.X100_DE_time;
fsNormal = dataNormal.X100RPM/60;
Normal = xNormal(1:L,1);
Normal  = rescale(Normal,-1,1);

fsignal = Inner;
nsignal = Normal;
fs = 12000;

T = wpdec(fsignal,3,'haar','shannon');
X =[];
Level =3;
for num = 2^Level-1:2^(Level+1)-2 
  wpc1 = wpcoef(T,num);
  y = pkurtosis(wpc1,fs);
%   y = rescale(y,0,1);
  X = [X;y'];
end

T = wpdec(nsignal,3,'haar','shannon');
N =[];
Level =3;
for num = 2^Level-1:2^(Level+1)-2 
  wpc1 = wpcoef(T,num);
  y = pkurtosis(wpc1,fs);
%   y = rescale(y,0,1);
  N = [N;y'];
end
F = 0:6000/(length(y)-1):6000;



P = [400,1000,2.2, 2800,700,1.7 ];
t0 = 1500; ylimit = 3;
t1 = P(1);t2 = P(2); p1= P(3); p2 = P(4); t3 = P(5); p3 = P(6); 
rect1 = [t0,-1, t2,p1];
rect2 = [2900,p3,t3,ylimit-p3];
rectangle('Position',rect1,'FaceColor',[  0.7616    0.9058    0.6299],'EdgeColor',[  0.7616    0.9058    0.6299])
hold on
rectangle('Position',rect2,'FaceColor',[ 0.9942 0.7583 0.4312],'EdgeColor',[ 0.9942 0.7583 0.4312]);
hold on
 
plot(F, X(idx,:),'-r' , F, N(idx,:),'-.b','linewidth',1.5)
xlabel('Frequency (Hz)')
ylabel('Amp')
legend('Faulty','Normal')
% dim =[0.22,0.56,0.1,0.1];
% annotation('textbox',dim,'String','[1500,1.2]','FitBoxToText','on','EdgeColor','white','Color','k','FontSize',15);
% dim =[0.37,0.56,0.1,0.1];
% annotation('textbox',dim,'String','[2500,1.2]','FitBoxToText','on','EdgeColor','white','Color','k','FontSize',15);

set(gca,'FontName','times')
set(gca, 'FontSize',15)
 
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
 
set(gca,'FontSize',15);
set(gca,'Color','none');
set(gca,'Box','on');