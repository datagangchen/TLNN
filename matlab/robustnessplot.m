function   robustnessplot()
close all;
clear all;
t=-100:40:1000;
t=t./250;

x1 = 8*t-3*t.^2+0.1*(t-5).^3-25;
x1=x1/20;
x2= -0.4*cos(0.5*pi.*t+6)-2;

x3= -cos(0.3*pi.*t+2)-0.5;

x4= 4.2*t-0.4*t.^2-0.1*(t-.01).^3-9;
x4=x4/5;
x1=-x1;
x2=-x2;
x3=-x3;
x4=-x4;
f1=figure;

plot(t,x1,'-.b','LineWidth',2)
hold on
plot(t,x2,':r','LineWidth',2)
plot(t,x3,'g','LineWidth',2)
plot(t,x4,'--m','LineWidth',2)
xlim([min(t),5])
ylim([-1.5,3])
hold off
[mx1, index1]=min(x1);
[mx2, index2]=min(x2);
[mx3, index3]=min(x3);
[mx4, index4]=min(x4);

ax=zeros(4,2);
ay=ax;
pos = get(gca, 'Position');
for index =1:4
mx =['mx',num2str(index)];
ind=['index',num2str(index)];
mxx=eval(mx);
indx=eval(ind);    
ax(index,:)=([ t(indx), t(indx)]+abs(min(xlim)))/diff(xlim)*pos(3)+pos(1);
ay(index,:)=([ mxx, 0]-min(ylim))/diff(ylim)*pos(4)+pos(2);
str=['$\rho(x_{',num2str(index),'}',', \varphi )$'];
a(index)=annotation(f1,'textarrow',ax(index,:),ay(index,:),'String',str,'Interpreter','latex','Color','black','FontSize',20,'LineWidth',2);
a(index).FontSize = 20;
end
axis= gca;
axis.Box = 'off';
axis.YAxisLocation = 'origin';
axis.XAxisLocation = 'origin';
%% create text box

for index =1:4
xx=['x',num2str(index)];
x=eval(xx);
y=t(end);
ax =([ y+0.1, y]+abs(min(xlim)))/diff(xlim)*pos(3)+pos(1);
ay=([ x(end)-0.5, x(end)+0.1]-min(ylim))/diff(ylim)*pos(4)+pos(2);
dim=[ax(1),ay(1),0.1,0.1];
str=['x_{',num2str(index),'}'];
a(index)=annotation('textbox',dim,'String',str,'FitBoxToText','on','EdgeColor','white','Color','k','FontSize',20,'fontname','times');
end
str=['$\varphi =Always(x(f)>0)$'];
text(1,-1.3,str,'Interpreter','latex','FontSize',16)
xlabel('Frequency (kHz)')
ylabel('Amplitude')
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [3 3 7 6]);
set(gcf, 'Alphamap',0.01);
set(gcf, 'Colormap', cool);
set(gcf,'Units', 'inches');
set(gcf,'Position',[2, 2, 6, 5]);
set(gcf,'OuterPosition',[1,1,7,6])
set(gcf,'Color','white')
set(gca,'LineWidth',1)
set(gca,'FontUnits','points')
set(gca,'FontSize',16);
set(gca,'fontname','times')
set(gca,'Color','none');
set(gca,'Box','off');
end