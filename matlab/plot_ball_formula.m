clear all
close all

dataname = {'TLNN_train_ball.mat','TLNN_train_inner.mat','TLNN_train_norm.mat','TLNN_train_outer.mat'};
fault =1;
M = 30;
N = 30;
data = load(dataname{fault});

colorR = [ 0.5686    0.9961    0.4980];
edgeR = [ 0.9961    0.9373    0.4980];
colorA = [0.9942 0.7583 0.4312];
edgeA = [  0.7616    0.9058    0.6299];
colorP = [52, 152, 219]/255;
colorN = [255, 0, 31 ]/255;
 
 
%%% plot signals
figure

rect2 = [65,0,8,0.3];
rectangle('Position',rect2,'FaceColor',colorA,'EdgeColor',edgeA)
hold on


%%% formula rectangle 

rect1 = [20,0.1, 30,0.9];
rectangle('Position',rect1,'FaceColor',colorA,'EdgeColor',edgeA)
hold on


for index =1:N
    x =data.trajs(index).X;
    t =  data.trajs(index).time;
    plot(t,x,'-', 'LineWidth',0.8 ,'color',[52, 152, 219]/255)
    hold on
end

for index =M+1:M+N
    x =data.trajs(index).X;
    t =  data.trajs(index).time;
    plot(t,x,'-.', 'LineWidth',0.8 ,'color', [236, 112, 99 ]/255)
    hold on
    
end    
xlim([0,max(t)])
xlabel('Index')
ylabel('Amp')

 set(gcf, 'Alphamap',0.01);
set(gcf, 'Colormap', cool);
set(gcf,'Units', 'inches');
set(gcf,'Position',[2, 2, 5,3.5]);
set(gcf,'OuterPosition',[1,1,5,3.5])
set(gcf,'Color','white')
set(gca,'LineWidth',1)
set(gca,'FontUnits','points')
set(gca,'FontSize',12);
set(gca,'Fontname', 'times')
set(gca,'Color','none');
set(gca,'Box','on');    

 



 



%% zooming 
ax = axes('position',[.65 .675 .25 .25]);
box on % put box around new pair of axes

rect1 = [44,0.1, 6,0.9];
rectangle('Position',rect1,'FaceColor',colorA,'EdgeColor',edgeA)
hold on
 


for index =1:N
        x =data.trajs(index).X;
    t =  data.trajs(index).time;
indexOfInterest = (t < 53) & (t > 43); % range of t near perturbation
plot(ax,t(indexOfInterest),x(indexOfInterest),'-', 'LineWidth',0.8 ,'color',[52, 152, 219]/255) % plot on new axes
hold on
end

for index =M+1:M+N
        x =data.trajs(index).X;
    t =  data.trajs(index).time;
indexOfInterest = (t < 53) & (t > 43); % range of t near perturbation
plot(ax,t(indexOfInterest),x(indexOfInterest),'-.', 'LineWidth',0.8 ,'color', [236, 112, 99 ]/255) % plot on new axes
hold on
end

  

 axis tight  
