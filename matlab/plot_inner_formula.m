clear all
close all

dataname = {'TLNN_train_ball.mat','TLNN_train_inner.mat','TLNN_train_norm.mat','TLNN_train_outer.mat'};
fault =2;
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

rect2 = [58,0,8,0.05];
rectangle('Position',rect2,'FaceColor',colorA,'EdgeColor',edgeA)
hold on
rect3 = [14,0.3,17,0.7];
rectangle('Position',rect3,'FaceColor',colorA,'EdgeColor',edgeA)
hold on


%%% formula rectangle 

rect1 = [45,0.04, 7,0.96];
rectangle('Position',rect1,'FaceColor',colorA,'EdgeColor',edgeA)
hold on


for index =1:N
    x =data.trajs(index).X;
    t =  data.trajs(index).time;
    plot(t,x,'-', 'LineWidth',0.8 ,'color',colorP)
    hold on
end

for index =M+1:M+N
    x =data.trajs(index).X;
    t =  data.trajs(index).time;
    plot(t,x,'-.', 'LineWidth',0.8 ,'color', colorN)
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

rect1 = [45,0.04, 7,0.96];
rectangle('Position',rect1,'FaceColor',colorA,'EdgeColor',edgeA)
hold on
 


for index =1:N
        x =data.trajs(index).X;
    t =  data.trajs(index).time;
indexOfInterest = (t < 62) & (t > 44); % range of t near perturbation
plot(ax,t(indexOfInterest),x(indexOfInterest),'-', 'LineWidth',0.8 ,'color',colorP) % plot on new axes
hold on
end

for index =M+1:M+N
        x =data.trajs(index).X;
    t =  data.trajs(index).time;
indexOfInterest = (t < 62) & (t > 44); % range of t near perturbation
plot(ax,t(indexOfInterest),x(indexOfInterest),'-.', 'LineWidth',0.8 ,'color', colorN) % plot on new axes
hold on
end

  

 axis tight  
