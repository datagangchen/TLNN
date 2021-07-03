clear all
outer = load('xOuter.mat');
inner = load('xInner.mat');
ball = load('xBall.mat');
norm = load('xNormal.mat');
data1 = outer.xOuter;
data2 = inner.xInner;
data3 = ball.xBall;
data4 = norm.xNormal;

L = 1024;
Level = 2;
len = L/2^Level;
Fs = 12000/2^Level;
FT=2;
name =[];
for index =1:2^Level
    name = [name; {['x',num2str(index)]}];
end
fpass =150;
M =30;
N =10;

param = zeros(length(name),1);
fault = {'outer','inner','ball', 'norm'};
idx =[1,2,3,4];
for index = 1:4
    trajs=[];
    data = eval(['data',num2str(index)]); 
    for idxm = 1:M
        sig = data(L*(idxm-1)+1:L*idxm);
        X =[];
        T = wpdec(sig,Level,'haar','shannon');
        for num = 2^Level-1:2^(Level+1)-2
          wpc1 = wpcoef(T,num);
          len  = length(wpc1);
          time = 0:1/Fs:(len-1)/Fs;
          [y,t] = tfsmoment(wpc1,time,2);
          y = rescale(y,0,1);
          X = [X,y' ];
        end
     traj.time =0:1/Fs:(length(X)-1)/Fs;
     traj.X = X;
     trajs =[trajs;traj];
        
    end
    ii =idx;
    
    ii(index) =[];
    for jj =1:3
        data = eval(['data',num2str(ii(jj))]); 
        for idxm = 1:N
          sig = data(L*(idxm-1)+1:L*idxm);  
          X =[];
        T = wpdec(sig,Level,'haar','shannon');
        for num = 2^Level-1:2^(Level+1)-2
          wpc1 = wpcoef(T,num);
          len  = length(wpc1);
          time = 0:1/Fs:(len-1)/Fs;
          [y,t] = tfsmoment(wpc1,time,2);
          y = rescale(y,0,1);
          X = [X,y' ];
        end
     traj.time =0:1/Fs:(length(X)-1)/Fs;
     traj.X = X;
     trajs =[trajs;traj];  
        end
        
    end
    
    label = ones(M+3*N,1);
    label(M+1:end)=-1;
    dataname = ['TLNN_train_',fault{index},'.mat'];
    save(dataname, 'trajs', 'name', 'label')       
end
    
%%pre test data
Mt =15;
Nt = 5;
for index = 1:4
    trajs=[];
    data = eval(['data',num2str(index)]); 
    for idxm = M+1:M+Mt
        sig = data(L*(idxm-1)+1:L*idxm);
        X =[];
        T = wpdec(sig,Level,'haar','shannon');
        for num = 2^Level-1:2^(Level+1)-2
          wpc1 = wpcoef(T,num);
          len  = length(wpc1);
          time = 0:1/Fs:(len-1)/Fs;
          [y,t] = tfsmoment(wpc1,time,2);
          y = rescale(y,0,1);
          X = [X,y' ];
        end
     traj.time =0:1/Fs:(length(X)-1)/Fs;
     traj.X = X;
     trajs =[trajs;traj];
        
    end
    ii =idx;
    
    ii(index) =[];
    for jj =1:3
        data = eval(['data',num2str(ii(jj))]); 
        for idxm = M+1:M+Nt
          sig = data(L*(idxm-1)+1:L*idxm);  
          X =[];
            T = wpdec(sig,Level,'haar','shannon');
        for num = 2^Level-1:2^(Level+1)-2
          wpc1 = wpcoef(T,num);
          len  = length(wpc1);
          time = 0:1/Fs:(len-1)/Fs;
          [y,t] = tfsmoment(wpc1,time,2);
          y = rescale(y,0,1);
          X = [X,y' ];
        end
     traj.time =0:1/Fs:(length(X)-1)/Fs;
     traj.X = X;
     trajs =[trajs;traj];
        end
        
    end
    
    label = ones(Mt+3*Nt,1);
    label(Mt+1:end)=-1;
    dataname = ['TLNN_test_',fault{index},'.mat'];
    save(dataname, 'trajs', 'name', 'label')       
end
