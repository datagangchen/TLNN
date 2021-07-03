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
Level = 1;
len = L/2^Level;
Fs = 12000/2^Level;
 
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
          [SK,M4,M2,f] = SK_W(wpc1,length(wpc1),10,20);
          y = M4(1:length(wpc1)/2);
          
            if isempty(X)
                  X = [X,y']; 
              else 
                 X = [X,flip(y',2)];
              end
        end
      X = rescale(X,0,1);  
     %X = X./150;
     X = X(1:4:end);
     len = length(X);
     time = 0:1:(len-1);
     traj.time = time;
     traj.X = X;
     traj.param = param;
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
          [SK,M4,M2,f] = SK_W(wpc1,length(wpc1),10,20);
          y = M4(1:length(wpc1)/2);
             if isempty(X)
                  X = [X,y']; 
              else 
                 X = [X,flip(y',2)];
              end
        end
         X = rescale(X,0,1);  
      %X = X./150;
       X = X(1:4:end);
        len = length(X);
         time = 0:1:(len-1);
         traj.time =time;
         traj.X = X; 
         traj.param = param;
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
          [SK,M4,M2,f] = SK_W(wpc1,length(wpc1),10,20);
          y = M4(1:length(wpc1)/2);
 
             if isempty(X)
                  X = [X,y']; 
              else 
                 X = [X,flip(y',2)];
              end
        end
        X = rescale(X,0,1);  
         %X = X./150;
     X = X(1:4:end);
     len = length(X);
     time = 0:1:(len-1);
     traj.time = time;
     traj.X = X;
     traj.param = param;
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
              [SK,M4,M2,f] = SK_W(wpc1,length(wpc1),10,20);
              y = M4(1:length(wpc1)/2);
              
              if isempty(X)
                  X = [X,y']; 
              else 
                 X = [X,flip(y',2)];
              end
            end
            X = rescale(X,0,1);  
            % X = X./150;
         X = X(1:4:end);
         
         len = length(X);
         time = 0:1:(len-1); 
         traj.time = time;
         traj.X = X;
         traj.param = param;
         trajs =[trajs;traj];   
        end
        
    end
    
    label = ones(Mt+3*Nt,1);
    label(Mt+1:end)=-1;
    dataname = ['TLNN_test_',fault{index},'.mat'];
    save(dataname, 'trajs', 'name', 'label')       
end
