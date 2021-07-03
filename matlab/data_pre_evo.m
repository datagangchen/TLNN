clear all
outer = load('xOuter.mat');
inner = load('xInner.mat');
ball = load('xBall.mat');
norm = load('xNormal.mat');
data1 = outer.xOuter;
data2 = inner.xInner;
data3 = ball.xBall;
data4 = norm.xNormal;

L = 2048;
level =4;
Fs = 12000;


fpass =150;
M =60;
N =20;

fault = {'outer','inner','ball', 'norm'};
idx =[1,2,3,4];
for index = 1:4
    trajs=[];
    data = eval(['data',num2str(index)]); 
    for idxm = 1:M
        sig = data(L*(idxm-1)+1:L*idxm);
        [~, ~, ~, fc, ~, BW] = kurtogram(sig, Fs, level);

        [pEnvOuterBpf, fEnvOuterBpf, xEnvOuterBpf, tEnvBpfOuter] = envspectrum(sig, Fs, ...
            'FilterOrder', 200, 'Band', [0, 2000]);
        X = rescale(xEnvOuterBpf,0,1);
     
     traj.time = tEnvBpfOuter;
     traj.X = [X];
     trajs =[trajs;traj];
        
    end
    ii =idx;
    
    ii(index) =[];
    for jj =1:3
        data = eval(['data',num2str(ii(jj))]); 
        for idxm = 1:N
          sig = data(L*(idxm-1)+1:L*idxm);  
        [~, ~, ~, fc, ~, BW] = kurtogram(sig, Fs, level);
        [pEnvOuterBpf, fEnvOuterBpf, xEnvOuterBpf, tEnvBpfOuter] = envspectrum(sig, Fs, ...
            'FilterOrder', 200, 'Band', [0, 2000]);
        
        
        X = rescale(xEnvOuterBpf,0,1);
     
     traj.time = tEnvBpfOuter;
     traj.X = [X];
     trajs =[trajs;traj];
        end
        
    end
    
    label = ones(M+3*N,1);
    label(M+1:end)=-1;
    dataname = ['TLNN_train_',fault{index},'.mat'];
    save(dataname, 'trajs',  'label')       
end
    
%%pre test data
Mt =15;
Nt = 5;
for index = 1:4
    trajs=[];
    data = eval(['data',num2str(index)]); 
    for idxm = M+1:M+Mt
        sig = data(L*(idxm-1)+1:L*idxm);
        [~, ~, ~, fc, ~, BW] = kurtogram(sig, Fs, level);
        [pEnvOuterBpf, fEnvOuterBpf, xEnvOuterBpf, tEnvBpfOuter] = envspectrum(sig, Fs, ...
            'FilterOrder', 200, 'Band', [0, 2000]);
        
        
        X = rescale(xEnvOuterBpf,0,1);
     
     traj.time = tEnvBpfOuter;
     traj.X = [X];
     trajs =[trajs;traj];
    end
    ii =idx;
    
    ii(index) =[];
    for jj =1:3
  
        data = eval(['data',num2str(ii(jj))]); 
        for idxm = M+1:M+Nt
          sig = data(L*(idxm-1)+1:L*idxm);  
        [~, ~, ~, fc, ~, BW] = kurtogram(sig, Fs, level);
        [pEnvOuterBpf, fEnvOuterBpf, xEnvOuterBpf, tEnvBpfOuter] = envspectrum(sig, Fs, ...
            'FilterOrder', 200, 'Band', [0, 2000]);
        
        
        X = rescale(xEnvOuterBpf,0,1);
     
     traj.time = tEnvBpfOuter;
     traj.X = [X];
     trajs =[trajs;traj];
        end
        
    end
    
    label = ones(Mt+3*Nt,1);
    label(Mt+1:end)=-1;
    dataname = ['TLNN_test_',fault{index},'.mat'];
    save(dataname, 'trajs',  'label')       
end
