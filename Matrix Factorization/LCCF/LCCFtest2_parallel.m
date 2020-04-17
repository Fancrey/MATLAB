tic
%Clustering on TDT2
clear;
load('TDT2_all.mat');
% load('COIL20.mat');
%Download random cluster index file 10Class.rar and uncompress.

%tfidf weighting and normalization 
fea = tfidf(fea);

%--------------------------------------
nTotal = 10;
a=cell(1,10);b=cell(10,1);
for i = 1:10
  load(['10Class/',num2str(i),'.mat']);
  a{1,i}=sampleIdx;
  b{i,1}=zeroIdx;
end
sampleIdx=a;
zeroIdx=b;
clear a b i
parpool('local',10)
spmd
  feaSet =  fea(sampleIdx{labindex},:);
  feaSet(:,zeroIdx{labindex,:})=[];
  gndSet = gnd(sampleIdx{labindex});
  nClass = length(unique(gndSet));
  
  rand('twister',5489);
  label = kmeans(feaSet,nClass,'Distance','cosine','EmptyAction','singleton','Start','cluster','Replicates',10);
  label = bestMap(gndSet,label);
  MIhat_KM = MutualInfo(gndSet,label);
  AC_KM = length(find(gndSet == label))/length(gndSet);
  
  
  LCCFoptions = [];
  LCCFoptions.WeightMode = 'Cosine';
  LCCFoptions.bNormalized = 1;
  W = constructW(feaSet,LCCFoptions);               %这里的W是论文中的Sij权重矩阵
  
  LCCFoptions.maxIter = 200;
  LCCFoptions.alpha = 1000;
  LCCFoptions.KernelType = 'Linear';
  LCCFoptions.weight = 'NCW';
  rand('twister',5489);
  [U,V] = LCCF(feaSet',nClass,W,LCCFoptions); %'
  
  rand('twister',5489);
  label = kmeans(V,nClass,'EmptyAction','singleton','Start','cluster','Replicates',10);
  label = bestMap(gndSet,label);
  MIhat_LCCF = MutualInfo(gndSet,label);
  AC_LCCF = length(find(gndSet == label))/length(gndSet);
  
  
  GOCFoptions = [];
  GOCFoptions.WeightMode = 'Cosine';
  GOCFoptions.bNormalized = 1;
  W = constructW(feaSet,GOCFoptions);               %这里的W是论文中的Sij权重矩阵
  
  GOCFoptions.maxIter = 200;
  GOCFoptions.alpha = 1000;
  GOCFoptions.miu   = 1000;                          %改进的mu项
  GOCFoptions.KernelType = 'Linear';
  GOCFoptions.weight = 'NCW';
  rand('twister',5489);
  [U,V] = GOCF(feaSet',nClass,W,GOCFoptions); %'
  
  rand('twister',5489);
  label = kmeans(V,nClass,'EmptyAction','singleton','Start','cluster','Replicates',10);
  label = bestMap(gndSet,label);
  MIhat_GOCF = MutualInfo(gndSet,label);
  AC_GOCF= length(find(gndSet == label))/length(gndSet);
  disp([num2str(labindex),' subset done.']);
end

for i=1:10
    ACLCCF(i)=AC_LCCF{i};
    ACKM(i)=AC_KM{i};
    ACGOCF(i)=AC_GOCF{i};
    MI_KM(i)=MIhat_KM{i};
    MI_LCCF(i)=MIhat_LCCF{i};
    MI_GOCF(i)=MIhat_GOCF{i};
end
disp(['Clustering in the original subspace. AC=',num2str(mean(ACKM)),' MI=',num2str(mean(MI_KM))]);
disp(['Clustering in the LCCF subspace. AC=',num2str(mean(ACLCCF)),' MI=',num2str(mean(MI_LCCF))]);
disp(['Clustering in the GOCF subspace. AC=',num2str(mean(ACGOCF)),' MI=',num2str(mean(MI_GOCF))]);
%Clustering in the original space. MIhat: 0.65129
%Clustering in the LCCF subspace. MIhat: 0.9156
delete(gcp('nocreate'));
toc



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %Clustering on TDT2
% clear;
% load('TDT2_all.mat');
% %Download random cluster index file 10Class.rar and uncompress.
% 
% %tfidf weighting and normalization 
% fea = tfidf(fea);
% 
% %--------------------------------------
% nTotal = 50;
% MIhat_KM = zeros(nTotal,1);
% MIhat_LCCF = zeros(nTotal,1);
% for i = 1:nTotal
%   load(['top56/10Class/',num2str(i),'.mat']);
%   feaSet = fea(sampleIdx,:);
%   feaSet(:,zeroIdx)=[];
%   gndSet = gnd(sampleIdx);
%   nClass = length(unique(gndSet));
%   
%   rand('twister',5489);
%   label = kmeans(feaSet,nClass,'Distance','cosine','EmptyAction','singleton','Start','cluster','Replicates',10);
%   label = bestMap(gndSet,label);
%   MIhat_KM(i) = MutualInfo(gndSet,label);
%   
%   LCCFoptions = [];
%   LCCFoptions.WeightMode = 'Cosine';
%   LCCFoptions.bNormalized = 1;
%   W = constructW(feaSet,LCCFoptions);
%   
%   LCCFoptions.maxIter = 200;
%   LCCFoptions.alpha = 100;
%   LCCFoptions.KernelType = 'Linear';
%   LCCFoptions.weight = 'NCW';
%   rand('twister',5489);
%   [U,V] = LCCF(feaSet',nClass,W,LCCFoptions); %'
%   
%   rand('twister',5489);
%   label = kmeans(V,nClass,'EmptyAction','singleton','Start','cluster','Replicates',10);
%   label = bestMap(gndSet,label);
%   MIhat_LCCF(i) = MutualInfo(gndSet,label);
%   disp([num2str(i),' subset done.']);
% end
% disp(['Clustering in the original space. MIhat: ',num2str(mean(MIhat_KM))]);
% disp(['Clustering in the LCCF subspace. MIhat: ',num2str(mean(MIhat_LCCF))]);
% 
% %Clustering in the original space. MIhat: 0.65129
% %Clustering in the LCCF subspace. MIhat: 0.9156

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%










% %%%%%%%%%  ATVNMFtest
% 
% %Clustering on COIL20
% 
% 
% load('COIL20.mat');	
% % load('COIL20.mat');	
% % load('ORL_32x32.mat');
% % load('PIE_pose27');
% % load('Yale_64x64');
% % load('Yale_32x32');
% 
% % fea=imnoise(fea,'gaussian',0,0.006);
% 
% 
% % load('Yale_32x32.mat');
% nClass = length(unique(gnd));  %%%% 找出不同数的种类数（个数）
% 
% % nClass = 10;  %%%% 找出不同数的种类数（个数）
% %Normalize each data vector to have L2-norm equal to 1  
% % fea = NormalizeFea(fea);
% 
% % %%%%%  original
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % %Clustering in the original space
% % % rand('twister',5489);
% % % label = litekmeans(fea,nClass,'Replicates',30);
% % % MIhat = MutualInfo(gnd,label);
% % % disp(['Clustering in the original space. MIhat: ',num2str(MIhat)]);
% % % %Clustering in the original space. MIhat: 0.7386
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% %%%%%%  NMF 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %NMF learning
% % options = [];
% % options.WeightMode = 'Binary';  
% % % W = constructW(fea,options);
% % % options.maxIter = 100;
% % options.maxIter =50;
% % options.alpha = 0;
% % %when alpha = 0, GNMF boils down to the ordinary NMF.
% % % rand('twister',5489);
% % [U,V] = GNMF(fea',nClass,[],options); %'
% % % [U,V] = GNMF(fea',nClass,W,options); %'
% % %Clustering in the NMF subspace
% % % rand('twister',5489);
% % label = litekmeans(V,nClass,'Replicates',40);
% % 
% % label = bestMap(gnd,label);
% % AC = length(find(gnd == label))/length(gnd);
% % 
% % MIhat = MutualInfo(gnd,label);
% % % disp(['Clustering in the NMF subspace. MIhat: ',num2str(MIhat)]);
% % disp(['Clustering in the NMF subspace. AC=',num2str(AC),' NMI=',num2str(MIhat)]);
% % 
% % % Clustering in the NMF subspace. MIhat:  0.74361
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% % %%%%%  GNMF 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % %%%% GNMF learning
% % 
% % %%%  nClass 相当于是非负矩阵分解的 r
% % %%%  fea 相当于是非负矩阵分解的原始矩阵 X
% options = [];
% % options.WeightMode = 'HeatKernel';  
% options.WeightMode = 'Binary';  
% W = constructW(fea,options);
% options.maxIter = 100;
% % options.maxIter = 50;
% options.alpha = 100;
% [U,V] = GNMF(fea',nClass,W,options); %'
% 
% % a=norm(fea'-U*V')
% %Clustering in the GNMF subspace
% % rand('twister',5489);
% label = litekmeans(V,nClass,'Replicates',20);
% 
% label = bestMap(gnd,label);
% AC = length(find(gnd == label))/length(gnd);
% 
% MIhat = MutualInfo(gnd,label);
% % disp(['Clustering in the GNMF subspace. MIhat: ',num2str(MIhat)]);
% disp(['Clustering in the GNMF subspace. AC=',num2str(AC),' NMI=',num2str(MIhat)]);
% %Clustering in the GNMF subspace. MIhat: 0.87599
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %%%%%  DNMF 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % DNMF learning
% % 
% % %%%  nClass 相当于是非负矩阵分解的 r
% % %%%  fea 相当于是非负矩阵分解的原始矩阵 X
% % % beta=0.001;
% % options = [];
% options.WeightMode = 'Binary';  
% W = constructW(fea,options);
% options.maxIter = 100;
% % options.maxIter = 50;
% options.alpha = 100;
% [U,V] = DNMF(fea',nClass,W,options); %'
% 
% %Clustering in the DNMF subspace 
% % rand('twister',5489);
% label = litekmeans(V,nClass,'Replicates',20);
% 
% label = bestMap(gnd,label);
% AC = length(find(gnd == label))/length(gnd);
% 
% MIhat = MutualInfo(gnd,label);
% disp(['Clustering in the DNMF subspace. AC=',num2str(AC),' NMI=',num2str(MIhat)]);
% % disp(['Clustering in the GTVNMF subspace. MIhat: ',num2str(MIhat)]);
% %Clustering in the GNMF subspace. MIhat: 0.87599
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% %%%%%  GSNMF 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % GSNMF learning
% 
% %%%  nClass 相当于是非负矩阵分解的 r
% %%%  fea 相当于是非负矩阵分解的原始矩阵 X
% % beta=0.001;
% % options = [];
% options.WeightMode = 'Binary';  
% W = constructW(fea,options);
% options.maxIter = 100;
% % options.maxIter = 50;
% options.alpha = 100;
% [U,V] = GSNMF(fea',nClass,W,options); %'
% 
% % b=norm(fea'-U*V')
% 
% % Clustering in the GSNMF subspace 
% % rand('twister',5489);
% label = litekmeans(V,nClass,'Replicates',20);
% 
% label = bestMap(gnd,label);
% AC = length(find(gnd == label))/length(gnd);
% 
% MIhat = MutualInfo(gnd,label);
% disp(['Clustering in the GSNMF subspace. AC=',num2str(AC),' NMI=',num2str(MIhat)]);
% % disp(['Clustering in the GSNMF subspace. MIhat: ',num2str(MIhat)]);
% % Clustering in the GNMF subspace. MIhat: 0.87599
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % 
% 
% 
% 
% 
% 
% 
% 
% % % %%%%%  ATVNMF 
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % ATVNMF learning
% % % 
% % % %%%  nClass 相当于是非负矩阵分解的 r
% % % %%%  fea 相当于是非负矩阵分解的原始矩阵 X
% % % % beta=0.001;
% % options = [];
% % options.WeightMode = 'Binary';  
% % W = constructW(fea,options);
% % % options.maxIter = 100;
% % options.maxIter = 100;
% % options.alpha = 100;
% % [U,V] = ATVNMF(fea',nClass,W,options); %'
% % 
% % b=norm(fea'-U*V')
% % 
% % %Clustering in the ATVNMF subspace 
% % % rand('twister',5489);
% % label = litekmeans(V,nClass,'Replicates',20);
% % 
% % label = bestMap(gnd,label);
% % AC = length(find(gnd == label))/length(gnd);
% % 
% % MIhat = MutualInfo(gnd,label);
% % disp(['Clustering in the ATVNMF subspace. AC=',num2str(AC),' NMI=',num2str(MIhat)]);
% % % disp(['Clustering in the GTVNMF subspace. MIhat: ',num2str(MIhat)]);
% % %Clustering in the GNMF subspace. MIhat: 0.87599
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% 
% 
% 
% 
% % fea = NormalizeFea(fea);
% % 
% % %kmeans clustering in the original space
% % % rand('twister',5489);
% % label = litekmeans(fea,nClass,'Replicates',12,'Distance','cosine');
% % label = bestMap(gnd,label);
% % AC = length(find(gnd == label))/length(gnd);
% % MIhat = MutualInfo(gnd,label)
% % disp(['Clustering in the kmeans clustering subspace. AC=',num2str(AC),' NMI=',num2str(MIhat)]);
% % %MIhat: 0.3941
% % 
% % %kmeans in the Laplacian Eigenmap subspace (Spectral Clustering)
% % % rand('twister',5489);
% % W = constructW(fea);
% % Y = Eigenmap(W,nClass);
% % % rand('twister',5489);
% % labelNew = litekmeans(Y,20,'Replicates',12,'Distance','cosine');
% % 
% % labelNew = bestMap(gnd,labelNew);
% % AC = length(find(gnd == labelNew))/length(gnd);
% % MIhatNew = MutualInfo(gnd,labelNew)
% % disp(['Clustering in the Spectral Clustering subspace. AC=',num2str(AC),' NMI=',num2str(MIhat)]);
% % %MIhat: 0.6899
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% % % Clustering using landmark-based spectral clustering
% % % rand('twister',5489) 
% % res = LSC(fea, 8);
% % %Elapsed time is 20.865842 seconds.
% % res = bestMap(gnd,res);
% % AC = length(find(gnd == res))/length(gnd);
% % MIhat = MutualInfo(gnd,res);
% % %AC: 0.7270
% % %MIhat:  0.7222
% % disp(['Clustering using landmark-based spectral clustering. AC=',num2str(AC),' NMI=',num2str(MIhat)]);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % label = litekmeans(Pz_d_LTM',nClass,'Replicates',10); %'
% % % label = bestMap(gnd,label);
% % % AC = length(find(gnd == label))/length(gnd);
% % % MIhat = MutualInfo(gnd,label);
% % % disp(['Clustering in the LTM space. AC=',num2str(AC),' NMI=',num2str(MIhat)]);
% % % %Clustering in the LTM space. AC=0.86159 NMI=0.8257
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%% PCA
% 
% % % options.ReducedDim = 64;
% % % eigvector = PCA(fea, options);
% % % newfea = fea*eigvector;
% % % newfea = NormalizeFea(newfea);
% % % %Clustering in 64-dim PCA subspace
% % % rand('twister',5489);
% % % label = litekmeans(newfea,nClass,'Replicates',10);
% % % MIhat = MutualInfo(gnd,label);
% % % disp(['kmeans in the 64-dim PCA subspace. MIhat: ',num2str(MIhat)]);
% % % %kmeans in the 64-dim PCA subspace. MIhat: 0.65946
% 
% 
% %%%%%%%%%% Spectral Clustering
% 
% %Normalize each data vector to have L2-norm equal to 1  
% % fea = NormalizeFea(fea);
% % 
% % %kmeans clustering in the original space
% % rand('twister',5489);
% % label = litekmeans(fea,nClass,'Replicates',10,'Distance','cosine');
% % MIhat = MutualInfo(gnd,label)
% % %MIhat: 0.3941
% % 
% % %kmeans in the Laplacian Eigenmap subspace (Spectral Clustering)
% % rand('twister',5489);
% % W = constructW(fea);
% % Y = Eigenmap(W,nClass);
% % rand('twister',5489);
% % labelNew = litekmeans(Y,68,'Replicates',10,'Distance','cosine');
% % MIhatNew = MutualInfo(gnd,labelNew)
% % %MIhat: 0.6899
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%  原始程序
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %Clustering on COIL20
% % clear;
% % load('COIL20.mat');	
% % % load('PIE_pose27.mat');
% % % load('TDT2.mat');
% % nClass = length(unique(gnd));  %%%% 找出不同数的种类数（个数）
% % 
% % %Normalize each data vector to have L2-norm equal to 1  
% % fea = NormalizeFea(fea);
% % 
% % %Clustering in the original space
% % % rand('twister',5489);
% % label = litekmeans(fea,nClass,'Replicates',20);
% % MIhat = MutualInfo(gnd,label);
% % disp(['Clustering in the original space. MIhat: ',num2str(MIhat)]);
% % %Clustering in the original space. MIhat: 0.7386
% % 
% % %NMF learning
% % options = [];
% % options.maxIter = 100;
% % options.alpha = 0;
% % %when alpha = 0, GNMF boils down to the ordinary NMF.
% % rand('twister',5489);
% % [U,V] = GNMF(fea',nClass,[],options); %'
% % 
% % %Clustering in the NMF subspace
% % rand('twister',5489);
% % label = litekmeans(V,nClass,'Replicates',20);
% % MIhat = MutualInfo(gnd,label);
% % disp(['Clustering in the NMF subspace. MIhat: ',num2str(MIhat)]);
% % %Clustering in the NMF subspace. MIhat:  0.74361
% % 
% % %GNMF learning
% % options = [];
% % options.WeightMode = 'Binary';  
% % W = constructW(fea,options);
% % options.maxIter = 100;
% % options.alpha = 100;
% % rand('twister',5489);
% % [U,V] = GNMF(fea',nClass,W,options); %'
% % 
% % %Clustering in the GNMF subspace
% % rand('twister',5489);
% % label = litekmeans(V,nClass,'Replicates',20);
% % MIhat = MutualInfo(gnd,label);
% % disp(['Clustering in the GNMF subspace. MIhat: ',num2str(MIhat)]);
% % %Clustering in the GNMF subspace. MIhat: 0.87599
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% % U=U(:,1:5);
% % [m n]=size(U);
% % b=zeros(sqrt(m),n);
% % for i=1:n
% %     b(:,32*i-31:32*i)=reshape(U(:,i),32,32);
% % end
% % imshow((b))
% 
% % fea=fea(:,1:5);
% % [m n]=size(fea);
% % b=zeros(sqrt(m),n);
% % for i=1:n
% %     b(:,32*i-31:32*i)=reshape(fea(:,i),32,32);
% % end
% % imshow((b))
% 
% 




%%%%%%% Deng Cai orignal codes 

% 
% %Clustering on TDT2
% clear;
% load('TDT2_all.mat');
% %Download random cluster index file 10Class.rar and uncompress.
% 
% %tfidf weighting and normalization 
% fea = tfidf(fea);
% 
% %--------------------------------------
% nTotal = 50;
% MIhat_KM = zeros(nTotal,1);
% MIhat_LCCF = zeros(nTotal,1);
% for i = 1:nTotal
%   load(['top56/10Class/',num2str(i),'.mat']);
%   feaSet = fea(sampleIdx,:);
%   feaSet(:,zeroIdx)=[];
%   gndSet = gnd(sampleIdx);
%   nClass = length(unique(gndSet));
%   
%   rand('twister',5489);
%   label = kmeans(feaSet,nClass,'Distance','cosine','EmptyAction','singleton','Start','cluster','Replicates',10);
%   label = bestMap(gndSet,label);
%   MIhat_KM(i) = MutualInfo(gndSet,label);
%   
%   LCCFoptions = [];
%   LCCFoptions.WeightMode = 'Cosine';
%   LCCFoptions.bNormalized = 1;
%   W = constructW(feaSet,LCCFoptions);
%   
%   LCCFoptions.maxIter = 200;
%   LCCFoptions.alpha = 100;
%   LCCFoptions.KernelType = 'Linear';
%   LCCFoptions.weight = 'NCW';
%   rand('twister',5489);
%   [U,V] = LCCF(feaSet',nClass,W,LCCFoptions); %'
%   
%   rand('twister',5489);
%   label = kmeans(V,nClass,'EmptyAction','singleton','Start','cluster','Replicates',10);
%   label = bestMap(gndSet,label);
%   MIhat_LCCF(i) = MutualInfo(gndSet,label);
%   disp([num2str(i),' subset done.']);
% end
% disp(['Clustering in the original space. MIhat: ',num2str(mean(MIhat_KM))]);
% disp(['Clustering in the LCCF subspace. MIhat: ',num2str(mean(MIhat_LCCF))]);
% 
% %Clustering in the original space. MIhat: 0.65129
% %Clustering in the LCCF subspace. MIhat: 0.9156
% % 
