tic
%Clustering on TDT2
clear;
%load('TDT2_all.mat');
 load('COIL20.mat');
%Download random cluster index file 10Class.rar and uncompress.
fea = NormalizeFea(fea);

%--------------------------------------
nTotal = 1000;
ACGOCF_TOTAL=zeros(nTotal,10);
MIGOCF_TOTAL=zeros(nTotal,10);
ACKM_TOTAL  =zeros(nTotal,10);
MIKM_TOTAL  =zeros(nTotal,10);
ACLCCF_TOTAL  =zeros(nTotal,10);
MILCCF_TOTAL  =zeros(nTotal,10);
  parpool('local',10)
  nClass = length(unique(gnd));
for j= 1:nTotal
  spmd
  rng('default');
  label = kmeans(fea,nClass,'Distance','cosine','EmptyAction','singleton','Start','cluster','Replicates',10);
  label = bestMap(gnd,label);
  MIhat_KM = MutualInfo(gnd,label);
  AC_KM = length(find(gnd == label))/length(gnd);
  
  LCCFoptions = [];
  LCCFoptions.WeightMode = 'Cosine';
  LCCFoptions.bNormalized = 1;
  W = constructW(fea,LCCFoptions);               %这里的W是论文中的Sij权重矩阵
  
  LCCFoptions.maxIter = 200;
  LCCFoptions.alpha = 100*labindex;
  LCCFoptions.KernelType = 'Linear';
  LCCFoptions.weight = 'NCW';
  rng('default');
  [U,V] = LCCF(fea',nClass,W,LCCFoptions); 
  
  rng('default');
  label = kmeans(V,nClass,'EmptyAction','singleton','Start','cluster','Replicates',10);
  label = bestMap(gnd,label);
  MIhat_LCCF = MutualInfo(gnd,label);
  AC_LCCF = length(find(gnd == label))/length(gnd);
  
  
  GOCFoptions = [];
  GOCFoptions.WeightMode = 'Cosine';
  GOCFoptions.bNormalized = 1;
  W = constructW(fea,GOCFoptions);               %这里的W是论文中的Sij权重矩阵
  
  GOCFoptions.maxIter = 200;
  GOCFoptions.alpha = 100*labindex;
  GOCFoptions.miu   = 100*j;                          %改进的mu项
  GOCFoptions.KernelType = 'Linear';
  GOCFoptions.weight = 'NCW';
  rng('default');
  [U,V] = GOCF(fea',nClass,W,GOCFoptions); %'
  
  rng('default');
  label = kmeans(V,nClass,'EmptyAction','singleton','Start','cluster','Replicates',10);
  label = bestMap(gnd,label);
  MIhat_GOCF = MutualInfo(gnd,label);
  AC_GOCF = length(find(gnd == label))/length(gnd);
    end
    for k=1:10
    ACKM_TOTAL(j,k)=AC_KM{k};
    MIKM_TOTAL(j,k)=MIhat_KM{k};
    ACLCCF_TOTAL(j,k)=AC_LCCF{k};
    MILCCF_TOTAL(j,k)=MIhat_LCCF{k};
    ACGOCF_TOTAL(j,k)=AC_GOCF{k};
    MIGOCF_TOTAL(j,k)=MIhat_GOCF{k};
    end
end
delete(gcp('nocreate'))
toc