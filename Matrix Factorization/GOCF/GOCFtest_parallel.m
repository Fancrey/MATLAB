tic
%Clustering on TDT2
clear;
load('TDT2_all.mat');
%load('COIL20.mat');
%Download random cluster index file 10Class.rar and uncompress.

%tfidf weighting and normalization 
fea = tfidf(fea);

%--------------------------------------
nTotal = 10;
a=cell(1,10);b=cell(10,1);
ACGOCF_TOTAL=zeros(5,5);
MIGOCF_TOTAL=zeros(5,5);
ACGOCF=zeros(1,10);
MI_GOCF=zeros(1,10);
for i = 1:10
  load(['10Class/',num2str(i),'.mat']);
  a{1,i}=sampleIdx;
  b{i,1}=zeroIdx;
end
sampleIdx=a;
zeroIdx=b;
clear a b i
parpool('local',10)
for i = 1:5
for j = 1:5
spmd
  feaSet =  fea(sampleIdx{labindex},:);
  feaSet(:,zeroIdx{labindex,:})=[];
  gndSet = gnd(sampleIdx{labindex});
  nClass = length(unique(gndSet));
  
  rng('default');
  GOCFoptions = [];
  GOCFoptions.WeightMode = 'Cosine';
  GOCFoptions.bNormalized = 1;
  W = constructW(feaSet,GOCFoptions);               %这里的W是论文中的Sij权重矩阵
  
  GOCFoptions.maxIter = 200;
  GOCFoptions.alpha = 10^i;

  GOCFoptions.miu   = 10^j;                         %改进的mu项
  GOCFoptions.KernelType = 'Linear';
  GOCFoptions.weight = 'NCW';
  rng('default');
  [U,V] = GOCF(feaSet',nClass,W,GOCFoptions); 
  
  rng('default');
  label = kmeans(V,nClass,'EmptyAction','singleton','Start','cluster','Replicates',10);
  label = bestMap(gndSet,label);
  MIhat_GOCF = MutualInfo(gndSet,label);
  AC_GOCF= length(find(gndSet == label))/length(gndSet);
  %disp([num2str(labindex),' subset done.']);
end
for k=1:10
    ACGOCF(k)=AC_GOCF{k};
    MI_GOCF(k)=MIhat_GOCF{k};
end
ACGOCF_TOTAL(i,j)=mean(ACGOCF);
MIGOCF_TOTAL(i,j)=mean(MI_GOCF);
disp(['Clustering in the GOCF Lamda=',num2str(10^i),' Mu=',num2str(10^j),' space,AC=',num2str(mean(ACGOCF)),' MI=',num2str(mean(MI_GOCF))]);
end
end
%Clustering in the original space. MIhat: 0.65129
%Clustering in the LCCF subspace. MIhat: 0.9156
delete(gcp('nocreate'));
toc