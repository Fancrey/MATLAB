tic
%Clustering on TDT2
clear;
%load('TDT2_all.mat');
 load('COIL20.mat');
%Download random cluster index file 10Class.rar and uncompress.
fea = NormalizeFea(fea);

%--------------------------------------
nTotal = 1000; %设置miu
mTotal = 1000;  %设置lamda
MIhat_LCCF=zeros(1,mTotal);
AC_LCCF=zeros(1,mTotal);
MIhat_GOCF=zeros(nTotal,mTotal);
AC_GOCF=zeros(nTotal,mTotal);
nClass = length(unique(gnd));
rng('default');                                       %17-21 kmeans
label = kmeans(fea,nClass,'Distance','cosine','EmptyAction','singleton','Start','cluster','Replicates',10);
label = bestMap(gnd,label);
MIhat_KM = MutualInfo(gnd,label);
AC_KM = length(find(gnd == label))/length(gnd);       %kmeans end
for labindex= 1:mTotal
    LCCFoptions = [];
    LCCFoptions.WeightMode = 'Cosine';
    LCCFoptions.bNormalized = 1;
    W = constructW(fea,LCCFoptions);               %这里的W是论文中的Sij权重矩阵
    
    LCCFoptions.maxIter = 200;
    LCCFoptions.alpha = 100*labindex;              %lamda
    LCCFoptions.KernelType = 'Linear';
    LCCFoptions.weight = 'NCW';
    rng('default');
    [~,V] = LCCF(fea',nClass,W,LCCFoptions);
    
    rng('default');
    label = kmeans(V,nClass,'EmptyAction','singleton','Start','cluster','Replicates',10);
    label = bestMap(gnd,label);
    MIhat_LCCF(labindex) = MutualInfo(gnd,label);
    AC_LCCF(labindex) = length(find(gnd == label))/length(gnd);
      
    parfor j=1:nTotal
        GOCFoptions = [];
        GOCFoptions.WeightMode = 'Cosine';
        GOCFoptions.bNormalized = 1;
        W = constructW(fea,GOCFoptions);               %这里的W是论文中的Sij权重矩阵
        
        GOCFoptions.maxIter = 200;
        GOCFoptions.alpha = 100*labindex;                   %lamda
        GOCFoptions.miu   = 100*j;                          %改进的mu项
        GOCFoptions.KernelType = 'Linear';
        GOCFoptions.weight = 'NCW';
        rng('default');
        [~,V] = GOCF(fea',nClass,W,GOCFoptions); %'
        
        rng('default');
        label = kmeans(V,nClass,'EmptyAction','singleton','Start','cluster','Replicates',10);
        label = bestMap(gnd,label);
        MIhat_GOCF(j,labindex) = MutualInfo(gnd,label);
        AC_GOCF(j,labindex) = length(find(gnd == label))/length(gnd);
        %disp(['Clustering in the LCCF subspace. AC=',num2str(AC_LCCF(j,labindex)),' MI=',num2str(MIhat_LCCF(j,labindex))]);
        %disp(['Clustering in the GOCF subspace. AC=',num2str(AC_GOCF(j,labindex)),' MI=',num2str(MIhat_GOCF(j,labindex))]);
    end
    disp([num2str(10*labindex),'subset done.']);
end
toc