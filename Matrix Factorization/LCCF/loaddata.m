a=cell(1,10);b=cell(10,1);
for i = 1:10
  load(['10Class/',num2str(i),'.mat']);
  a{1,i}=sampleIdx;
  b{i,1}=zeroIdx;
end
clear sampleIdx zeroIdx
sampleIdx=a;
zeroIdx=b;
clear a b i