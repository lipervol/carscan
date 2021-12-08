function[output]=block(input)
[x,y]=size(input);
if(x>y)%将输出尺寸改变为长和宽中较大值的1.1倍（取整）
    s=floor(1.1*x);
else
    s=floor(1.1*y);
end
xs=floor((s-x)/2);%计算复制起始行
ys=floor((s-y)/2);%计算复制起始列
output=true(s,s);
ii=1;
for i=xs+1:xs+x
    jj=1;
    for j=ys+1:ys+y
       output(i,j)=input(ii,jj);%将原图像数据复制到指定区域
       jj=jj+1;
    end
    ii=ii+1;
end
    
    
