function[output]=block(input)
[x,y]=size(input);
if(x>y)%������ߴ�ı�Ϊ���Ϳ��нϴ�ֵ��1.1����ȡ����
    s=floor(1.1*x);
else
    s=floor(1.1*y);
end
xs=floor((s-x)/2);%���㸴����ʼ��
ys=floor((s-y)/2);%���㸴����ʼ��
output=true(s,s);
ii=1;
for i=xs+1:xs+x
    jj=1;
    for j=ys+1:ys+y
       output(i,j)=input(ii,jj);%��ԭͼ�����ݸ��Ƶ�ָ������
       jj=jj+1;
    end
    ii=ii+1;
end
    
    
