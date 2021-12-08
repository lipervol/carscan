clc;clear;
image=imread("./input.jpeg");
% image=imread("./test1.jpg");
output=rgb2gray(image);%�ҶȻ�
output=edge(output,'roberts');%��roberts���ӽ��б�Ե���
output=imerode(output,ones(4,1));%�˸�ʴ�ɽ��ǳ��������������Ϣ��ʴ��

se1=strel('rectangle',[25,25]);%���αջ�����
output=imclose(output,se1);%ͼ�񿪲���

output=imerode(output,ones(1,50));
output=imdilate(output,ones(4,1));
output=imdilate(output,ones(1,50));

output=bwareaopen(output,1500);%����ͨ�����С��1500���ص�����ɾ�����˷�����Ϊ�˰ѳ��������������ɾ��
% figure,imshow(output);
[y,x]=size(output);
output=double(output);

Y=zeros(y,1);
for i=1:y%ͳ��ÿһ�е�����ֵΪ1�ĸ���
    for j=1:x
        if(output(i,j,1)==1)
            Y(i,1)=Y(i,1)+1;
        end
    end
end
% figure,plot(1:y,Y);
[temp,MaxY]=max(Y);%tempΪY1�����ֵ��MaxYΪ�����ڵ�����

Y1=MaxY;
while(Y(Y1,1)>temp/2)%�����ϱ߽�
    Y1=Y1-1;
end

Y2=MaxY;
while(Y(Y2,1)>temp/2)%�����±߽�
    Y2=Y2+1;
end

X=zeros(1,x);
for j=1:x%ͳ��ÿһ�е�����ֵΪ1�ĸ�����ֻͳ�Ƴ������±߽�֮���������
    for i=Y1:Y2
        if(output(i,j,1)==1)
            X(1,j)=X(1,j)+1;
        end
    end
end
% figure,plot(1:x,X);
[temp,MaxX]=max(X);

X1=MaxX;
while(X(1,X1)>temp/2)%������߽�
    X1=X1-1;
end
X2=MaxX;
while(X(1,X2)>temp/2)%�����ұ߽�
    X2=X2+1;
end

output=image(Y1-2:Y2+2,X1+2:X2-2,:);%��ó�������
% figure,imshow(output);

output=rgb2gray(output);%�ҶȻ�
output=imbinarize(output);%��ֵ��

[y,x]=size(output);
Count=0;%ͳ������ȡ�ĳ���������1������
for i=1:x
    for j=1:y
       if(output(j,i)==1)
            Count=Count+1;
        end       
    end
end
C=0;
if(Count<y*x/2)%���ͼ����1������Ϊ��������Ϊ�ַ�Ϊ��ɫ������Ϊ��ɫ
    C=1;
end

A=zeros(1,x);
for i=1:x
    for j=1:y
        if(output(j,i)==C)%ͳ��ÿ�����ַ���ɫ��ͬ�����ص�����
            A(1,i)=A(1,i)+1;
        end
    end
end
% figure,stem(A);

s=0;e=0;
p1=0;p2=0;
id=0;%Ϊ�з�ͼ����
path="./discern/outputs/";%�з�ͼ�񱣴�·��
for i=1:x
    if(A(1,i)<floor(y/10))%��Ϊ�������ͳ��ֵ����ÿ������������1/10���Ƿָ��������ʼ��        
        if(s==0)
            s=i;%��¼��ʼ��
        end
    else%�����Ƿָ�����
        if(s~=0)%����Ѿ���¼����ʼ��
            e=i;%��¼�ָ����������
            if(p1==0)%���û�м�¼�ָ��1
                p1=floor((s+e)/2);%���ɷָ��1
                s=0;e=0;%���s��e������Ѱ����һ���ָ��
                if(p2~=0)%����Ѽ�¼�ָ��2
%                     figure,imshow(output(:,p2:p1,:));
                    imwrite(block(output(:,p2:p1,:)),path+int2str(id)+".jpg");
                    id=id+1;
                end                              
            else%����Ѽ�¼�ָ��1
                p2=floor((s+e)/2);%���ɷָ��2
                s=0;e=0;
%                 figure,imshow(output(:,p1:p2,:));
                imwrite(block(output(:,p1:p2,:)),path+int2str(id)+".jpg");
                id=id+1;
                p1=0;
            end
        end
    end
end
%�������һ���ָ��û��ȷ����Ĭ�����һ���ָ��Ϊ�������һ��λ��
if(p1<p2)
    p1=(x);
%     figure,imshow(output(:,p2:p1,:));
    imwrite(block(output(:,p2:p1,:)),path+int2str(id)+".jpg");
else
    p2=(x);
%     figure,imshow(output(:,p1:p2,:));
    imwrite(block(output(:,p1:p2,:)),path+int2str(id)+".jpg");
end

            