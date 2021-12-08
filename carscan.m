clc;clear;
image=imread("./input.jpeg");
% image=imread("./test1.jpg");
output=rgb2gray(image);%灰度化
output=edge(output,'roberts');%用roberts算子进行边缘检测
output=imerode(output,ones(4,1));%此腐蚀可将非车牌区域的噪声信息腐蚀掉

se1=strel('rectangle',[25,25]);%方形闭环算子
output=imclose(output,se1);%图像开操作

output=imerode(output,ones(1,50));
output=imdilate(output,ones(4,1));
output=imdilate(output,ones(1,50));

output=bwareaopen(output,1500);%将连通域面积小于1500像素的区域都删除，此方法是为了把除车牌以外的区域都删除
% figure,imshow(output);
[y,x]=size(output);
output=double(output);

Y=zeros(y,1);
for i=1:y%统计每一行的像素值为1的个数
    for j=1:x
        if(output(i,j,1)==1)
            Y(i,1)=Y(i,1)+1;
        end
    end
end
% figure,plot(1:y,Y);
[temp,MaxY]=max(Y);%temp为Y1的最大值，MaxY为其所在的行数

Y1=MaxY;
while(Y(Y1,1)>temp/2)%求车牌上边界
    Y1=Y1-1;
end

Y2=MaxY;
while(Y(Y2,1)>temp/2)%求车牌下边界
    Y2=Y2+1;
end

X=zeros(1,x);
for j=1:x%统计每一列的像素值为1的个数，只统计车牌上下边界之间的像素数
    for i=Y1:Y2
        if(output(i,j,1)==1)
            X(1,j)=X(1,j)+1;
        end
    end
end
% figure,plot(1:x,X);
[temp,MaxX]=max(X);

X1=MaxX;
while(X(1,X1)>temp/2)%求车牌左边界
    X1=X1-1;
end
X2=MaxX;
while(X(1,X2)>temp/2)%求车牌右边界
    X2=X2+1;
end

output=image(Y1-2:Y2+2,X1+2:X2-2,:);%求得车牌区域
% figure,imshow(output);

output=rgb2gray(output);%灰度化
output=imbinarize(output);%二值化

[y,x]=size(output);
Count=0;%统计所截取的车牌区域中1的数量
for i=1:x
    for j=1:y
       if(output(j,i)==1)
            Count=Count+1;
        end       
    end
end
C=0;
if(Count<y*x/2)%如果图像中1的数量为少数，认为字符为黑色，背景为白色
    C=1;
end

A=zeros(1,x);
for i=1:x
    for j=1:y
        if(output(j,i)==C)%统计每列与字符颜色相同的像素的数量
            A(1,i)=A(1,i)+1;
        end
    end
end
% figure,stem(A);

s=0;e=0;
p1=0;p2=0;
id=0;%为切分图像编号
path="./discern/outputs/";%切分图像保存路径
for i=1:x
    if(A(1,i)<floor(y/10))%认为如果该列统计值少于每列像素总数的1/10则是分割区域的起始点        
        if(s==0)
            s=i;%记录起始点
        end
    else%若不是分割区域
        if(s~=0)%如果已经记录的起始点
            e=i;%记录分割区域结束点
            if(p1==0)%如果没有记录分割点1
                p1=floor((s+e)/2);%生成分割点1
                s=0;e=0;%清除s和e，重新寻找下一个分割点
                if(p2~=0)%如果已记录分割点2
%                     figure,imshow(output(:,p2:p1,:));
                    imwrite(block(output(:,p2:p1,:)),path+int2str(id)+".jpg");
                    id=id+1;
                end                              
            else%如果已记录分割点1
                p2=floor((s+e)/2);%生成分割点2
                s=0;e=0;
%                 figure,imshow(output(:,p1:p2,:));
                imwrite(block(output(:,p1:p2,:)),path+int2str(id)+".jpg");
                id=id+1;
                p1=0;
            end
        end
    end
end
%由于最后一个分割点没法确定，默认最后一个分割点为区域最后一列位置
if(p1<p2)
    p1=(x);
%     figure,imshow(output(:,p2:p1,:));
    imwrite(block(output(:,p2:p1,:)),path+int2str(id)+".jpg");
else
    p2=(x);
%     figure,imshow(output(:,p1:p2,:));
    imwrite(block(output(:,p1:p2,:)),path+int2str(id)+".jpg");
end

            