%%test network
cam1_size=size(label_test_cam1);
cam2_size=size(label_test_cam2);
cam3_size=size(label_test_cam3);
cam1_feature1=[];
cam1_feature2=[];
cam1_feature3=[];
cam1_score=[];
test_batch_data=zeros(224,224,3,param.test_batch_size,'single');
for m=0:floor(cam1_size(1)/param.test_batch_size)-1
    fprintf('process cam1:%d/%d\n',m,floor(cam1_size(1)/param.test_batch_size)-1);
    for n=1:param.test_batch_size
        im_data=imread(test_image_name_cam1{m*param.test_batch_size+n});
        im_data=imresize(single(im_data),[224 224]);
        im_data=im_data(:,:,[3,2,1]);
        im_data=permute(im_data,[2,1,3]);
        test_batch_data(:,:,:,n)=im_data;
        test_batch_data(:,:,1,n)=test_batch_data(:,:,1,n)-104;
        test_batch_data(:,:,2,n)=test_batch_data(:,:,2,n)-117;
        test_batch_data(:,:,3,n)=test_batch_data(:,:,3,n)-123;
    end
    net.blobs('data').set_data(test_batch_data);
    net.forward_prefilled;
    cam1_feature1=[cam1_feature1;(squeeze(net.blobs('data1_p_1').get_data))'];
    cam1_feature2=[cam1_feature2;(squeeze(net.blobs('data1_p_2').get_data))'];
    cam1_feature3=[cam1_feature3;(squeeze(net.blobs('data1_p_3').get_data))'];
    cam1_score=[cam1_score;(squeeze(net.blobs('sig').get_data))'];
end
test_batch_data=zeros(224,224,3,param.test_batch_size,'single');
for m=(floor(cam1_size(1)/param.test_batch_size))*param.test_batch_size+1:cam1_size(1)
    fprintf('process cam1:%d/%d\n',m,cam1_size(1));
        count=m-(floor(cam1_size(1)/param.test_batch_size))*param.test_batch_size;
        im_data=imread(test_image_name_cam1{m});
        im_data=imresize(single(im_data),[224 224]);
        im_data=im_data(:,:,[3,2,1]);
        im_data=permute(im_data,[2,1,3]);
        test_batch_data(:,:,:,count)=im_data;
        test_batch_data(:,:,1,count)=test_batch_data(:,:,1,count)-104;
        test_batch_data(:,:,2,count)=test_batch_data(:,:,2,count)-117;
        test_batch_data(:,:,3,count)=test_batch_data(:,:,3,count)-123;
end
net.blobs('data').set_data(test_batch_data);
net.forward_prefilled;
index=cam1_size(1)-(floor(cam1_size(1)/param.test_batch_size))*param.test_batch_size;
result1=(squeeze(net.blobs('data1_p_1').get_data))';
result2=(squeeze(net.blobs('data1_p_2').get_data))';
result3=(squeeze(net.blobs('data1_p_3').get_data))';
score=(squeeze(net.blobs('sig').get_data))';
cam1_feature1=[cam1_feature1;result1(1:index,:)];
cam1_feature2=[cam1_feature2;result2(1:index,:)];
cam1_feature3=[cam1_feature3;result3(1:index,:)];
cam1_score=[cam1_score;score(1:index,:)];
gallery_feature=[];
gallery_label=[];
for m=1:param.test_person_num
    test_label_index=find(label_test_cam1==(m));
    if isempty(test_label_index)
        continue;
    end
    a=bsxfun(@times,cam1_feature1(test_label_index,:),cam1_score(test_label_index,1));
    b=bsxfun(@times,cam1_feature2(test_label_index,:),cam1_score(test_label_index,2));
    c=bsxfun(@times,cam1_feature3(test_label_index,:),cam1_score(test_label_index,3));
    gallery_feature=[gallery_feature;sum(a/sum(cam1_score(test_label_index,1))),sum(b/sum(cam1_score(test_label_index,2))),sum(c/sum(cam1_score(test_label_index,3)))];
%     gallery_feature=[gallery_feature;sum(a/sum(cam1_score(test_label_index,1)))];
%     gallery_feature=[gallery_feature;sum(b/sum(cam1_score(test_label_index,2)))];
%     gallery_feature=[gallery_feature;sum(c/sum(cam1_score(test_label_index,3)))];
    gallery_label=[gallery_label;m];
end
%extract feature cam3
cam3_feature1=[];
cam3_feature2=[];
cam3_feature3=[];
cam3_score=[];
test_batch_data=zeros(224,224,3,param.test_batch_size,'single');
for m=0:floor(cam3_size(1)/param.test_batch_size)-1
        fprintf('process cam3:%d/%d\n',m,floor(cam3_size(1)/param.test_batch_size)-1);
    for n=1:param.test_batch_size
        im_data=imread(test_image_name_cam3{m*param.test_batch_size+n});
        im_data=imresize(single(im_data),[224 224]);
        im_data=im_data(:,:,[3,2,1]);
        im_data=permute(im_data,[2,1,3]);
        test_batch_data(:,:,:,n)=im_data;
        test_batch_data(:,:,1,n)=test_batch_data(:,:,1,n)-104;
        test_batch_data(:,:,2,n)=test_batch_data(:,:,2,n)-117;
        test_batch_data(:,:,3,n)=test_batch_data(:,:,3,n)-123;
    end
    net.blobs('data').set_data(test_batch_data);
    net.forward_prefilled;
    cam3_feature1=[cam3_feature1;(squeeze(net.blobs('data1_p_1').get_data))'];
    cam3_feature2=[cam3_feature2;(squeeze(net.blobs('data1_p_2').get_data))'];
    cam3_feature3=[cam3_feature3;(squeeze(net.blobs('data1_p_3').get_data))'];
    cam3_score=[cam3_score;(squeeze(net.blobs('sig').get_data))'];
end
test_batch_data=zeros(224,224,3,param.test_batch_size,'single');
for m=(floor(cam3_size(1)/param.test_batch_size))*param.test_batch_size+1:cam3_size(1)
        fprintf('process cam3:%d/%d\n',m,cam3_size(1));
        count=m-(floor(cam3_size(1)/param.test_batch_size))*param.test_batch_size;
        im_data=imread(test_image_name_cam3{m});
        im_data=imresize(single(im_data),[224 224]);
        im_data=im_data(:,:,[3,2,1]);
        im_data=permute(im_data,[2,1,3]);
        test_batch_data(:,:,:,count)=im_data;
        test_batch_data(:,:,1,count)=test_batch_data(:,:,1,count)-104;
        test_batch_data(:,:,2,count)=test_batch_data(:,:,2,count)-117;
        test_batch_data(:,:,3,count)=test_batch_data(:,:,3,count)-123;
end
net.blobs('data').set_data(test_batch_data);
net.forward_prefilled;
index=cam3_size(1)-(floor(cam3_size(1)/param.test_batch_size))*param.test_batch_size;
result1=(squeeze(net.blobs('data1_p_1').get_data))';
result2=(squeeze(net.blobs('data1_p_2').get_data))';
result3=(squeeze(net.blobs('data1_p_3').get_data))';
score=(squeeze(net.blobs('sig').get_data))';
cam3_feature1=[cam3_feature1;result1(1:index,:)];
cam3_feature2=[cam3_feature2;result2(1:index,:)];
cam3_feature3=[cam3_feature3;result3(1:index,:)];
cam3_score=[cam3_score;score(1:index,:)];
for m=1:param.test_person_num
    test_label_index=find(label_test_cam3==(m));
    if isempty(test_label_index)
        continue;
    end
    a=bsxfun(@times,cam3_feature1(test_label_index,:),cam3_score(test_label_index,1));
    b=bsxfun(@times,cam3_feature2(test_label_index,:),cam3_score(test_label_index,2));
    c=bsxfun(@times,cam3_feature3(test_label_index,:),cam3_score(test_label_index,3));
    gallery_feature=[gallery_feature;sum(a/sum(cam3_score(test_label_index,1))),sum(b/sum(cam3_score(test_label_index,2))),sum(c/sum(cam3_score(test_label_index,3)))];
%     gallery_feature=[gallery_feature;sum(a/sum(cam3_score(test_label_index,1)))];
%     gallery_feature=[gallery_feature;sum(b/sum(cam3_score(test_label_index,2)))];
%     gallery_feature=[gallery_feature;sum(c/sum(cam3_score(test_label_index,3)))];
    gallery_label=[gallery_label;m];
end
%extract feature cam2
cam2_feature1=[];
cam2_feature2=[];
cam2_feature3=[];
cam2_score=[];
test_batch_data=zeros(224,224,3,param.test_batch_size,'single');
for m=0:floor(cam2_size(1)/param.test_batch_size)-1
        fprintf('process cam2:%d/%d\n',m,floor(cam2_size(1)/param.test_batch_size)-1);
    for n=1:param.test_batch_size
        im_data=imread(test_image_name_cam2{m*param.test_batch_size+n});
        im_data=imresize(single(im_data),[224 224]);
        im_data=im_data(:,:,[3,2,1]);
        im_data=permute(im_data,[2,1,3]);
        test_batch_data(:,:,:,n)=im_data;
        test_batch_data(:,:,1,n)=test_batch_data(:,:,1,n)-104;
        test_batch_data(:,:,2,n)=test_batch_data(:,:,2,n)-117;
        test_batch_data(:,:,3,n)=test_batch_data(:,:,3,n)-123;
    end
    net.blobs('data').set_data(test_batch_data);
    net.forward_prefilled;
    cam2_feature1=[cam2_feature1;(squeeze(net.blobs('data1_p_1').get_data))'];
    cam2_feature2=[cam2_feature2;(squeeze(net.blobs('data1_p_2').get_data))'];
    cam2_feature3=[cam2_feature3;(squeeze(net.blobs('data1_p_3').get_data))'];
    cam2_score=[cam2_score;(squeeze(net.blobs('sig').get_data))'];
end
test_batch_data=zeros(224,224,3,param.test_batch_size,'single');
for m=(floor(cam2_size(1)/param.test_batch_size))*param.test_batch_size+1:cam2_size(1)
        fprintf('process cam2:%d/%d\n',m,cam2_size(1));
        count=m-(floor(cam2_size(1)/param.test_batch_size))*param.test_batch_size;
        im_data=imread(test_image_name_cam2{m});
        im_data=imresize(single(im_data),[224 224]);
        im_data=im_data(:,:,[3,2,1]);
        im_data=permute(im_data,[2,1,3]);
        test_batch_data(:,:,:,count)=im_data;
        test_batch_data(:,:,1,count)=test_batch_data(:,:,1,count)-104;
        test_batch_data(:,:,2,count)=test_batch_data(:,:,2,count)-117;
        test_batch_data(:,:,3,count)=test_batch_data(:,:,3,count)-123;
end
net.blobs('data').set_data(test_batch_data);
net.forward_prefilled;
index=cam2_size(1)-(floor(cam2_size(1)/param.test_batch_size))*param.test_batch_size;
result1=(squeeze(net.blobs('data1_p_1').get_data))';
result2=(squeeze(net.blobs('data1_p_2').get_data))';
result3=(squeeze(net.blobs('data1_p_3').get_data))';
score=(squeeze(net.blobs('sig').get_data))';
cam2_feature1=[cam2_feature1;result1(1:index,:)];
cam2_feature2=[cam2_feature2;result2(1:index,:)];
cam2_feature3=[cam2_feature3;result3(1:index,:)];
cam2_score=[cam2_score;score(1:index,:)];
prob_feature=[];
prob_label=[];
for m=1:param.test_person_num
    test_label_index=find(label_test_cam2==(m));
    if isempty(test_label_index)
        continue;
    end
    a=bsxfun(@times,cam2_feature1(test_label_index,:),cam2_score(test_label_index,1));
    b=bsxfun(@times,cam2_feature2(test_label_index,:),cam2_score(test_label_index,2));
    c=bsxfun(@times,cam2_feature3(test_label_index,:),cam2_score(test_label_index,3));
    prob_feature=[prob_feature;sum(a/sum(cam2_score(test_label_index,1))),sum(b/sum(cam2_score(test_label_index,2))),sum(c/sum(cam2_score(test_label_index,3)))];

%     prob_feature=[prob_feature;sum(a/sum(cam2_score(test_label_index,1)))];
%     prob_feature=[prob_feature;sum(b/sum(cam2_score(test_label_index,2)))];
%     prob_feature=[prob_feature;sum(c/sum(cam2_score(test_label_index,3)))];
    
    prob_label=[prob_label;m];
end
% cal cmc
prob_norm=bsxfun(@rdivide,prob_feature,sum(abs(prob_feature).^2,2).^(1/2));
gallery_norm=bsxfun(@rdivide,gallery_feature,sum(abs(gallery_feature).^2,2).^(1/2));
score_matrix=prob_norm*gallery_norm';
rank1_hit=0;
rank5_hit=0;
rank10_hit=0;
rank20_hit=0;
problen=size(prob_feature,1);
for m=1:problen
    [~,location_temp]=sort(score_matrix(m,:),2,'descend');
    location=gallery_label(location_temp);
    if find(location(1)==prob_label(m))
        rank1_hit=rank1_hit+1;
    end
    if find(location(1:5)==prob_label(m))
        rank5_hit=rank5_hit+1;
    end
    if find(location(1:10)==prob_label(m))
        rank10_hit=rank10_hit+1;
    end
    if find(location(1:20)==prob_label(m))
        rank20_hit=rank20_hit+1;
    end
end
rank_acc=[rank1_hit/problen,rank5_hit/problen,rank10_hit/problen,rank20_hit/problen];
fin=fopen(param.result_save_file,'a');
fprintf(fin,'up split_index:%d,iter:%d, rank1:%f,rank5:%f,rank10:%f,rank20:%f\n',split_index,iter,rank_acc(1),rank_acc(2),rank_acc(3),rank_acc(4));
fprintf('split_index:%d,iter:%d, rank1:%f,rank5:%f,rank10:%f,rank20:%f\n',split_index,iter,rank_acc(1),rank_acc(2),rank_acc(3),rank_acc(4));
fclose(fin);