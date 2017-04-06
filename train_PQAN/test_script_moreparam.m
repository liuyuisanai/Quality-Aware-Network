%%test network
cam1_size=size(test_data_cam1);
cam2_size=size(test_data_cam2);
cam1_feature1=[];
test_batch_data=zeros(224,224,3,param.test_batch_size,'single');
for m=0:floor(cam1_size(1)/param.test_batch_size)-1
    for n=1:param.test_batch_size
        im_data=imresize((reshape(single(test_data_cam1(m*param.test_batch_size+n,:)),128,64,3)),[224 224]);
        im_data=im_data(:,:,[3,2,1]);
        im_data=permute(im_data,[2,1,3]);
        test_batch_data(:,:,:,n)=im_data;
        test_batch_data(:,:,1,n)=test_batch_data(:,:,1,n)-104;
        test_batch_data(:,:,2,n)=test_batch_data(:,:,2,n)-117;
        test_batch_data(:,:,3,n)=test_batch_data(:,:,3,n)-123;
    end
    net.blobs('data').set_data(test_batch_data);
    net.forward_prefilled;
    cam1_feature1=[cam1_feature1;(squeeze(net.blobs('normfeature1').get_data))'];
end
test_batch_data=zeros(224,224,3,param.test_batch_size,'single');
for m=(floor(cam1_size(1)/param.test_batch_size))*param.test_batch_size+1:cam1_size(1)
        count=m-(floor(cam1_size(1)/param.test_batch_size))*param.test_batch_size;
        im_data=imresize((reshape(single(test_data_cam1(m,:)),128,64,3)),[224 224]);
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
result1=(squeeze(net.blobs('normfeature1').get_data))';
cam1_feature1=[cam1_feature1;result1(1:index,:)];
prob_feature=[];
for m=1:param.test_person_num
    test_label_index=find(label_test_cam1==(m+param.test_person_num));
    a=sum(cam1_feature1(test_label_index,:))/size(test_label_index,1);
    prob_feature=[prob_feature;a];
end
%extract feature cam2
cam2_feature1=[];
test_batch_data=zeros(224,224,3,param.test_batch_size,'single');
for m=0:floor(cam2_size(1)/param.test_batch_size)-1
    for n=1:param.test_batch_size
        im_data=imresize((reshape(single(test_data_cam2(m*param.test_batch_size+n,:)),128,64,3)),[224 224]);
        im_data=im_data(:,:,[3,2,1]);
        im_data=permute(im_data,[2,1,3]);
        test_batch_data(:,:,:,n)=im_data;
        test_batch_data(:,:,1,n)=test_batch_data(:,:,1,n)-104;
        test_batch_data(:,:,2,n)=test_batch_data(:,:,2,n)-117;
        test_batch_data(:,:,3,n)=test_batch_data(:,:,3,n)-123;
    end
    net.blobs('data').set_data(test_batch_data);
    net.forward_prefilled;
    cam2_feature1=[cam2_feature1;(squeeze(net.blobs('normfeature1').get_data))'];
end
test_batch_data=zeros(224,224,3,param.test_batch_size,'single');
for m=(floor(cam2_size(1)/param.test_batch_size))*param.test_batch_size+1:cam2_size(1)
        count=m-(floor(cam2_size(1)/param.test_batch_size))*param.test_batch_size;
        im_data=imresize((reshape(single(test_data_cam2(m,:)),128,64,3)),[224 224]);
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
result1=(squeeze(net.blobs('normfeature1').get_data))';
cam2_feature1=[cam2_feature1;result1(1:index,:)];
gallery_feature=[];
for m=1:param.test_person_num
    test_label_index=find(label_test_cam2==(m+param.test_person_num));
    a=sum(cam2_feature1(test_label_index,:))/size(test_label_index,1);
    gallery_feature=[gallery_feature;a];
end
% cal cmc
prob_norm=bsxfun(@rdivide,prob_feature,sum(abs(prob_feature).^2,2).^(1/2));
gallery_norm=bsxfun(@rdivide,gallery_feature,sum(abs(gallery_feature).^2,2).^(1/2));
% [~,it]=sort(prob_norm*gallery_norm',2,'descend');
score_matrix=prob_norm*gallery_norm';
rank1_hit=0;
rank5_hit=0;
rank10_hit=0;
rank20_hit=0;
problen=size(prob_feature,1);
for m=1:problen
    [~,location]=sort(score_matrix(m,:),2,'descend');
    if find(location(1)==m)
        rank1_hit=rank1_hit+1;
    end
    if find(location(1:5)==m)
        rank5_hit=rank5_hit+1;
    end
    if find(location(1:10)==m)
        rank10_hit=rank10_hit+1;
    end
    if find(location(1:20)==m)
        rank20_hit=rank20_hit+1;
    end
end
rank_acc=[rank1_hit/problen,rank5_hit/problen,rank10_hit/problen,rank20_hit/problen];
fin=fopen(param.result_save_file,'a');
fprintf(fin,'split_index:%d,iter:%d, rank1:%f,rank5:%f,rank10:%f,rank20:%f\n',split_index,iter,rank_acc(1),rank_acc(2),rank_acc(3),rank_acc(4));
fprintf('split_index:%d,iter:%d, rank1:%f,rank5:%f,rank10:%f,rank20:%f\n',split_index,iter,rank_acc(1),rank_acc(2),rank_acc(3),rank_acc(4));
fclose(fin);