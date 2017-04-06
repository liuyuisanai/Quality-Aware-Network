function [batch_data,batch_label]=get_train_minibatch(batch_data,batch_label,batch_size,train_data_cam1,train_data_cam2,label_train_cam1,label_train_cam2)
    train_data=[train_data_cam1;train_data_cam2];
    train_labels=[label_train_cam1;label_train_cam2];
    data_size=size(train_data,1);
    data_sel=randperm(data_size,batch_size);
    batch_label(:)=train_labels(data_sel,1)-1;
    for m=1:batch_size
        sel=data_sel(m);
        im_data=imresize((reshape(single(train_data(sel,:)),128,64,3)),[224 224]);
        im_data=im_data(:,:,[3,2,1]);
        im_data=permute(im_data,[2,1,3]);
        batch_data(:,:,:,m)=im_data;
        batch_data(:,:,1,m)=batch_data(:,:,1,m)-104;
        batch_data(:,:,2,m)=batch_data(:,:,2,m)-117;
        batch_data(:,:,3,m)=batch_data(:,:,3,m)-123;
    end
end