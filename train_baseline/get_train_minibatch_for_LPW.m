function [batch_data,batch_label]=get_train_minibatch_for_LPW(batch_data,batch_label,batch_size, ...
            label_train,train_image_name)
    data_size=size(label_train,1);
    data_sel=randperm(data_size,batch_size);
    batch_label(:)=label_train(data_sel,1)-1;
    for m=1:batch_size
        sel=data_sel(m);
        train_data=imread(train_image_name{sel});
        im_data=imresize((single(train_data)),[224 224]);
        im_data=im_data(:,:,[3,2,1]);
        im_data=permute(im_data,[2,1,3]);
        batch_data(:,:,:,m)=im_data;
        batch_data(:,:,1,m)=batch_data(:,:,1,m)-104;
        batch_data(:,:,2,m)=batch_data(:,:,2,m)-117;
        batch_data(:,:,3,m)=batch_data(:,:,3,m)-123;
    end
end