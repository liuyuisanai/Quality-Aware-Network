function [batch_data,batch_label]=get_train_minibatch(param,batch_data,batch_label,batch_size,train_data_cam1,train_data_cam2,label_train_cam1,label_train_cam2)
    data_an=[];
    data_pos=[];
    data_neg=[];
    select_view1_anr=randperm(2,1);
    select_view1_pos=randperm(2,1);
    select_view1_neg=randperm(2,1);
    person_sel=randperm(param.train_person_num,2);
    if select_view1_anr==1
        person_1_1=find(label_train_cam1==person_sel(1));
    else
        person_1_1=find(label_train_cam2==person_sel(1));
    end
    if select_view1_pos==1
        person_1_2=find(label_train_cam1==person_sel(1));
    else
        person_1_2=find(label_train_cam2==person_sel(1));
    end
    if select_view1_neg==1
        person_2=find(label_train_cam1==person_sel(2));
    else
        person_2=find(label_train_cam2==person_sel(2));
    end
    size_a=size(person_1_1);
    size_p=size(person_1_2);
    size_n=size(person_2);
    data_sel_anr=randperm(size_a(1),batch_size/3);
    data_sel_pos=randperm(size_p(1),batch_size/3);
    data_sel_neg=randperm(size_n(1),batch_size/3);
    for t=1:batch_size/3
        if select_view1_anr==1
            data_t=train_data_cam1(person_1_1(data_sel_anr(t)),:);
            batch_label(:,:,:,t)=label_train_cam1(person_1_1(data_sel_anr(t)))-1;
        else
            data_t=train_data_cam2(person_1_1(data_sel_anr(t)),:);
            batch_label(:,:,:,t)=label_train_cam2(person_1_1(data_sel_anr(t)))-1;
        end
        data_t=reshape(data_t,128,64,3);
        im_data=imresize((single(data_t)),[224 224]);
        im_data=im_data(:,:,[3,2,1]);
        im_data=permute(im_data,[2,1,3]);
        batch_data(:,:,:,t)=im_data;
        batch_data(:,:,1,t)=batch_data(:,:,1,t)-104;
        batch_data(:,:,2,t)=batch_data(:,:,2,t)-117;
        batch_data(:,:,3,t)=batch_data(:,:,3,t)-123;
        
        if select_view1_pos==1
            data_t=train_data_cam1(person_1_2(data_sel_pos(t)),:);
            batch_label(:,:,:,t+batch_size/3)=label_train_cam1(person_1_2(data_sel_pos(t)))-1;
        else
            data_t=train_data_cam2(person_1_2(data_sel_pos(t)),:);
            batch_label(:,:,:,t+batch_size/3)=label_train_cam2(person_1_2(data_sel_pos(t)))-1;
        end
        data_t=reshape(data_t,128,64,3);
        im_data=imresize((single(data_t)),[224 224]);
        im_data=im_data(:,:,[3,2,1]);
        im_data=permute(im_data,[2,1,3]);
        batch_data(:,:,:,t+batch_size/3)=im_data;
        batch_data(:,:,1,t+batch_size/3)=batch_data(:,:,1,t+batch_size/3)-104;
        batch_data(:,:,2,t+batch_size/3)=batch_data(:,:,2,t+batch_size/3)-117;
        batch_data(:,:,3,t+batch_size/3)=batch_data(:,:,3,t+batch_size/3)-123;
        
        if select_view1_neg==1
            data_t=train_data_cam1(person_2(data_sel_neg(t)),:);
            batch_label(:,:,:,t+2*batch_size/3)=label_train_cam1(person_2(data_sel_neg(t)))-1;
        else
            data_t=train_data_cam2(person_2(data_sel_neg(t)),:);
            batch_label(:,:,:,t+2*batch_size/3)=label_train_cam2(person_2(data_sel_neg(t)))-1;
        end
        data_t=reshape(data_t,128,64,3);
        im_data=imresize((single(data_t)),[224 224]);
        im_data=im_data(:,:,[3,2,1]);
        im_data=permute(im_data,[2,1,3]);
        batch_data(:,:,:,t+2*batch_size/3)=im_data;
        batch_data(:,:,1,t+2*batch_size/3)=batch_data(:,:,1,t+2*batch_size/3)-104;
        batch_data(:,:,2,t+2*batch_size/3)=batch_data(:,:,2,t+2*batch_size/3)-117;
        batch_data(:,:,3,t+2*batch_size/3)=batch_data(:,:,3,t+2*batch_size/3)-123;
    end
end