function         [batch_data,batch_label]=get_train_minibatch_for_LPW(batch_data,batch_label,batch_size, ...
            label_train_view1,label_train_view2,label_train_view3,label_train_view4, ...
        train_image_name_view1,train_image_name_view2,train_image_name_view3,train_image_name_view4);
    select_view1_anr=randperm(4,1);
    select_view1_pos=randperm(4,1);
    select_view1_neg=randperm(4,1);    
    if select_view1_anr==1
        if select_view1_pos==1
            anr_sel=label_train_view1;
        end
        if select_view1_pos==2
            anr_sel=intersect(label_train_view1,label_train_view2);
        end
        if select_view1_pos==3
            anr_sel=intersect(label_train_view1,label_train_view3);
        end
        if select_view1_pos==4
            anr_sel=intersect(label_train_view1,label_train_view4);
        end
    end
   if select_view1_anr==2
        if select_view1_pos==1
            anr_sel=intersect(label_train_view2,label_train_view1);
        end
        if select_view1_pos==2
            anr_sel=label_train_view2;
        end
        if select_view1_pos==3
            anr_sel=intersect(label_train_view2,label_train_view3);
        end
        if select_view1_pos==4
            anr_sel=intersect(label_train_view2,label_train_view4);
        end
   end
   if select_view1_anr==3
        if select_view1_pos==1
            anr_sel=intersect(label_train_view3,label_train_view1);;
        end
        if select_view1_pos==2
            anr_sel=intersect(label_train_view3,label_train_view2);
        end
        if select_view1_pos==3
            anr_sel=label_train_view3;
        end
        if select_view1_pos==4
            anr_sel=intersect(label_train_view3,label_train_view4);
        end
   end
   if select_view1_anr==4
        if select_view1_pos==1
            anr_sel=intersect(label_train_view4,label_train_view1);
        end
        if select_view1_pos==2
            anr_sel=intersect(label_train_view4,label_train_view2);
        end
        if select_view1_pos==3
            anr_sel=intersect(label_train_view4,label_train_view3);
        end
        if select_view1_pos==4
            anr_sel=label_train_view4;
        end
   end
   if select_view1_neg==1
        neg_sel=label_train_view1;
   end
   if select_view1_neg==2
        neg_sel=label_train_view2;
   end
   if select_view1_neg==3
        neg_sel=label_train_view3;
   end
   if select_view1_neg==4
        neg_sel=label_train_view4;
   end
   anr_sel=unique(anr_sel);
   neg_sel=unique(neg_sel);
   anr_sel_index=randperm(size(anr_sel,1),1);
   person_sel_anrlabel=anr_sel(anr_sel_index);
   neg_sel=setdiff(neg_sel,person_sel_anrlabel);
   neg_sel_index=randperm(size(neg_sel,1),1);
   person_sel_neglabel=neg_sel(neg_sel_index);
    if select_view1_anr==1
        person_1_1=find(label_train_view1==person_sel_anrlabel);
    end
    if select_view1_anr==2
        person_1_1=find(label_train_view2==person_sel_anrlabel);
    end
    if select_view1_anr==3
        person_1_1=find(label_train_view3==person_sel_anrlabel);
    end
    if select_view1_anr==4
        person_1_1=find(label_train_view4==person_sel_anrlabel);
    end
    if select_view1_pos==1
        person_1_2=find(label_train_view1==person_sel_anrlabel);
    end
    if select_view1_pos==2
        person_1_2=find(label_train_view2==person_sel_anrlabel);
    end
    if select_view1_pos==3
        person_1_2=find(label_train_view3==person_sel_anrlabel);
    end
    if select_view1_pos==4
        person_1_2=find(label_train_view4==person_sel_anrlabel);
    end
    if select_view1_neg==1
        person_2=find(label_train_view1==person_sel_neglabel);
    end
    if select_view1_neg==2
        person_2=find(label_train_view2==person_sel_neglabel);
    end
    if select_view1_neg==3
        person_2=find(label_train_view3==person_sel_neglabel);
    end
    if select_view1_neg==4
        person_2=find(label_train_view4==person_sel_neglabel);
    end
    size_a=size(person_1_1);
    size_p=size(person_1_2);
    size_n=size(person_2);
    data_sel_anr=randperm(size_a(1),batch_size/3);
    data_sel_pos=randperm(size_p(1),batch_size/3);
    data_sel_neg=randperm(size_n(1),batch_size/3);
    for t=1:batch_size/3
        if select_view1_anr==1
            data_t=train_image_name_view1{person_1_1(data_sel_anr(t))};
            batch_label(:,:,:,t)=label_train_view1(person_1_1(data_sel_anr(t)))-1;
        end
        if select_view1_anr==2
            data_t=train_image_name_view2{person_1_1(data_sel_anr(t))};
            batch_label(:,:,:,t)=label_train_view2(person_1_1(data_sel_anr(t)))-1;
        end
        if select_view1_anr==3
            data_t=train_image_name_view3{person_1_1(data_sel_anr(t))};
            batch_label(:,:,:,t)=label_train_view3(person_1_1(data_sel_anr(t)))-1;
        end
        if select_view1_anr==4
            data_t=train_image_name_view4{person_1_1(data_sel_anr(t))};
            batch_label(:,:,:,t)=label_train_view4(person_1_1(data_sel_anr(t)))-1;
        end
        data_t=imread(data_t);
        im_data=imresize((single(data_t)),[224 224]);
        im_data=im_data(:,:,[3,2,1]);
        im_data=permute(im_data,[2,1,3]);
        batch_data(:,:,:,t)=im_data;
        batch_data(:,:,1,t)=batch_data(:,:,1,t)-104;
        batch_data(:,:,2,t)=batch_data(:,:,2,t)-117;
        batch_data(:,:,3,t)=batch_data(:,:,3,t)-123;
        
        if select_view1_pos==1
            data_t=train_image_name_view1{person_1_2(data_sel_pos(t))};
            batch_label(:,:,:,t+batch_size/3)=label_train_view1(person_1_2(data_sel_pos(t)))-1;
        end
        if select_view1_pos==2
            data_t=train_image_name_view2{person_1_2(data_sel_pos(t))};
            batch_label(:,:,:,t+batch_size/3)=label_train_view2(person_1_2(data_sel_pos(t)))-1;
        end
        if select_view1_pos==3
            data_t=train_image_name_view3{person_1_2(data_sel_pos(t))};
            batch_label(:,:,:,t+batch_size/3)=label_train_view3(person_1_2(data_sel_pos(t)))-1;
        end
        if select_view1_pos==4
            data_t=train_image_name_view4{person_1_2(data_sel_pos(t))};
            batch_label(:,:,:,t+batch_size/3)=label_train_view4(person_1_2(data_sel_pos(t)))-1;
        end
        
        data_t=imread(data_t);
        im_data=imresize((single(data_t)),[224 224]);
        im_data=im_data(:,:,[3,2,1]);
        im_data=permute(im_data,[2,1,3]);
        batch_data(:,:,:,t+batch_size/3)=im_data;
        batch_data(:,:,1,t+batch_size/3)=batch_data(:,:,1,t+batch_size/3)-104;
        batch_data(:,:,2,t+batch_size/3)=batch_data(:,:,2,t+batch_size/3)-117;
        batch_data(:,:,3,t+batch_size/3)=batch_data(:,:,3,t+batch_size/3)-123;
        
        if select_view1_neg==1
            data_t=train_image_name_view1{person_2(data_sel_neg(t))};
            batch_label(:,:,:,t+2*batch_size/3)=label_train_view1(person_2(data_sel_neg(t)))-1;
        end
        if select_view1_neg==2
            data_t=train_image_name_view2{person_2(data_sel_neg(t))};
            batch_label(:,:,:,t+2*batch_size/3)=label_train_view2(person_2(data_sel_neg(t)))-1;
        end
        if select_view1_neg==3
            data_t=train_image_name_view3{person_2(data_sel_neg(t))};
            batch_label(:,:,:,t+2*batch_size/3)=label_train_view3(person_2(data_sel_neg(t)))-1;
        end
        if select_view1_neg==4
            data_t=train_image_name_view4{person_2(data_sel_neg(t))};
            batch_label(:,:,:,t+2*batch_size/3)=label_train_view4(person_2(data_sel_neg(t)))-1;
        end
        data_t=imread(data_t);
        im_data=imresize((single(data_t)),[224 224]);
        im_data=im_data(:,:,[3,2,1]);
        im_data=permute(im_data,[2,1,3]);
        batch_data(:,:,:,t+2*batch_size/3)=im_data;
        batch_data(:,:,1,t+2*batch_size/3)=batch_data(:,:,1,t+2*batch_size/3)-104;
        batch_data(:,:,2,t+2*batch_size/3)=batch_data(:,:,2,t+2*batch_size/3)-117;
        batch_data(:,:,3,t+2*batch_size/3)=batch_data(:,:,3,t+2*batch_size/3)-123;
    end
end